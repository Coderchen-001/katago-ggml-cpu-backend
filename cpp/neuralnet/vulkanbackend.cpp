// Vulkan backend for KataGo.
//
// Runs neural net inference on any Vulkan-capable GPU, using the vendored ggml
// Vulkan backend (ggml-vulkan) as the kernel base. The model is a .onnx file
// dumped by 'katago dumponnx' (or a .bin.gz, which is converted to the same
// ONNX graph in memory at load time). At compute-handle creation the ONNX
// graph is interpreted into a ggml graph: every node maps to a ggml op, the
// dynamic batch dim is instantiated to maxBatchSize, and the resulting fixed
// graph is allocated on the Vulkan backend. Batch policy is FixedShape, so
// short batches are padded up to maxBatchSize (see onnxbackend.cpp).
//
// The IO tensor protocol is identical to the ONNX/TensorRT path: four NCHW
// float32 inputs declared in the order InputSpatial, InputGlobal, InputMeta,
// InputMask, and five NCHW float32 outputs OutputPolicyPass / OutputPolicy /
// OutputValue / OutputScoreValue / OutputOwnership, all raw logits.
// getOutput below reproduces the ONNX backend's post-processing exactly
// (per-row optimism blend, inverse-symmetry, version-branched score-value
// decode).

#ifdef USE_VULKAN_BACKEND

#include "../neuralnet/nninterface.h"
#include "../neuralnet/nneval.h"
#include "../neuralnet/nninputs.h"
#include "../neuralnet/modelversion.h"
#include "../neuralnet/onnxmodelbuilder.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-vulkan.h"
#include "ggml-cpu.h"

#include "onnx.pb.h"

#include <vulkan/vulkan.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <memory>
#include <mutex>
#include <set>
#include <unordered_map>
#include <vector>

using namespace std;

//--------------------------------------------------------------

struct LoadedModel {
  ModelDesc modelDesc;
  string modelFileName;
  bool isExternalOnnx;

  // For .onnx input: the load result (serialized bytes + build settings +
  // fp32 node lists). For .bin.gz input: empty, the graph is built per handle.
  OnnxModelBuilder::LoadResult externalOnnx;

  // Parsed ONNX model for .onnx input, shared read-only across handles.
  // For .bin.gz input the bytes are built per handle, so this is null.
  std::shared_ptr<const onnx::ModelProto> modelProto;

  // One-time scale8 transform (see maybeApplyScale8).
  mutable bool scale8Resolved;
  mutable bool scale8Applied;
  mutable std::mutex scale8Mutex;

  LoadedModel(const string& fileName, const string& expectedSha256)
    : modelFileName(fileName), isExternalOnnx(false), scale8Resolved(false), scale8Applied(false)
  {
    if(OnnxModelBuilder::isOnnxFileName(fileName)) {
      isExternalOnnx = true;
      externalOnnx = OnnxModelBuilder::load(fileName, expectedSha256, modelDesc, NULL);
      auto proto = std::make_shared<onnx::ModelProto>();
      if(!proto->ParseFromArray(externalOnnx.serializedModel.data(), (int)externalOnnx.serializedModel.size()))
        throw StringError("VULKAN backend: could not parse ONNX model file " + fileName);
      modelProto = proto;
      scale8Applied = externalOnnx.buildParams.scale8Applied;
    }
    else {
      ModelDesc::loadFromFileMaybeGZipped(fileName, modelDesc, expectedSha256);
    }
  }

  void maybeApplyScale8(bool skip, Logger* logger) const {
    std::lock_guard<std::mutex> lock(scale8Mutex);
    if(scale8Resolved)
      return;
    scale8Resolved = true;
    if(isExternalOnnx)
      return;
    if(!skip) {
      // Like the TensorRT backend: always apply the 1/8 activation rescaling
      // when the model is eligible (conv nets). Transformer nets are not
      // eligible and are left unchanged.
      scale8Applied = const_cast<LoadedModel*>(this)->modelDesc.applyScale8ToReduceActivations();
      if(scale8Applied && logger != NULL)
        logger->write("VULKAN backend: applied the scale8 activation rescaling to the model");
    }
  }

  LoadedModel() = delete;
  LoadedModel(const LoadedModel&) = delete;
  LoadedModel& operator=(const LoadedModel&) = delete;
};

LoadedModel* NeuralNet::loadModelFile(const string& file, const string& expectedSha256) {
  return new LoadedModel(file, expectedSha256);
}

void NeuralNet::freeLoadedModel(LoadedModel* loadedModel) {
  delete loadedModel;
}

const ModelDesc& NeuralNet::getModelDesc(const LoadedModel* loadedModel) {
  return loadedModel->modelDesc;
}

//--------------------------------------------------------------
// ONNX -> ggml graph interpreter.
//
// Walks the ONNX graph and builds a ggml computation graph. The ONNX batch
// dim ("batch", dynamic in the exported graphs) is instantiated to maxBatchSize
// at input creation; all ops are batch-generic so the rest of the graph is
// built from the actual tensor shapes. All tensors are FP32 in this phase.

struct OnnxToGgml {
  ggml_context* ctx;
  ggml_context* wctx;   // initializer (weight) tensors live here so gallocr
                        // never reuses their buffers; allocated separately.
  int maxBatchSize;

  const onnx::ModelProto& model;
  unordered_map<string, ggml_tensor*> tensors;

  // Host-side data of initializer tensors, set into device memory after graph
  // allocation.
  unordered_map<string, vector<uint8_t>> initData;

  // Name of the ONNX node currently being interpreted (diagnostics).
  string currentOpName;

  // One shared full-shape reference tensor per distinct target shape, reused
  // by ggml_repeat for broadcasts (repeat reads only the small operand; the
  // reference only supplies the shape, so a handful of them per handle is all
  // it takes and the transient repeat outputs reuse graph memory).
  unordered_map<uint64_t, ggml_tensor*> shapeRefs;

  OnnxToGgml(ggml_context* c, ggml_context* wc, int maxBatch, const onnx::ModelProto& m)
    : ctx(c), wctx(wc), maxBatchSize(maxBatch), model(m)
  {}

  ggml_tensor* shapeRefFor(const int64_t ne[4]) {
    uint64_t key = ((uint64_t)ne[0] << 48) | ((uint64_t)ne[1] << 32) | ((uint64_t)ne[2] << 16) | (uint64_t)ne[3];
    auto it = shapeRefs.find(key);
    if(it != shapeRefs.end())
      return it->second;
    ggml_tensor* t = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ne[0], ne[1], ne[2], ne[3]);
    ggml_set_input(t);
    shapeRefs[key] = t;
    return t;
  }

  // ---- initializer access ----

  const onnx::TensorProto* findInitializer(const string& name) const {
    for(const onnx::TensorProto& t : model.graph().initializer()) {
      if(t.name() == name)
        return &t;
    }
    return NULL;
  }

  static vector<float> initializerFloats(const onnx::TensorProto& t) {
    if(t.has_raw_data()) {
      const string& raw = t.raw_data();
      testAssert(raw.size() % 4 == 0);
      vector<float> v(raw.size() / 4);
      memcpy(v.data(), raw.data(), raw.size());
      return v;
    }
    if(t.float_data_size() > 0)
      return vector<float>(t.float_data().begin(), t.float_data().end());
    return vector<float>();
  }

  static vector<int64_t> initializerInt64(const onnx::TensorProto& t) {
    if(t.has_raw_data()) {
      const string& raw = t.raw_data();
      testAssert(raw.size() % 8 == 0);
      vector<int64_t> v(raw.size() / 8);
      memcpy(v.data(), raw.data(), raw.size());
      return v;
    }
    if(t.int64_data_size() > 0)
      return vector<int64_t>(t.int64_data().begin(), t.int64_data().end());
    return vector<int64_t>();
  }

  static void dimsToNe(const onnx::TensorProto& t, int64_t ne[4]) {
    ne[0] = ne[1] = ne[2] = ne[3] = 1;
    for(int i = 0; i < (int)t.dims_size() && i < 4; i++)
      ne[3 - i] = t.dims(i);
  }

  // Create (or fetch) a ggml tensor for an initializer, remembering its host
  // data for the post-allocation upload.
  ggml_tensor* initializerTensor(const string& name, ggml_type type) {
    auto it = tensors.find(name);
    if(it != tensors.end())
      return it->second;
    const onnx::TensorProto* t = findInitializer(name);
    if(t == NULL)
      throw StringError("VULKAN backend: initializer '" + name + "' not found in the ONNX graph");
    int64_t ne[4];
    dimsToNe(*t, ne);
    if(type == GGML_TYPE_F32) {
      vector<float> data = initializerFloats(*t);
      testAssert((int64_t)data.size() == ne[0] * ne[1] * ne[2] * ne[3]);
      vector<uint8_t> bytes(data.size() * 4);
      memcpy(bytes.data(), data.data(), bytes.size());
      initData[name] = std::move(bytes);
      ggml_tensor* tensor = ggml_new_tensor_4d(wctx, type, ne[0], ne[1], ne[2], ne[3]);
      ggml_set_name(tensor, name.c_str());
      // Weights live in the dedicated weight context (allocated separately
      // from the graph), so gallocr never reuses their buffers.
      tensors[name] = tensor;
      return tensor;
    }
    if(type == GGML_TYPE_I32) {
      vector<int64_t> data = initializerInt64(*t);
      testAssert((int64_t)data.size() == ne[0] * ne[1] * ne[2] * ne[3]);
      vector<uint8_t> bytes(data.size() * 4);
      int32_t* dst = (int32_t*)bytes.data();
      for(size_t i = 0; i < data.size(); i++)
        dst[i] = (int32_t)data[i];
      initData[name] = std::move(bytes);
      // Gather index tensors are always used as a flat 1D index list.
      ggml_tensor* tensor = ggml_new_tensor_4d(wctx, type, (int64_t)data.size(), 1, 1, 1);
      ggml_set_name(tensor, name.c_str());
      tensors[name] = tensor;
      return tensor;
    }
    throw StringError("VULKAN backend: unsupported initializer type for '" + name + "'");
  }

  // ---- shape helpers ----

  static int64_t onnxDimOf(ggml_tensor* t, int onnxIdx) {
    // ONNX dims are the reverse of ggml ne: dim 0 (N) is ne[3].
    return t->ne[3 - onnxIdx];
  }

  // Broadcast tensor t to a target shape expressed in ggml ne order via
  // ggml_repeat against a shared shape reference. If any dim differs and is
  // not 1, the shapes are incompatible.
  ggml_tensor* broadcastTo(ggml_tensor* t, const int64_t targetNe[4]) {
    bool needsBroadcast = false;
    for(int i = 0; i < 4; i++) {
      if(t->ne[i] == targetNe[i])
        continue;
      if(t->ne[i] == 1) {
        needsBroadcast = true;
        continue;
      }
      throw StringError("VULKAN backend: shape mismatch in broadcast");
    }
    if(!needsBroadcast)
      return t;
    return ggml_repeat(ctx, t, shapeRefFor(targetNe));
  }

  ggml_tensor* binary(ggml_tensor* a, ggml_tensor* b, int op) {
    // Elementwise op with broadcasting; both operands expanded to the
    // elementwise-max shape.
    int64_t targetNe[4];
    for(int i = 0; i < 4; i++)
      targetNe[i] = std::max(a->ne[i], b->ne[i]);
    ggml_tensor* ba = broadcastTo(a, targetNe);
    ggml_tensor* bb = broadcastTo(b, targetNe);
    switch(op) {
    case 0: return ggml_add(ctx, ba, bb);
    case 1: return ggml_mul(ctx, ba, bb);
    case 2: return ggml_div(ctx, ba, bb);
    case 3: return ggml_sub(ctx, ba, bb);
    default: throw StringError("VULKAN backend: unknown binary op");
    }
  }

  // ONNX MatMul: batched matrix multiply. A [bA..., M, K], B [bB..., K, N].
  // ggml_mul_mat is batched-aware: a must be [K, M, bA2, bA3] (A already is,
  // since K is its fastest dim), b must be [K, N, bB2, bB3] (a batched
  // transpose of B), and the raw result is [M, N, bB2, bB3] with M as the
  // fastest dim. ONNX lays out the result [..., M, N] with N fastest, i.e.
  // transposed relative to ggml, so we permute+cont to match ONNX memory
  // order before the result is consumed by subsequent ops.
  ggml_tensor* matmul(ggml_tensor* A, ggml_tensor* B) {
    const int64_t K = A->ne[0];
    if(B->ne[1] != K)
      throw StringError("VULKAN backend: MatMul contraction dim mismatch");
    ggml_tensor* bT = ggml_transpose(ctx, B);   // [K, N, bB2, bB3] view
    ggml_tensor* b = ggml_is_contiguous(bT) ? bT : ggml_cont(ctx, bT);
    ggml_tensor* r = ggml_mul_mat(ctx, A, b);   // [M, N, bB2, bB3], M fastest
    // Transpose to ONNX layout: [N, M, bB2, bB3] so the batch dims line up
    // and the spatial (N) dim is fastest like ONNX.
    ggml_tensor* p = ggml_permute(ctx, r, 1, 0, 2, 3);
    return ggml_cont(ctx, p);
  }

  // Reduce over the given ONNX axes (keepdims). isSum selects ReduceSum vs
  // ReduceMean; both reduce over the fastest dims of a permuted+contiguous
  // view so a single ggml_sum_rows / ggml_mean over ne0 does the job.
  ggml_tensor* reduceAxes(ggml_tensor* x, const vector<int64_t>& onnxAxes, bool isSum) {
    int64_t reduced = 1;
    int64_t rest = 1;
    ggml_tensor* flat = prepareReduction(x, onnxAxes, reduced, rest);
    ggml_tensor* red = isSum ? ggml_sum_rows(ctx, flat) : ggml_mean(ctx, flat);  // [1, rest]
    return reshapeReduced(flat, x, onnxAxes, red);
  }

  // ReduceMax over the given ONNX axes. ggml has no max-reduce, so reduce over
  // ne0 with a binary tree of pairwise maxes built from primitives:
  // max(a,b) = a + relu(b - a). O(log R) ops.
  ggml_tensor* reduceMaxAxes(ggml_tensor* x, const vector<int64_t>& onnxAxes) {
    int64_t reduced = 1;
    int64_t rest = 1;
    ggml_tensor* flat = prepareReduction(x, onnxAxes, reduced, rest);
    ggml_tensor* acc = flat;
    int64_t r = reduced;
    const size_t es = ggml_element_size(acc);
    while(r > 1) {
      int64_t half = r / 2;
      ggml_tensor* a1 = ggml_view_2d(ctx, acc, half, rest, acc->nb[1], 0);
      ggml_tensor* a2 = ggml_view_2d(ctx, acc, half, rest, acc->nb[1], half * es);
      ggml_tensor* m = ggml_add(ctx, a1, ggml_relu(ctx, ggml_sub(ctx, a2, a1)));
      if(r % 2 == 1) {
        ggml_tensor* odd = ggml_view_2d(ctx, acc, 1, rest, acc->nb[1], (r - 1) * es);
        acc = ggml_concat(ctx, m, odd, 0);
      }
      else {
        acc = m;
      }
      r = (r + 1) / 2;
    }
    return reshapeReduced(flat, x, onnxAxes, acc);
  }

  // Permute+cont+flatten so the reduced ONNX axes form ne0 of a [R, rest] view.
  ggml_tensor* prepareReduction(ggml_tensor* x, const vector<int64_t>& onnxAxes, int64_t& reduced, int64_t& rest) {
    testAssert(ggml_n_dims(x) <= 4);
    std::set<int> gAxes;
    for(int64_t a : onnxAxes) {
      testAssert(a >= 0 && a < 4);
      gAxes.insert(3 - (int)a);
    }
    int perm[4];
    // ggml_permute's args are the destination of each old dim (result->ne[p_i] = a->ne[i]):
    // reduced dims move to the front (in order), the rest follow.
    {
      int gCount = 0, rCount = 0;
      for(int d = 0; d < 4; d++) {
        if(gAxes.count(d))
          perm[d] = gCount++;
        else
          perm[d] = (int)gAxes.size() + rCount++;
      }
    }
    reduced = 1;
    for(int d : gAxes)
      reduced *= x->ne[d];
    testAssert(reduced > 0);
    rest = ggml_nelements(x) / reduced;
    ggml_tensor* p = ggml_permute(ctx, x, perm[0], perm[1], perm[2], perm[3]);
    ggml_tensor* pc = ggml_is_contiguous(p) ? p : ggml_cont(ctx, p);
    return ggml_reshape_2d(ctx, pc, reduced, rest);
  }

  // Reshape a [1, rest] reduction result back to the keep-dims shape.
  ggml_tensor* reshapeReduced(ggml_tensor* /*flat*/, ggml_tensor* x, const vector<int64_t>& onnxAxes, ggml_tensor* red) {
    std::set<int> gAxes;
    for(int64_t a : onnxAxes)
      gAxes.insert(3 - (int)a);
    int64_t outNe[4];
    for(int d = 0; d < 4; d++)
      outNe[d] = gAxes.count(d) ? 1 : x->ne[d];
    return ggml_reshape_4d(ctx, red, outNe[0], outNe[1], outNe[2], outNe[3]);
  }

  // ONNX Gather along the given axis. ggml_get_rows selects along ne1 of a 2D
  // tensor, so the gathered axis is moved to the innermost dim (ne3), the
  // tensor is flattened to 2D [rest, dim], gathered, and rearranged back.
  // NOTE: ggml_permute's args are the DESTINATION of each old dim (result->ne[p_i] = a->ne[i]).
  ggml_tensor* gather(ggml_tensor* x, ggml_tensor* idx, int64_t onnxAxis) {
    testAssert(onnxAxis >= 0 && onnxAxis < 4);
    const int g = 3 - (int)onnxAxis;
    // Move dim g to ne3, keeping the others in order.
    int fwd[4];
    for(int i = 0; i < 4; i++) {
      if(i == g)
        fwd[i] = 3;
      else {
        int c = 0;
        for(int j = 0; j < i; j++)
          if(j != g)
            c++;
        fwd[i] = c;
      }
    }
    ggml_tensor* p = ggml_permute(ctx, x, fwd[0], fwd[1], fwd[2], fwd[3]);
    ggml_tensor* pc = ggml_is_contiguous(p) ? p : ggml_cont(ctx, p);
    const int64_t gDim = x->ne[g];
    const int64_t rest = ggml_nelements(x) / gDim;
    ggml_tensor* flat = ggml_reshape_2d(ctx, pc, rest, gDim);  // [rest, g]
    ggml_tensor* r = ggml_get_rows(ctx, flat, idx);             // [rest, n_idx]
    // r's memory is in the permuted arrangement: reshape back to the permuted
    // ne (with the gathered dim replaced by the index count), then undo the
    // permutation and materialize.
    ggml_tensor* r4 = ggml_reshape_4d(ctx, r, pc->ne[0], pc->ne[1], pc->ne[2], idx->ne[0]);
    int bwd[4];
    for(int i = 0; i < 4; i++) {
      if(i == 3)
        bwd[i] = g;
      else
        bwd[i] = (i < g) ? i : i + 1;
    }
    return ggml_cont(ctx, ggml_permute(ctx, r4, bwd[0], bwd[1], bwd[2], bwd[3]));
  }

  ggml_tensor* transposeOp(ggml_tensor* x, const vector<int64_t>& perm) {
    testAssert(perm.size() == 4);
    int p[4];
    for(int j = 0; j < 4; j++)
      p[j] = (int)(3 - perm[3 - j]);
    bool identity = true;
    for(int j = 0; j < 4; j++)
      if(p[j] != j)
        identity = false;
    if(identity)
      return x;
    ggml_tensor* v = ggml_permute(ctx, x, p[0], p[1], p[2], p[3]);
    return ggml_cont(ctx, v);
  }

  ggml_tensor* reshapeOp(ggml_tensor* x, const vector<int64_t>& target) {
    int64_t dims[4] = {1, 1, 1, 1};
    int64_t known = 1;
    int inferIdx = -1;
    for(int i = 0; i < (int)target.size() && i < 4; i++) {
      if(target[i] == 0) {
        dims[i] = onnxDimOf(x, i);
        known *= dims[i];
      }
      else if(target[i] == -1) {
        inferIdx = i;
      }
      else {
        dims[i] = target[i];
        known *= dims[i];
      }
    }
    if(inferIdx >= 0)
      dims[inferIdx] = ggml_nelements(x) / known;
    testAssert(ggml_nelements(x) == dims[0] * dims[1] * dims[2] * dims[3]);
    return ggml_reshape_4d(ctx, x, dims[3], dims[2], dims[1], dims[0]);
  }

  // ---- node attributes ----

  static bool hasAttr(const onnx::NodeProto& node, const char* name) {
    for(const onnx::AttributeProto& a : node.attribute())
      if(a.name() == name)
        return true;
    return false;
  }

  static const onnx::AttributeProto& getAttr(const onnx::NodeProto& node, const char* name) {
    for(const onnx::AttributeProto& a : node.attribute())
      if(a.name() == name)
        return a;
    throw StringError(string("VULKAN backend: missing ONNX attribute '") + name + "' on op " + node.op_type());
  }

  // ---- main loop ----

  void build(const vector<pair<string, ggml_tensor*>>& inputs) {
    for(auto& kv : inputs)
      tensors[kv.first] = kv.second;

    for(const onnx::NodeProto& node : model.graph().node()) {
      const string& op = node.op_type();
      const string& outName = node.output_size() > 0 ? node.output(0) : string();
      testAssert(node.output_size() == 1);
      currentOpName = node.name().empty() ? op : node.name();

      auto in = [&](int i) -> ggml_tensor* {
        testAssert(i < node.input_size());
        const string& name = node.input(i);
        if(name.empty())
          throw StringError("VULKAN backend: op " + op + " is missing an optional input the interpreter requires");
        auto it = tensors.find(name);
        if(it != tensors.end())
          return it->second;
        // Inputs that are initializers (weights, scalars, RoPE tables) are
        // materialized as F32 tensors on first use.
        if(findInitializer(name) != NULL)
          return initializerTensor(name, GGML_TYPE_F32);
        throw StringError("VULKAN backend: tensor '" + name + "' used by op " + op + " was not produced yet (graph not topologically ordered?)");
      };
      auto registerOut = [&](ggml_tensor* t) {
        tensors[outName] = t;
      };

      if(op == "Conv") {
        ggml_tensor* x = in(0);
        ggml_tensor* w = initializerTensor(node.input(1), GGML_TYPE_F32);
        // 1x1 and odd-kernel stride-1 same-pad convs map to ggml_conv_2d_s1_ph.
        // Kernel a is [kW, kH, inC, outC] (same bytes as ONNX [outC,inC,kH,kW]).
        int64_t kY = w->ne[1], kX = w->ne[0];
        testAssert(kY == kX && (kY % 2) == 1);
        if(hasAttr(node, "strides")) {
          const onnx::AttributeProto& s = getAttr(node, "strides");
          testAssert(s.ints_size() == 2 && s.ints(0) == 1 && s.ints(1) == 1);
        }
        if(hasAttr(node, "dilations")) {
          const onnx::AttributeProto& d = getAttr(node, "dilations");
          testAssert(d.ints_size() == 2 && d.ints(0) == 1 && d.ints(1) == 1);
        }
        if(hasAttr(node, "group")) {
          testAssert(getAttr(node, "group").i() == 1);
        }
        if(hasAttr(node, "pads")) {
          const onnx::AttributeProto& p = getAttr(node, "pads");
          testAssert(p.ints_size() == 4);
          testAssert(p.ints(0) == (kY - 1) / 2 && p.ints(1) == (kX - 1) / 2);
          testAssert(p.ints(2) == (kY - 1) / 2 && p.ints(3) == (kX - 1) / 2);
        }
        registerOut(ggml_conv_2d_s1_ph(ctx, w, x));
      }
      else if(op == "MatMul") {
        registerOut(matmul(in(0), in(1)));
      }
      else if(op == "Add") {
        registerOut(binary(in(0), in(1), 0));
      }
      else if(op == "Mul") {
        registerOut(binary(in(0), in(1), 1));
      }
      else if(op == "Div") {
        registerOut(binary(in(0), in(1), 2));
      }
      else if(op == "Sub") {
        registerOut(binary(in(0), in(1), 3));
      }
      else if(op == "Relu") {
        registerOut(ggml_relu(ctx, in(0)));
      }
      else if(op == "Sigmoid") {
        registerOut(ggml_sigmoid(ctx, in(0)));
      }
      else if(op == "Tanh") {
        registerOut(ggml_tanh(ctx, in(0)));
      }
      else if(op == "Softplus") {
        registerOut(ggml_softplus(ctx, in(0)));
      }
      else if(op == "Sqrt") {
        registerOut(ggml_sqrt(ctx, in(0)));
      }
      else if(op == "Identity") {
        registerOut(in(0));
      }
      else if(op == "Pow") {
        // Only Pow(x, const 2) is used (RMSNorm squaring).
        const onnx::TensorProto* e = findInitializer(node.input(1));
        testAssert(e != NULL);
        vector<float> expo = initializerFloats(*e);
        testAssert(expo.size() == 1 && expo[0] == 2.0f);
        ggml_tensor* x = in(0);
        registerOut(ggml_mul(ctx, x, x));
      }
      else if(op == "ReduceMean" || op == "ReduceSum" || op == "ReduceMax") {
        const onnx::TensorProto* axes = findInitializer(node.input(1));
        testAssert(axes != NULL);
        vector<int64_t> ax = initializerInt64(*axes);
        if(op == "ReduceMax")
          registerOut(reduceMaxAxes(in(0), ax));
        else
          registerOut(reduceAxes(in(0), ax, op == "ReduceSum"));
      }
      else if(op == "Gather") {
        int64_t axis = 0;
        if(hasAttr(node, "axis"))
          axis = getAttr(node, "axis").i();
        ggml_tensor* idx = initializerTensor(node.input(1), GGML_TYPE_I32);
        registerOut(gather(in(0), idx, axis));
      }
      else if(op == "Transpose") {
        vector<int64_t> perm;
        if(hasAttr(node, "perm")) {
          const onnx::AttributeProto& p = getAttr(node, "perm");
          for(int i = 0; i < p.ints_size(); i++)
            perm.push_back(p.ints(i));
        }
        else {
          for(int i = 0; i < 4; i++)
            perm.push_back(3 - i);
        }
        registerOut(transposeOp(in(0), perm));
      }
      else if(op == "Reshape") {
        const onnx::TensorProto* shape = findInitializer(node.input(1));
        testAssert(shape != NULL);
        registerOut(reshapeOp(in(0), initializerInt64(*shape)));
      }
      else if(op == "Concat") {
        int64_t axis = getAttr(node, "axis").i();
        testAssert(axis >= 0 && axis < 4);
        int ggmlAxis = 3 - (int)axis;
        ggml_tensor* acc = in(0);
        for(int i = 1; i < node.input_size(); i++)
          acc = ggml_concat(ctx, acc, in(i), ggmlAxis);
        registerOut(acc);
      }
      else if(op == "Softmax") {
        int64_t axis = getAttr(node, "axis").i();
        if(axis != 3)
          throw StringError("VULKAN backend: Softmax is only supported on the last ONNX axis");
        registerOut(ggml_soft_max(ctx, in(0)));
      }
      else {
        throw StringError("VULKAN backend: unsupported ONNX op '" + op + "'");
      }
    }
  }
};

//--------------------------------------------------------------

struct ComputeContext {
  int nnXLen;
  int nnYLen;

  ComputeContext(int xLen, int yLen)
    : nnXLen(xLen), nnYLen(yLen)
  {}

  ComputeContext() = delete;
  ComputeContext(const ComputeContext&) = delete;
  ComputeContext& operator=(const ComputeContext&) = delete;
};

ComputeContext* NeuralNet::createComputeContext(
  const std::vector<int>& gpuIdxs,
  Logger* logger,
  int nnXLen,
  int nnYLen,
  const string& homeDataDirOverride,
  enabled_t useFP16Mode,
  const LoadedModel* loadedModel,
  ConfigParser& cfg
) {
  (void)gpuIdxs;
  (void)homeDataDirOverride;
  (void)useFP16Mode;
  (void)cfg;

  // scale8 (1/8 activation rescaling) is applied here, before the graph is
  // built, exactly like the ONNX/TensorRT backends do. Transformer models are
  // not eligible and are skipped automatically.
  loadedModel->maybeApplyScale8(false, logger);

  ComputeContext* ctx = new ComputeContext(nnXLen, nnYLen);
  if(logger != NULL)
    logger->write("VULKAN backend: creating compute context for " +
                  Global::intToString(nnXLen) + "x" + Global::intToString(nnYLen));
  return ctx;
}

void NeuralNet::freeComputeContext(ComputeContext* computeContext) {
  delete computeContext;
}

//--------------------------------------------------------------

struct ComputeHandle {
  ComputeContext* ctx;
  const LoadedModel* loadedModel;
  int maxBatchSize;

  ggml_backend_t backend;
  ggml_context* ggmlCtx;
  ggml_context* wctx;   // initializer tensors, allocated separately from the graph
  ggml_cgraph* graph;
  ggml_gallocr_t gallocr;
  ggml_backend_buffer_t wbuf;

  ggml_tensor* inputSpatial;
  ggml_tensor* inputGlobal;
  ggml_tensor* inputMeta;
  ggml_tensor* inputMask;

  ggml_tensor* outPolicyPass;
  ggml_tensor* outPolicy;
  ggml_tensor* outValue;
  ggml_tensor* outScoreValue;
  ggml_tensor* outOwnership;

  bool padBatch;

  ComputeHandle(ComputeContext* context, const LoadedModel* loaded, Logger* logger,
                int maxBatch, bool requireExactNNLen, int deviceIdx)
    : ctx(context),
      loadedModel(loaded),
      maxBatchSize(maxBatch),
      backend(NULL),
      ggmlCtx(NULL),
      graph(NULL),
      gallocr(NULL),
      inputSpatial(NULL),
      inputGlobal(NULL),
      inputMeta(NULL),
      inputMask(NULL),
      outPolicyPass(NULL),
      outPolicy(NULL),
      outValue(NULL),
      outScoreValue(NULL),
      outOwnership(NULL),
      padBatch(true)
  {
    (void)requireExactNNLen;

    // Debug cross-check: KATAGO_VULKAN_CPU=1 runs the same graph on the ggml
    // CPU backend instead of Vulkan (validates the interpreter in isolation).
    if(getenv("KATAGO_VULKAN_CPU") != nullptr && getenv("KATAGO_VULKAN_CPU")[0] == '1') {
      backend = ggml_backend_cpu_init();
      if(backend == NULL)
        throw StringError("VULKAN backend: could not initialize the ggml CPU backend");
    }
    else {
      backend = ggml_backend_vk_init(deviceIdx >= 0 ? (size_t)deviceIdx : 0);
      if(backend == NULL)
        throw StringError("VULKAN backend: could not initialize Vulkan backend for device " + Global::intToString(deviceIdx));
    }

    // --- Get the ONNX bytes (external .onnx shared, .bin.gz built here) ---
    const string* onnxBytesPtr = NULL;
    string builtOnnxBytes;
    std::shared_ptr<const onnx::ModelProto> sharedProto;
    if(loadedModel->isExternalOnnx) {
      onnxBytesPtr = &loadedModel->externalOnnx.serializedModel;
      sharedProto = loadedModel->modelProto;
    }
    else {
      OnnxModelBuilder::BuildParams buildParams;
      buildParams.nnXLen = ctx->nnXLen;
      buildParams.nnYLen = ctx->nnYLen;
      buildParams.requireExactNNLen = false;
      buildParams.transformerNHWC = false;
      buildParams.scale8Applied = loadedModel->scale8Applied;
      if(logger != NULL)
        logger->write("VULKAN backend: building ONNX graph from model weights...");
      OnnxModelBuilder::Result onnxResult = OnnxModelBuilder::build(loadedModel->modelDesc, buildParams, logger);
      builtOnnxBytes = onnxResult.serializedModel;
      onnxBytesPtr = &builtOnnxBytes;
      auto proto = std::make_shared<onnx::ModelProto>();
      if(!proto->ParseFromArray(onnxBytesPtr->data(), (int)onnxBytesPtr->size()))
        throw StringError("VULKAN backend: could not parse the built ONNX graph");
      sharedProto = proto;
    }
    const string& onnxBytes = *onnxBytesPtr;

    const onnx::ModelProto& model = *sharedProto;
    const onnx::GraphProto& graphP = model.graph();

    // --- Size the ggml contexts ---
    // The interpreter expands each ONNX op into several ggml ops (broadcast
    // repeats, transposes+conts, the ReduceMax tree), so size generously.
    const size_t nInitializers = (size_t)graphP.initializer_size();
    const size_t nNodes = (size_t)graphP.node_size();
    const size_t graphSize = std::max(nNodes * 5 + 1024, nInitializers * 2 + 1024);
    const size_t nTensors = nNodes * 8 + nInitializers + 8192;
    ggml_init_params params;
    params.mem_size = ggml_tensor_overhead() * nTensors + ggml_graph_overhead_custom(graphSize, false) + 1024 * 1024;
    params.mem_buffer = NULL;
    params.no_alloc = true;
    ggmlCtx = ggml_init(params);
    if(ggmlCtx == NULL)
      throw StringError("VULKAN backend: ggml_init failed");

    // Dedicated context for initializer tensors. It is allocated with
    // ggml_backend_alloc_ctx_tensors (not gallocr), so the weights keep their
    // own buffers for the lifetime of the handle and gallocr never reuses them
    // for intermediates.
    {
      size_t weightBytes = 0;
      for(const onnx::TensorProto& t : graphP.initializer()) {
        size_t n = 1;
        for(int i = 0; i < t.dims_size(); i++)
          n *= (size_t)t.dims(i);
        weightBytes += n * 4;
      }
      ggml_init_params wparams;
      wparams.mem_size = ggml_tensor_overhead() * (nInitializers + 64) + weightBytes / 16 + 65536;
      wparams.mem_buffer = NULL;
      wparams.no_alloc = true;
      wctx = ggml_init(wparams);
      if(wctx == NULL)
        throw StringError("VULKAN backend: ggml_init(wctx) failed");
    }

    // --- Create graph inputs (NCHW, batch = maxBatchSize) ---
    OnnxToGgml builder(ggmlCtx, wctx, maxBatchSize, model);
    vector<pair<string, ggml_tensor*>> inputTensors;

    // Read the declared input shapes from the graph.
    for(const onnx::ValueInfoProto& vi : graphP.input()) {
      const string& name = vi.name();
      const onnx::TensorShapeProto& shape = vi.type().tensor_type().shape();
      int64_t dims[4] = {1, 1, 1, 1};
      int d = 0;
      for(const onnx::TensorShapeProto_Dimension& dim : shape.dim()) {
        if(d >= 4)
          break;
        if(dim.has_dim_value())
          dims[d] = dim.dim_value();
        else
          dims[d] = maxBatchSize;  // the dynamic "batch" dim
        d++;
      }
      ggml_tensor* t = ggml_new_tensor_4d(ggmlCtx, GGML_TYPE_F32, dims[3], dims[2], dims[1], dims[0]);
      ggml_set_input(t);
      inputTensors.push_back({name, t});
      if(name == "InputSpatial") inputSpatial = t;
      else if(name == "InputGlobal") inputGlobal = t;
      else if(name == "InputMeta") inputMeta = t;
      else if(name == "InputMask") inputMask = t;
    }
    if(inputSpatial == NULL || inputGlobal == NULL || inputMask == NULL)
      throw StringError("VULKAN backend: graph is missing expected inputs InputSpatial/InputGlobal/InputMask");

    builder.build(inputTensors);

    // --- Resolve outputs ---
    auto getOut = [&](const char* name) -> ggml_tensor* {
      auto it = builder.tensors.find(name);
      if(it == builder.tensors.end())
        throw StringError(string("VULKAN backend: graph is missing expected output ") + name);
      return it->second;
    };
    outPolicyPass = getOut("OutputPolicyPass");
    outPolicy = getOut("OutputPolicy");
    outValue = getOut("OutputValue");
    outScoreValue = getOut("OutputScoreValue");
    outOwnership = getOut("OutputOwnership");
    ggml_set_output(outPolicyPass);
    ggml_set_output(outPolicy);
    ggml_set_output(outValue);
    ggml_set_output(outScoreValue);
    ggml_set_output(outOwnership);

    // --- Build the cgraph ---
    graph = ggml_new_graph_custom(ggmlCtx, graphSize, false);
    ggml_build_forward_expand(graph, outPolicyPass);
    ggml_build_forward_expand(graph, outPolicy);
    ggml_build_forward_expand(graph, outValue);
    ggml_build_forward_expand(graph, outScoreValue);
    ggml_build_forward_expand(graph, outOwnership);

    // --- Allocate and upload weights ---
    // Weights get their own buffers first (either per-tensor in the debug
    // NOALLOC mode or one dedicated buffer in the normal path), so gallocr can
    // never overwrite them when it reuses intermediate buffers.
    if(getenv("KATAGO_VULKAN_NOALLOC") != nullptr && getenv("KATAGO_VULKAN_NOALLOC")[0] == '1') {
      // Debug: allocate every tensor separately (no buffer reuse) to rule out
      // gallocr liveness/aliasing issues. Allocate the weight context first so
      // views of weight tensors in the main context can resolve their buffers.
      wbuf = ggml_backend_alloc_ctx_tensors(wctx, backend);
      if(wbuf == NULL)
        throw StringError("VULKAN backend: weight ctx allocation failed");
      if(ggml_backend_alloc_ctx_tensors(ggmlCtx, backend) == NULL)
        throw StringError("VULKAN backend: ctx tensor allocation failed");
    }
    else {
      wbuf = ggml_backend_alloc_ctx_tensors(wctx, backend);
      if(wbuf == NULL)
        throw StringError("VULKAN backend: weight ctx allocation failed");
      gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
      if(!ggml_gallocr_alloc_graph(gallocr, graph))
        throw StringError("VULKAN backend: graph allocation failed (out of device memory?)");
    }
    for(auto& kv : builder.initData) {
      ggml_tensor* t = builder.tensors[kv.first];
      ggml_backend_tensor_set(t, kv.second.data(), 0, kv.second.size());
    }

    if(logger != NULL)
      logger->write("VULKAN backend: graph built (" + Global::uint64ToString((uint64_t)onnxBytes.size()) + " onnx bytes, " +
                    Global::uint64ToString((uint64_t)ggml_graph_n_nodes(graph)) + " ggml nodes)");
  }

  ~ComputeHandle() {
    if(wbuf != NULL)
      ggml_backend_buffer_free(wbuf);
    if(gallocr != NULL)
      ggml_gallocr_free(gallocr);
    if(backend != NULL)
      ggml_backend_free(backend);
    if(wctx != NULL)
      ggml_free(wctx);
    if(ggmlCtx != NULL)
      ggml_free(ggmlCtx);
  }

  ComputeHandle() = delete;
  ComputeHandle(const ComputeHandle&) = delete;
  ComputeHandle& operator=(const ComputeHandle&) = delete;
};

ComputeHandle* NeuralNet::createComputeHandle(
  ComputeContext* context,
  const LoadedModel* loadedModel,
  Logger* logger,
  int maxBatchSize,
  bool requireExactNNLen,
  bool inputsUseNHWC,
  int gpuIdxForThisThread,
  int serverThreadIdx
) {
  if(inputsUseNHWC)
    throw StringError("VULKAN backend: inputsUseNHWC = true not supported, must use NCHW");

  if(logger != NULL)
    logger->write("VULKAN backend thread " + Global::intToString(serverThreadIdx) +
                  ": Model version " + Global::intToString(loadedModel->modelDesc.modelVersion));

  try {
    return new ComputeHandle(context, loadedModel, logger, maxBatchSize, requireExactNNLen, gpuIdxForThisThread);
  }
  catch(const std::exception& e) {
    // nn-server threads do not catch, so an escaping exception would become a
    // silent fastfail on Windows; log the cause first.
    string msg = string("VULKAN backend: compute handle creation failed: ") + e.what();
    if(logger != NULL)
      logger->write(msg);
    cerr << msg << endl;
    throw;
  }
}

void NeuralNet::freeComputeHandle(ComputeHandle* computeHandle) {
  delete computeHandle;
}

bool NeuralNet::isUsingFP16(const ComputeHandle* handle) {
  (void)handle;
  return false;
}

bool NeuralNet::setIsWarmup(const ComputeHandle* handle, bool isWarmup) {
  (void)handle;
  (void)isWarmup;
  return false;
}

//--------------------------------------------------------------

struct InputBuffers {
  int maxBatchSize;

  size_t singleMaskElts;
  size_t singleInputElts;
  size_t singleInputGlobalElts;
  size_t singleInputMetaElts;

  size_t singlePolicyPassResultElts;
  size_t singlePolicyResultElts;
  size_t singleValueResultElts;
  size_t singleScoreValueResultElts;
  size_t singleOwnershipResultElts;

  vector<float> maskInput;
  vector<float> spatialInput;
  vector<float> globalInput;
  vector<float> metaInput;

  InputBuffers(const LoadedModel* loadedModel, int maxBatchSz, int nnXLen, int nnYLen) {
    const ModelDesc& m = loadedModel->modelDesc;

    if(nnXLen > NNPos::MAX_BOARD_LEN)
      throw StringError(
        Global::strprintf("nnXLen (%d) is greater than NNPos::MAX_BOARD_LEN (%d)", nnXLen, NNPos::MAX_BOARD_LEN));
    if(nnYLen > NNPos::MAX_BOARD_LEN)
      throw StringError(
        Global::strprintf("nnYLen (%d) is greater than NNPos::MAX_BOARD_LEN (%d)", nnYLen, NNPos::MAX_BOARD_LEN));

    maxBatchSize = maxBatchSz;
    singleMaskElts = (size_t)nnXLen * nnYLen;
    singleInputElts = (size_t)m.numInputChannels * nnXLen * nnYLen;
    singleInputGlobalElts = (size_t)m.numInputGlobalChannels;
    singleInputMetaElts = (size_t)m.numInputMetaChannels;
    singlePolicyPassResultElts = (size_t)m.numPolicyChannels;
    singlePolicyResultElts = (size_t)m.numPolicyChannels * nnXLen * nnYLen;
    singleValueResultElts = (size_t)m.numValueChannels;
    singleScoreValueResultElts = (size_t)m.numScoreValueChannels;
    singleOwnershipResultElts = (size_t)m.numOwnershipChannels * nnXLen * nnYLen;

    testAssert(NNModelVersion::getNumSpatialFeatures(m.modelVersion) == m.numInputChannels);
    testAssert(NNModelVersion::getNumGlobalFeatures(m.modelVersion) == m.numInputGlobalChannels);
    if(m.numInputMetaChannels > 0)
      testAssert(SGFMetadata::METADATA_INPUT_NUM_CHANNELS == m.numInputMetaChannels);

    maskInput.assign(singleMaskElts * maxBatchSize, 0.0f);
    spatialInput.assign(singleInputElts * maxBatchSize, 0.0f);
    globalInput.assign(singleInputGlobalElts * maxBatchSize, 0.0f);
    if(singleInputMetaElts > 0)
      metaInput.assign(singleInputMetaElts * maxBatchSize, 0.0f);
  }

  InputBuffers() = delete;
  InputBuffers(const InputBuffers&) = delete;
  InputBuffers& operator=(const InputBuffers&) = delete;
};

InputBuffers* NeuralNet::createInputBuffers(const LoadedModel* loadedModel, int maxBatchSize, int nnXLen, int nnYLen) {
  return new InputBuffers(loadedModel, maxBatchSize, nnXLen, nnYLen);
}

void NeuralNet::freeInputBuffers(InputBuffers* inputBuffers) {
  delete inputBuffers;
}

//--------------------------------------------------------------

void NeuralNet::globalInitialize() {
}

void NeuralNet::globalCleanup() {
}

//--------------------------------------------------------------

void NeuralNet::getOutput(
  ComputeHandle* gpuHandle,
  InputBuffers* inputBuffers,
  int numBatchEltsFilled,
  NNResultBuf** inputBufs,
  vector<NNOutput*>& outputs
) {
  assert(numBatchEltsFilled <= inputBuffers->maxBatchSize);
  assert(numBatchEltsFilled > 0);

  const int batchSize = numBatchEltsFilled;
  // The ggml graph is fixed at maxBatchSize (FixedShape policy), so short
  // batches are padded with a copy of row 0, exactly like onnxbackend.cpp.
  const int runBatchSize = inputBuffers->maxBatchSize;
  const int nnXLen = gpuHandle->ctx->nnXLen;
  const int nnYLen = gpuHandle->ctx->nnYLen;
  const int modelVersion = gpuHandle->loadedModel->modelDesc.modelVersion;

  const int numSpatialFeatures = NNModelVersion::getNumSpatialFeatures(modelVersion);
  const int numGlobalFeatures = NNModelVersion::getNumGlobalFeatures(modelVersion);
  const int numMetaFeatures = (int)inputBuffers->singleInputMetaElts;
  assert(numSpatialFeatures * nnXLen * nnYLen == inputBuffers->singleInputElts);
  assert(numGlobalFeatures == inputBuffers->singleInputGlobalElts);

  // Fill host input buffers, mirroring the ONNX backend exactly:
  //  - global / meta are straight copies (no symmetry)
  //  - spatial is symmetry-transformed (NCHW, useNHWC=false)
  //  - mask = channel 0 of the symmetry-transformed spatial input
  for(int nIdx = 0; nIdx < batchSize; nIdx++) {
    float* rowMaskInput = inputBuffers->maskInput.data() + inputBuffers->singleMaskElts * nIdx;
    float* rowSpatialInput = inputBuffers->spatialInput.data() + inputBuffers->singleInputElts * nIdx;
    float* rowGlobalInput = inputBuffers->globalInput.data() + inputBuffers->singleInputGlobalElts * nIdx;

    const float* rowGlobal = inputBufs[nIdx]->rowGlobalBuf.data();
    const float* rowSpatial = inputBufs[nIdx]->rowSpatialBuf.data();
    std::copy(rowGlobal, rowGlobal + numGlobalFeatures, rowGlobalInput);
    SymmetryHelpers::copyInputsWithSymmetry(
      rowSpatial, rowSpatialInput, 1, nnYLen, nnXLen, numSpatialFeatures, false, inputBufs[nIdx]->symmetry);
    std::copy(rowSpatialInput, rowSpatialInput + inputBuffers->singleMaskElts, rowMaskInput);

    if(numMetaFeatures > 0) {
      float* rowMetaInput = inputBuffers->metaInput.data() + inputBuffers->singleInputMetaElts * nIdx;
      const float* rowMeta = inputBufs[nIdx]->rowMetaBuf.data();
      testAssert(inputBufs[nIdx]->hasRowMeta);
      std::copy(rowMeta, rowMeta + numMetaFeatures, rowMetaInput);
    }
    else {
      testAssert(!inputBufs[nIdx]->hasRowMeta);
    }
  }

  // Fill the padding rows with a copy of row 0 (their outputs are discarded,
  // but an all-zero row would break the mask-weighted pooling divisions).
  for(int nIdx = batchSize; nIdx < runBatchSize; nIdx++) {
    std::copy(
      inputBuffers->maskInput.data(), inputBuffers->maskInput.data() + inputBuffers->singleMaskElts,
      inputBuffers->maskInput.data() + inputBuffers->singleMaskElts * nIdx);
    std::copy(
      inputBuffers->spatialInput.data(), inputBuffers->spatialInput.data() + inputBuffers->singleInputElts,
      inputBuffers->spatialInput.data() + inputBuffers->singleInputElts * nIdx);
    std::copy(
      inputBuffers->globalInput.data(), inputBuffers->globalInput.data() + inputBuffers->singleInputGlobalElts,
      inputBuffers->globalInput.data() + inputBuffers->singleInputGlobalElts * nIdx);
    if(numMetaFeatures > 0)
      std::copy(
        inputBuffers->metaInput.data(), inputBuffers->metaInput.data() + inputBuffers->singleInputMetaElts,
        inputBuffers->metaInput.data() + inputBuffers->singleInputMetaElts * nIdx);
  }

  // Upload inputs to the device tensors (NCHW memory matches ggml's ne order
  // exactly, so the host buffers copy straight across).
  ggml_backend_tensor_set(gpuHandle->inputSpatial, inputBuffers->spatialInput.data(), 0,
                          inputBuffers->singleInputElts * runBatchSize * sizeof(float));
  ggml_backend_tensor_set(gpuHandle->inputGlobal, inputBuffers->globalInput.data(), 0,
                          inputBuffers->singleInputGlobalElts * runBatchSize * sizeof(float));
  ggml_backend_tensor_set(gpuHandle->inputMask, inputBuffers->maskInput.data(), 0,
                          inputBuffers->singleMaskElts * runBatchSize * sizeof(float));
  if(numMetaFeatures > 0 && gpuHandle->inputMeta != NULL)
    ggml_backend_tensor_set(gpuHandle->inputMeta, inputBuffers->metaInput.data(), 0,
                            inputBuffers->singleInputMetaElts * runBatchSize * sizeof(float));

  if(ggml_backend_graph_compute(gpuHandle->backend, gpuHandle->graph) != GGML_STATUS_SUCCESS)
    throw StringError("VULKAN backend: graph compute failed");

  vector<float> policyPassData(inputBuffers->singlePolicyPassResultElts * runBatchSize);
  vector<float> policyData(inputBuffers->singlePolicyResultElts * runBatchSize);
  vector<float> valueData(inputBuffers->singleValueResultElts * runBatchSize);
  vector<float> scoreValueData(inputBuffers->singleScoreValueResultElts * runBatchSize);
  vector<float> ownershipData(inputBuffers->singleOwnershipResultElts * runBatchSize);

  ggml_backend_tensor_get(gpuHandle->outPolicyPass, policyPassData.data(), 0, policyPassData.size() * sizeof(float));
  ggml_backend_tensor_get(gpuHandle->outPolicy, policyData.data(), 0, policyData.size() * sizeof(float));
  ggml_backend_tensor_get(gpuHandle->outValue, valueData.data(), 0, valueData.size() * sizeof(float));
  ggml_backend_tensor_get(gpuHandle->outScoreValue, scoreValueData.data(), 0, scoreValueData.size() * sizeof(float));
  ggml_backend_tensor_get(gpuHandle->outOwnership, ownershipData.data(), 0, ownershipData.size() * sizeof(float));

  assert((int)outputs.size() == batchSize);

  const int numPolicyChannels = (int)inputBuffers->singlePolicyPassResultElts;
  assert(inputBuffers->singlePolicyResultElts == (size_t)numPolicyChannels * nnXLen * nnYLen);
  const int numValueChannels = (int)inputBuffers->singleValueResultElts;
  const int numScoreValueChannels = (int)inputBuffers->singleScoreValueResultElts;

  // Per-row decode, reproducing the ONNX backend's post-processing exactly.
  // Outputs are raw logits, so the client applies softmax / tanh / etc.
  float policyProbsTmp[NNPos::MAX_NN_POLICY_SIZE];

  for(int row = 0; row < batchSize; row++) {
    NNOutput* output = outputs[row];
    assert(output->nnXLen == nnXLen);
    assert(output->nnYLen == nnYLen);
    float policyOptimism = (float)inputBufs[row]->policyOptimism;

    // Policy: OutputPolicyPass is [N, numPolicyChannels, 1, 1] and OutputPolicy is [N, numPolicyChannels, H, W].
    {
      const float* policyPassSrcBuf = policyPassData.data() + row * numPolicyChannels;
      const float* policySrcBuf = policyData.data() + row * numPolicyChannels * nnXLen * nnYLen;
      float* policyProbs = output->policyProbs;

      if(numPolicyChannels == 2 || (numPolicyChannels == 4 && modelVersion >= 16)) {
        // NCHW: channel 0 = base logits, channel 1 = optimism logits.
        for(int i = 0; i < nnXLen * nnYLen; i++) {
          float p = policySrcBuf[i];
          float pOpt = policySrcBuf[i + nnXLen * nnYLen];
          policyProbsTmp[i] = p + (pOpt - p) * policyOptimism;
        }
        SymmetryHelpers::copyOutputsWithSymmetry(
          policyProbsTmp, policyProbs, 1, nnYLen, nnXLen, inputBufs[row]->symmetry);
        policyProbs[nnXLen * nnYLen] =
          policyPassSrcBuf[0] + (policyPassSrcBuf[1] - policyPassSrcBuf[0]) * policyOptimism;
      }
      else {
        assert(numPolicyChannels == 1);
        SymmetryHelpers::copyOutputsWithSymmetry(
          policySrcBuf, policyProbs, 1, nnYLen, nnXLen, inputBufs[row]->symmetry);
        policyProbs[nnXLen * nnYLen] = policyPassSrcBuf[0];
      }
    }

    // Value: [N, 3, 1, 1] raw categorical logits (win/loss/noresult).
    {
      assert(numValueChannels == 3);
      output->whiteWinProb = valueData[row * numValueChannels];
      output->whiteLossProb = valueData[row * numValueChannels + 1];
      output->whiteNoResultProb = valueData[row * numValueChannels + 2];
    }

    // Ownership: [N, 1, H, W] raw, inverse-symmetried back to canonical orientation.
    if(output->whiteOwnerMap != NULL) {
      assert(inputBuffers->singleOwnershipResultElts == (size_t)nnXLen * nnYLen);
      const float* ownershipSrcBuf = ownershipData.data() + row * nnXLen * nnYLen;
      SymmetryHelpers::copyOutputsWithSymmetry(
        ownershipSrcBuf, output->whiteOwnerMap, 1, nnYLen, nnXLen, inputBufs[row]->symmetry);
    }

    // ScoreValue: [N, numScoreValueChannels, 1, 1] raw, version-dependent channel interpretation.
    {
      if(modelVersion >= 9) {
        assert(numScoreValueChannels == 6);
        output->whiteScoreMean = scoreValueData[row * numScoreValueChannels];
        output->whiteScoreMeanSq = scoreValueData[row * numScoreValueChannels + 1];
        output->whiteLead = scoreValueData[row * numScoreValueChannels + 2];
        output->varTimeLeft = scoreValueData[row * numScoreValueChannels + 3];
        output->shorttermWinlossError = scoreValueData[row * numScoreValueChannels + 4];
        output->shorttermScoreError = scoreValueData[row * numScoreValueChannels + 5];
      }
      else if(modelVersion >= 8) {
        assert(numScoreValueChannels == 4);
        output->whiteScoreMean = scoreValueData[row * numScoreValueChannels];
        output->whiteScoreMeanSq = scoreValueData[row * numScoreValueChannels + 1];
        output->whiteLead = scoreValueData[row * numScoreValueChannels + 2];
        output->varTimeLeft = scoreValueData[row * numScoreValueChannels + 3];
        output->shorttermWinlossError = 0;
        output->shorttermScoreError = 0;
      }
      else if(modelVersion >= 4) {
        assert(numScoreValueChannels == 2);
        output->whiteScoreMean = scoreValueData[row * numScoreValueChannels];
        output->whiteScoreMeanSq = scoreValueData[row * numScoreValueChannels + 1];
        output->whiteLead = output->whiteScoreMean;
        output->varTimeLeft = 0;
        output->shorttermWinlossError = 0;
        output->shorttermScoreError = 0;
      }
      else if(modelVersion >= 3) {
        assert(numScoreValueChannels == 1);
        output->whiteScoreMean = scoreValueData[row * numScoreValueChannels];
        output->whiteScoreMeanSq = output->whiteScoreMean * output->whiteScoreMean;
        output->whiteLead = output->whiteScoreMean;
        output->varTimeLeft = 0;
        output->shorttermWinlossError = 0;
        output->shorttermScoreError = 0;
      }
      else {
        ASSERT_UNREACHABLE;
      }
    }
  }
}

//--------------------------------------------------------------

std::string NeuralNet::getRuntimeBackendDetail(ConfigParser& cfg) {
  (void)cfg;
  return "vulkan";
}

int NeuralNet::getNumEffectiveDevices(ConfigParser& cfg, const std::vector<int>& gpuIdxByServerThread) {
  (void)cfg;
  std::set<int> distinctDevices;
  for(int idx : gpuIdxByServerThread)
    distinctDevices.insert(idx);
  return std::max(1, (int)distinctDevices.size());
}

NeuralNet::BatchPolicy NeuralNet::getBatchPolicy(ConfigParser& cfg) {
  (void)cfg;
  // The ggml graph is compiled for one fixed batch size (maxBatchSize), so the
  // batch is held constant and short batches are padded up to it, like the
  // ONNX backend's DirectML path.
  return NeuralNet::BatchPolicy::FixedShape;
}

//--------------------------------------------------------------

static const char* physicalDeviceTypeString(VkPhysicalDeviceType type) {
  switch(type) {
  case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return "integrated";
  case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:   return "discrete";
  case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:    return "virtual";
  case VK_PHYSICAL_DEVICE_TYPE_CPU:            return "cpu";
  default:                                     return "other";
  }
}

void NeuralNet::printDevices() {
  VkInstance instance = VK_NULL_HANDLE;
  VkApplicationInfo appInfo{};
  appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  appInfo.pApplicationName = "KataGo";
  appInfo.applicationVersion = 1;
  appInfo.pEngineName = "KataGo";
  appInfo.engineVersion = 1;

  VkInstanceCreateInfo createInfo{};
  createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  createInfo.pApplicationInfo = &appInfo;

  VkResult result = vkCreateInstance(&createInfo, nullptr, &instance);
  if(result != VK_SUCCESS) {
    cout << "Vulkan backend: could not create a Vulkan instance (is a Vulkan driver installed?)" << endl;
    return;
  }

  uint32_t deviceCount = 0;
  vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
  vector<VkPhysicalDevice> devices(deviceCount);
  vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

  if(deviceCount == 0)
    cout << "Vulkan backend: no Vulkan devices found" << endl;
  for(uint32_t i = 0; i < deviceCount; i++) {
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(devices[i], &props);
    cout << "Vulkan device " << i << ": " << props.deviceName
         << " (vendor " << Global::uint64ToString((uint64_t)props.vendorID)
         << ", device " << Global::uint64ToString((uint64_t)props.deviceID)
         << ", " << physicalDeviceTypeString(props.deviceType)
         << ", apiVersion " << (props.apiVersion >> 22) << "."
         << ((props.apiVersion >> 12) & 0x3FF) << "."
         << (props.apiVersion & 0xFFF) << ")" << endl;
  }

  vkDestroyInstance(instance, nullptr);
}

//--------------------------------------------------------------
// The layer-level test entry points are not implemented for this backend.

bool NeuralNet::testEvaluateConv(
  const ConvLayerDesc*, int, int, int, bool, bool,
  const std::vector<float>&, std::vector<float>&
) {
  return false;
}

bool NeuralNet::testEvaluateBatchNorm(
  const BatchNormLayerDesc*, int, int, int, bool, bool,
  const std::vector<float>&, const std::vector<float>&, std::vector<float>&
) {
  return false;
}

bool NeuralNet::testEvaluateResidualBlock(
  const ResidualBlockDesc*, int, int, int, bool, bool,
  const std::vector<float>&, const std::vector<float>&, std::vector<float>&
) {
  return false;
}

bool NeuralNet::testEvaluateGlobalPoolingResidualBlock(
  const GlobalPoolingResidualBlockDesc*, int, int, int, bool, bool,
  const std::vector<float>&, const std::vector<float>&, std::vector<float>&
) {
  return false;
}

#endif  // USE_VULKAN_BACKEND
