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
#include "../neuralnet/vulkantuner.h"

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
#include <map>
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
  // ---- layout annotation (L1: infra only, behavior identical) ----
  // L_NCHW: ne=[W,H,C,N] (classic KataGo NCHW ggml layout)
  // L_CS:   ne=[C,S,N], S=W*H, s=h*W+w (channel-fastest; L2)
  // L_MAT:  matrix-semantics domain inside attention (layout-orthogonal; L2)
  enum class Layout { L_NCHW, L_CS, L_MAT };
  struct AnnotatedTensor {
    ggml_tensor* t;
    Layout layout;
    ggml_type dtype;   // F32 | F16 (annotation; actual ggml type may differ)
  };

  ggml_context* ctx;
  ggml_context* wctx;   // initializer (weight) tensors live here so gallocr
                        // never reuses their buffers; allocated separately.
  int maxBatchSize;
  int nnXLen;   // board spatial dims; CS <-> NCHW conversions need the explicit
  int nnYLen;   // W/H split of S (a CS tensor alone cannot recover them)

  const onnx::ModelProto& model;
  unordered_map<string, ggml_tensor*> tensors;
  // Parallel annotation table (same keys as tensors). Initializer/constant
  // tensors have no layout semantics and are exempt (ensure() never touches
  // them; C10-4②).
  unordered_map<string, AnnotatedTensor> annTensors;
  // All prescan passes used to rebuild this map locally: 8 identical copies of the same two loops.
  // It is built ONCE in build() and read-only afterwards -- one truth, and a new prescan cannot
  // forget to build it.  Named `producerIndex` and not `producer` on purpose: a member called
  // `producer` would be SHADOWED by any leftover local of that name and the compiler would stay
  // silent (it would just use the empty local) -- the exact failure this refactor must avoid.
  unordered_map<string, int> producerIndex;

  // C-N34 (2026-08-28): fa-feed GEMM pointer set for selective f32acc (R1).
  // Filled at graph-build time with the fused qkv projection GEMM output (Y
  // in buildFusedProjections) and the unfused q/k/v projection GEMM outputs
  // (1x1-conv branch when the conv output name is in faFeedConvOuts).
  unordered_set<const ggml_tensor*> faFeedGemmPtrs;
  // q/k/v conv output names that feed fa (prescan-collected; fused + unfused).
  unordered_set<string> faFeedConvOuts;
  static bool faEnabled() {
    // fa-default (2026-09-02, fa-default-exec.md): ON by default; KATAGO_VK_FA=0
    // is the explicit opt-out (rollback channel). Char compare, no <cstring>.
    const char* v = getenv("KATAGO_VK_FA");
    static const bool e = (v == nullptr) || !(v[0] == '0' && v[1] == '\0');
    return e;
  }

  // S3-8/C-N28: layout hint table (production-side domain assignment, O5
  // two-layer structure). registerOut consults it to decide a tensor's output
  // domain instead of the caller's hardcoded default; consume-side ensure()
  // stays untouched as the fallback for dynamic/unhinted tensors.
  // Env-gated KATAGO_VK_LAYOUT_HINT, default OFF = current behavior
  // (bit-identical baseline). Skeleton: empty map, populated by the prescan
  // layout assignment (7-edge greedy, see s3-8-design.md).
  unordered_map<string, Layout> layoutHints;
  static bool layoutHintEnabled() {
    static const bool e = getenv("KATAGO_VK_LAYOUT_HINT") != nullptr;
    return e;
  }

  // Host-side data of initializer tensors, set into device memory after graph
  // allocation.
  unordered_map<string, vector<uint8_t>> initData;

  // Conversion-edge caches (R2): a converted tensor for a given source is
  // created once and reused by every consumer (e.g. InputMask is masked-mul'd
  // at ~40 sites; all share one CS copy instead of 40 identical edges).
  unordered_map<ggml_tensor*, ggml_tensor*> convertCacheToCS;     // NCHW -> CS
  unordered_map<ggml_tensor*, ggml_tensor*> convertCacheToNCHW;   // CS -> NCHW
  // Debug counters: how many unique conversion edges were inserted, and from
  // which op (diagnostics for the L2 CONT budget).
  int nConvToCS = 0;
  int nConvToNCHW = 0;

  // Name of the ONNX node currently being interpreted (diagnostics).
  string currentOpName;

  // ---- S1 fusion: SiLU decomposition (SwiGLU) ----
  // ONNX exports SwiGLU as Sigmoid(gate?) -> Mul(a, sig) -> Mul(silu, other).
  // Actual pattern in this model: silu = linear1 * sigmoid(linear1), then
  // swiglu = silu * linear_gate. We absorb the Sigmoid and the first Mul and
  // rebuild the swiglu Mul as ggml_silu(linear1) * linear_gate.
  // Map: swiglu Mul output name -> { silu-input tensor name (a), gate name }.
  struct SiluFusion {
    bool active = false;
    string siluInput;  // tensor name whose silu feeds the final mul
    string gateSrc;    // other input of the final mul
  };
  unordered_map<string, SiluFusion> siluFusions;   // by final Mul output name
  unordered_set<string> absorbedSiluNodes;         // Sigmoid + first Mul names

  // ---- F-4: plain swish (x*sigmoid(x)) -> ggml_silu ----
  // prescanSilu's guard (gate must be a Conv/MatMul output) deliberately
  // excludes plain swish activations (e.g. normactconv.act, policy head
  // actg), leaving 26 SIGMOID ops per eval. A Mul(x, Sigmoid(x)) with the
  // same source tensor is exactly silu(x) = x*sigmoid(x) = swish(x) --
  // ggml_silu exists, so the pair (Sigmoid + Mul) collapses into one op.
  // Map: swish Mul output name -> source tensor name.
  struct SwishFusion {
    bool active = false;
    string xSrc;
  };
  unordered_map<string, SwishFusion> swishFusions;   // by Mul output name

  void prescanSwish() {
    const auto& nodes = model.graph().node();
    for(int i = 0; i < (int)nodes.size(); i++) {
      const onnx::NodeProto& n = nodes[i];
      if(n.op_type() != "Mul" || n.input_size() != 2)
        continue;
      for(int k = 0; k < 2; k++) {
        auto itSig = producerIndex.find(n.input(k));
        if(itSig == producerIndex.end())
          continue;
        const onnx::NodeProto& sig = nodes[itSig->second];
        if(sig.op_type() != "Sigmoid" || sig.input_size() < 1)
          continue;
        const string& x = sig.input(0);
        if(n.input(1 - k) != x)
          continue;   // not swish (needs the same source on both sides)
        SwishFusion f;
        f.active = true;
        f.xSrc = x;
        swishFusions[n.output(0)] = f;
        absorbedSiluNodes.insert(sig.name());   // Sigmoid absorbed into silu
        break;
      }
    }
  }

  // ---- S1 fusion: RMSNorm chain ----
  // ONNX exports RMSNorm as Pow(x,2) -> ReduceMean(axes=[C]) -> Add(eps) ->
  // Sqrt -> Div(x, rms) -> Mul(weight). We rebuild the chain as
  // ggml_rms_norm(x, eps) * weight. The ReduceMean axis is the channel dim
  // (ONNX axis 1 == ggml ne2), which ggml_rms_norm does NOT reduce (it reduces
  // ne0), so the fused op runs on a permuted [C, W*H, b] view.
  // Map: Div output name -> { x tensor, weight tensor, eps }.
  struct RmsFusion {
    bool active = false;
    string xSrc;       // tensor name fed to Pow and Div (the activation)
    string weightSrc;  // tensor name of the per-channel weight
    string mulOut;     // output tensor name of the absorbed weight Mul
    float eps = 0.0f;
  };
  unordered_map<string, RmsFusion> rmsFusions;     // by Div output name
  unordered_set<string> absorbedRmsNodes;          // Pow/ReduceMean/Add/Sqrt node names
  unordered_map<string, ggml_tensor*> rmsWeightViews;  // weightSrc -> [C,1,1,1] view

  // ---- L2-B1: residual mask reassociation (C19-1/C20-1) ----
  // Residual pattern: Add(A, Mul(B, InputMask)) where B is a projection
  // (1x1 conv / matmul output) and A is an already-masked residual. The mask
  // is a per-position 0/1 multiplier and 1x1 convs are linear, so
  //   A + B*mask == (A + B) * mask
  // Reassociating moves the Mul after the Add, letting the Add fuse into the
  // projection's MUL_MAT epilogue (ggml-vulkan MUL_MAT+ADD fusion).
  // Map: Add output name -> { residual src (A), projection src (B) }.
  struct MaskReassoc {
    bool active = false;
    string resSrc;   // A: the (masked) residual tensor feeding the Add
    string projSrc;  // B: the unmasked projection output
  };
  unordered_map<string, MaskReassoc> maskReassocs;   // by Add output name
  unordered_set<string> absorbedMaskMulNodes;        // the Mul(B, InputMask) nodes

  // ---- L2-B2: softmax scale absorption (C19-2/C20-2) ----
  // Pattern: Softmax(Add(Mul(scores, scale_init), mask)) where scale_init is a
  // scalar initializer (1/sqrt(D)) and mask is a per-key additive bias.
  // Rebuilt as ggml_soft_max_ext(scores, mask_full, scale, 0) -- one fused
  // kernel instead of Mul+Add+Softmax. The mask must be materialized as
  // [S_k, S_q, 1, N] (F21; soft_max_ext asserts mask->ne[1] >= a->ne[1]).
  // Map: Softmax output name -> { scores, mask, scale }.
  struct SoftmaxScaleAbsorb {
    bool active = false;
    string scoresSrc;  // raw scores tensor (before Mul(scale))
    string maskSrc;    // additive mask tensor (InputMask/bias/reshape)
    float scale = 1.0f;
  };
  unordered_map<string, SoftmaxScaleAbsorb> softmaxAbsorbs;   // by Softmax output
  unordered_set<string> absorbedSoftmaxNodes;   // the absorbed Mul/Add node names
  // Materialized [S_k,S_q,1,N] mask, shared by all attention softmaxes.
  ggml_tensor* maskFullShared = NULL;
  int64_t maskFullSK = 0, maskFullSQ = 0, maskFullN = 0;

  void prescanSoftmaxScale() {
    const auto& nodes = model.graph().node();

    for(int i = 0; i < (int)nodes.size(); i++) {
      const onnx::NodeProto& n = nodes[i];
      if(n.op_type() != "Softmax")
        continue;
      bool axisLast = true;
      for(const onnx::AttributeProto& a : n.attribute())
        if(a.name() == "axis" && a.i() != 3)
          axisLast = false;
      if(!axisLast)
        continue;
      auto itAdd = producerIndex.find(n.input(0));
      if(itAdd == producerIndex.end())
        continue;
      const onnx::NodeProto& addN = nodes[itAdd->second];
      if(addN.op_type() != "Add" || addN.input_size() != 2)
        continue;
      // One Add input is Mul(scores, scale), the other is the mask.
      SoftmaxScaleAbsorb f;
      int mulIdx = -1;
      string mulNodeName, addNodeName = addN.name();
      for(int k = 0; k < 2; k++) {
        auto itMul = producerIndex.find(addN.input(k));
        if(itMul == producerIndex.end())
          continue;
        const onnx::NodeProto& mulN = nodes[itMul->second];
        if(mulN.op_type() != "Mul" || mulN.input_size() != 2)
          continue;
        // scale must be a scalar initializer; the other input is scores.
        const onnx::TensorProto* sc = NULL;
        int scIdx = -1;
        for(int j = 0; j < 2; j++) {
          sc = findInitializer(mulN.input(j));
          if(sc != NULL) { scIdx = j; break; }
        }
        if(sc == NULL)
          continue;
        vector<float> sv = initializerFloats(*sc);
        if(sv.size() != 1)
          continue;
        f.scale = sv[0];
        f.scoresSrc = mulN.input(1 - scIdx);
        f.maskSrc = addN.input(1 - k);
        mulIdx = k;
        mulNodeName = mulN.name();
        break;
      }
      if(mulIdx < 0)
        continue;
      f.active = true;
      softmaxAbsorbs[n.output(0)] = f;
      absorbedSoftmaxNodes.insert(mulNodeName);
      absorbedSoftmaxNodes.insert(addNodeName);
    }
  }

  // ---- M1: Flash Attention replacement (fa) ----
  // Replaces the per-block scores mul_mat + soft_max_ext + sv mul_mat chain
  // with a single ggml_flash_attn_ext. q/k/v are the L_MAT [D,H,S,N] tensors
  // (rope outputs for q/k, reshape output for v); fa requires [D,S,H,N]
  // (ne1=seq, ne2=heads, ggml.h doc) which we get as zero-dispatch strided
  // views (ggml_view_4d with swapped nb -- the fa shader addresses via
  // p.nb01/nb02/nb03, flash_attn.comp:110). fa output is [D,H,S,N]
  // (ggml.c: res = [v.ne0, q.ne2, q.ne1, q.ne3]) -- contiguous, so the
  // attnnchw reshape is a zero-copy view (saves the current permute+cont).
  // mask must be F16 (ggml.c:5434); we build a separate F16 cache from the
  // same mask source as the soft_max_ext fallback (caches kept separate per
  // OpenCode review). scale comes from the ONNX-absorbed softmax scale.
  struct FaSite {
    bool active = false;
    string qName, kName, vName;      // L_MAT tensor names [D,H,S,N]
    string maskSrc;                  // Add mask input (same as softmaxAbsorbs)
    string scoresOut, softmaxOut, svOut;  // output names to register
    float scale = 0.0f;
  };
  unordered_map<string, FaSite> faSites;   // by scores MatMul outName
  unordered_set<string> faAbsorbedSoftmax; // softmax outNames to skip
  unordered_set<string> faAbsorbedSv;      // sv MatMul outNames to skip
  ggml_tensor* faMaskF16Shared = NULL;     // F16 [S_k,S_q,1,N] (fa-only cache)
  int64_t faMaskF16SK = 0, faMaskF16SQ = 0, faMaskF16N = 0;

  void prescanFa() {
    const auto& nodes = model.graph().node();
    for(int i = 0; i < (int)nodes.size(); i++) {
      const onnx::NodeProto& n = nodes[i];
      if(n.op_type() != "MatMul" || n.input_size() < 2)
        continue;
      const string& scoresOut = n.output(0);
      if(scoresOut.find("/scores") == string::npos && scoresOut.find("scores") != 0)
        continue;
      // find the softmax absorbing this scores output
      string softmaxOut, svOut, maskSrc;
      float scale = 0.0f;
      for(auto& kv : softmaxAbsorbs) {
        if(!kv.second.active || kv.second.scoresSrc != scoresOut)
          continue;
        softmaxOut = kv.first;
        maskSrc = kv.second.maskSrc;
        scale = kv.second.scale;
        break;
      }
      if(softmaxOut.empty())
        continue;
      // find the sv MatMul consuming the softmax output (via its own outName)
      for(const auto& c : nodes) {
        if(c.op_type() != "MatMul" || c.input_size() < 2)
          continue;
        if(c.input(0) == softmaxOut || c.input(1) == softmaxOut) {
          const string& so = c.output(0);
          if(so.find("/sv") != string::npos) { svOut = so; break; }
        }
      }
      if(svOut.empty())
        continue;
      // q: scores input(0) is qT transpose output; its producer's input is q rope out
      auto itQ = producerIndex.find(n.input(0));
      if(itQ == producerIndex.end())
        continue;
      const onnx::NodeProto& qT = nodes[itQ->second];
      if(qT.op_type() != "Transpose" || qT.input_size() < 1)
        continue;
      // k: scores input(1) is the rope output directly
      // v: sv input(1) is vT transpose output; producer's input is v reshape out
      auto itSv = producerIndex.find(svOut);
      if(itSv == producerIndex.end())
        continue;
      const onnx::NodeProto& svN = nodes[itSv->second];
      if(svN.input_size() < 2)
        continue;
      auto itVT = producerIndex.find(svN.input(1));
      if(itVT == producerIndex.end())
        continue;
      const onnx::NodeProto& vT = nodes[itVT->second];
      if(vT.op_type() != "Transpose" || vT.input_size() < 1)
        continue;
      FaSite f;
      f.active = true;
      f.qName = qT.input(0);
      f.kName = n.input(1);
      f.vName = vT.input(0);
      f.maskSrc = maskSrc;
      f.scoresOut = scoresOut;
      f.softmaxOut = softmaxOut;
      f.svOut = svOut;
      f.scale = scale;
      faSites[scoresOut] = f;
      faAbsorbedSoftmax.insert(softmaxOut);
      faAbsorbedSv.insert(svOut);
    }
  }


  // In the CS domain the q/k reshape target [0,H,D,S] maps to a zero-copy
  // reshape_4d(x, D, H, S, N) view [D,H,S,N] (head-major c=h*D+d). The whole
  // rope subgraph (gather + Mul(cos) + Mul(sin) + Add) is rebuilt efficiently:
  //   hsnd = cont(permute(3,0,1,2))(q)     // [H,S,N,D] contiguous (h fastest)
  //   q2   = reshape_2d(H*S*N, D)          // rows (h,s,n), cols d
  //   sw2  = get_rows(q2, swapidx[32])     // row-internal channel-pair swap;
  //                                        // get_rows: dst[i0][d] = q2[i0][swapidx[d]]
  //   out2 = q2*cos2 + sw2*sin2            // [H*S*N, D] (tables in same layout)
  //   ret  = cont(permute(1,2,3,0))(reshape_4d(H,S,N,D)(out2))  // [D,H,S,N]
  // get_rows' ne0 is H*S*N (2166) so its shader uses the x dimension well;
  // the naive 3D-ids variant (ne0=H=6) wasted 506/512 lanes (393us/site).
  // Verified exhaustively in l2b3-rope-verify.py / l2b3-fulltrace.py.
  struct RopeSite {
    bool active = false;
    string swapIdxInit;  // ropeswapidx initializer
    string cosInit;      // ropecos initializer
    string sinInit;      // ropesinsigned initializer
    int64_t D = 0, H = 0, S = 0;   // head dim / heads / positions (per site)
    string t1MulName, t2MulName, addName;   // absorbed nodes (by node name)
    string addOut;   // rope_out tensor name (the Add output)
  };
  unordered_map<string, RopeSite> ropeSites;      // by reshape output name
  unordered_set<string> absorbedRopeNodes;        // absorbed Mul/Add node names
  int64_t ropeIdsN = 0;                           // batch for the 2D tables

  void prescanRope() {
    const auto& nodes = model.graph().node();

    for(int i = 0; i < (int)nodes.size(); i++) {
      const onnx::NodeProto& n = nodes[i];
      if(n.op_type() != "Gather")
        continue;
      int64_t axis = 0;
      for(const onnx::AttributeProto& a : n.attribute())
        if(a.name() == "axis")
          axis = a.i();
      if(axis != 2)
        continue;
      // data = reshape output (q/k), indices = swapidx initializer
      auto itData = producerIndex.find(n.input(0));
      if(itData == producerIndex.end())
        continue;
      const onnx::NodeProto& reshapeN = nodes[itData->second];
      if(reshapeN.op_type() != "Reshape")
        continue;
      const onnx::TensorProto* sw = findInitializer(n.input(1));
      if(sw == NULL)
        continue;
      // Locate t1 = Mul(q, cos) and t2 = Mul(swap, sin), then Add(t1, t2).
      RopeSite site;
      site.active = true;
      site.swapIdxInit = n.input(1);
      string t1Out, t2Out, addOut;
      for(int j = 0; j < (int)nodes.size(); j++) {
        const onnx::NodeProto& c = nodes[j];
        if(c.op_type() != "Mul" || c.input_size() != 2)
          continue;
        const string& a0 = c.input(0), a1 = c.input(1);
        if((a0 == reshapeN.output(0) || a1 == reshapeN.output(0)) &&
           (a0 == n.output(0) || a1 == n.output(0)))
          continue;   // mul of q and swap? not part of the pattern
        if(a0 == reshapeN.output(0) || a1 == reshapeN.output(0)) {
          const string& other = (a0 == reshapeN.output(0)) ? a1 : a0;
          if(strstr(other.c_str(), "cos") != NULL && site.cosInit.empty()) {
            site.cosInit = other;
            site.t1MulName = c.name();
            t1Out = c.output(0);
          }
        }
        else if(a0 == n.output(0) || a1 == n.output(0)) {
          const string& other = (a0 == n.output(0)) ? a1 : a0;
          if(strstr(other.c_str(), "sin") != NULL && site.sinInit.empty()) {
            site.sinInit = other;
            site.t2MulName = c.name();
            t2Out = c.output(0);
          }
        }
      }
      if(site.cosInit.empty() || site.sinInit.empty() || t1Out.empty() || t2Out.empty())
        continue;
      // The Add consuming t1 and t2 is the rope output.
      bool foundAdd = false;
      for(int j = 0; j < (int)nodes.size(); j++) {
        const onnx::NodeProto& c = nodes[j];
        if(c.op_type() != "Add" || c.input_size() != 2)
          continue;
        if((c.input(0) == t1Out && c.input(1) == t2Out) ||
           (c.input(0) == t2Out && c.input(1) == t1Out)) {
          site.addName = c.name();
          addOut = c.output(0);
          foundAdd = true;
          break;
        }
      }
      if(!foundAdd)
        continue;
      // head dims from the cos table shape (1,H,D,S)
      const onnx::TensorProto* ct = findInitializer(site.cosInit);
      if(ct == NULL || ct->dims_size() != 4)
        continue;
      site.H = ct->dims(1);
      site.D = ct->dims(2);
      site.S = ct->dims(3);
      site.addOut = addOut;
      ropeSites[reshapeN.output(0)] = site;
      absorbedRopeNodes.insert(site.t1MulName);
      absorbedRopeNodes.insert(site.t2MulName);
      absorbedRopeNodes.insert(site.addName);
    }
  }

  // ---- L2-B4: q/k/v projection fusion (C22/C23/C24-2) ----
  // The three attention projections of one blockstack (q_proj/k_proj/v_proj,
  // all 1x1 convs) share the same input tensor. Merge them into one GEMM:
  //   W_fused [IC, 3*OC] F16  (rows of q, k, v stacked)
  //   Y = mul_mat(W_fused, x2) -> [3*OC, S*N]
  //   q/k/v = view_2d segments -> cont -> reshape_3d [OC, S, N]
  // One src1 read instead of three (src1 bandwidth -66%), one dispatch.
  struct ProjFusion {
    bool active = false;
    bool built = false;
    string xSrc;             // shared Conv input tensor name
    string qConvOut, kConvOut, vConvOut;   // Conv output names (registered)
    string qInit, kInit, vInit;            // weight initializer names
    int64_t IC = 0, OC = 0;
  };
  unordered_map<string, ProjFusion> projFusions;   // by xSrc
  unordered_map<string, string> projSegOf;         // convOut -> xSrc

  // ---- Sequence ①: RMSNorm gamma absorption into downstream 1x1 convs ----
  // Graph: Div(normed) -> Mul(gamma)[scaled] -> Mul(InputMask)[*/mask] -> Conv(W)
  // Absorb gamma into W by input channel (W'[oc,ic]=W[oc,ic]*gamma[ic], exact for
  // linear 1x1 convs), leaving Div(normed) -> Mul(InputMask) -> Conv(W') so the
  // existing RMS_NORM_MUL fusion (ggml-vulkan) eats rms_norm -> mul(mask) -- the
  // mask is a spatial broadcast [1,S,N] which rms_norm.comp handles correctly
  // (verified H-2: fastmod(col, ne10=1) -> per-channel broadcast).
  // Map: conv weight initializer -> rms weight initializer (gamma).
  unordered_map<string, string> gammaAbsorbOf;
  // Div output names whose gamma was absorbed (Div branch then skips the gamma mul).
  unordered_set<string> gammaAbsorbedNorm;

  void prescanGammaAbsorb() {
    const auto& nodes = model.graph().node();
    for(auto& kv : rmsFusions) {
      if(!kv.second.active)
        continue;
      const RmsFusion& f = kv.second;
      // 1) find the mask mul consuming the gamma-scaled output: Mul(scaled, InputMask)
      string maskOut;
      for(const auto& c : nodes) {
        if(c.op_type() != "Mul" || c.input_size() != 2)
          continue;
        bool hasMask = false, hasScaled = false;
        for(int k = 0; k < 2; k++) {
          if(c.input(k) == "InputMask")
            hasMask = true;
          if(c.input(k) == f.mulOut)
            hasScaled = true;
        }
        if(hasMask && hasScaled) {
          maskOut = c.output(0);
          break;
        }
      }
      if(maskOut.empty())
        continue;   // norm not mask-corrected (e.g. trunk-final): keep gamma mul
      // 2) every 1x1 Conv consuming the masked output gets the gamma absorbed
      bool anyConv = false;
      for(const auto& c : nodes) {
        if(c.op_type() != "Conv" || c.input_size() < 2)
          continue;
        bool consumes = false;
        for(int k = 0; k < c.input_size(); k++)
          if(c.input(k) == maskOut)
            consumes = true;
        if(!consumes)
          continue;
        const onnx::TensorProto* wt = findInitializer(c.input(1));
        if(wt == NULL || wt->dims_size() != 4 || wt->dims(2) != 1 || wt->dims(3) != 1)
          continue;   // only 1x1 convs are absorb targets
        gammaAbsorbOf[c.input(1)] = f.weightSrc;
        anyConv = true;
      }
      if(anyConv)
        gammaAbsorbedNorm.insert(kv.first);
    }
  }

  void prescanProjFusion() {
    const auto& nodes = model.graph().node();

    // Map xSrc -> {convOut -> (init, nodeName)} for every 1x1 Conv, plus the
    // q/k roles from the rope sites (their reshape consumes the conv output).
    unordered_map<string, vector<pair<string, string>>> convsByIn;  // xSrc -> (convOut, init)
    for(int i = 0; i < (int)nodes.size(); i++) {
      const onnx::NodeProto& n = nodes[i];
      if(n.op_type() != "Conv" || n.input_size() < 2)
        continue;
      const onnx::TensorProto* w = findInitializer(n.input(1));
      if(w == NULL || w->dims_size() != 4)
        continue;
      int64_t kH = w->dims(2), kW = w->dims(3);
      if(kH != 1 || kW != 1)
        continue;
      convsByIn[n.input(0)].push_back({n.output(0), n.input(1)});
    }
    // q/k conv outputs: rope sites keyed by reshape output; reshape consumes
    // the conv output.
    unordered_map<string, bool> isQ;   // convOut -> is q (vs k)
    for(auto& kv : ropeSites) {
      const string& reshapeOut = kv.first;
      auto itReshape = producerIndex.find(reshapeOut);
      if(itReshape == producerIndex.end())
        continue;
      const onnx::NodeProto& rn = nodes[itReshape->second];
      if(rn.op_type() != "Reshape" || rn.input_size() < 1)
        continue;
      const string& convOut = rn.input(0);
      bool q = strstr(kv.second.cosInit.c_str(), "/qrope/") != NULL;
      isQ[convOut] = q;
    }
    for(auto& kv : convsByIn) {
      const string& xSrc = kv.first;
      string qConv, kConv, vConv;
      string qInit, kInit, vInit;
      for(auto& pr : kv.second) {
        const string& convOut = pr.first;
        auto it = isQ.find(convOut);
        if(it != isQ.end()) {
          if(it->second && qConv.empty()) { qConv = convOut; qInit = pr.second; }
          else if(!it->second && kConv.empty()) { kConv = convOut; kInit = pr.second; }
        }
        else if(vConv.empty()) { vConv = convOut; vInit = pr.second; }
      }
      if(qConv.empty() || kConv.empty() || vConv.empty())
        continue;   // not a q/k/v triple (e.g. o_proj or other convs)
      // all three must be 1x1 with identical IC/OC
      const onnx::TensorProto* wq = findInitializer(qInit);
      const onnx::TensorProto* wv = findInitializer(vInit);
      if(wq == NULL || wv == NULL)
        continue;
      if(wq->dims(0) != wv->dims(0) || wq->dims(1) != wv->dims(1))
        continue;   // IC/OC mismatch (should not happen)
      ProjFusion f;
      f.active = true;
      f.xSrc = xSrc;
      f.qConvOut = qConv; f.kConvOut = kConv; f.vConvOut = vConv;
      f.qInit = qInit; f.kInit = kInit; f.vInit = vInit;
      f.IC = wq->dims(1);
      f.OC = wq->dims(0);
      projFusions[xSrc] = f;
      projSegOf[qConv] = xSrc;
      projSegOf[kConv] = xSrc;
      projSegOf[vConv] = xSrc;
    }

    // C-N34: collect every q/k/v projection conv output that feeds fa.
    // - fused: the triple detected above (q/k/v convOuts).
    // - unfused: trace ropeSites (q/k reshape inputs) and faSites.vName back
    //   to their nearest upstream Conv. Unfused q/k may not appear in ropeSites
    //   (fa's own rope path), so also trace the fa qName/kName; the v path is
    //   traced via faSites.vName (v may be roped or not).
    if(faEnabled()) {
      auto findConvOut = [&](const string& start) -> string {
        string cur = start;
        for(int hop = 0; hop < 5; hop++) {
          auto it = producerIndex.find(cur);
          if(it == producerIndex.end())
            return "";
          const onnx::NodeProto& pn = nodes[it->second];
          if(pn.op_type() == "Conv" && pn.output_size() > 0)
            return pn.output(0);
          if(pn.input_size() == 0)
            return "";
          cur = pn.input(0);
        }
        return "";
      };
      for(auto& kv : projFusions) {
        if(!kv.second.qConvOut.empty()) faFeedConvOuts.insert(kv.second.qConvOut);
        if(!kv.second.kConvOut.empty()) faFeedConvOuts.insert(kv.second.kConvOut);
        if(!kv.second.vConvOut.empty()) faFeedConvOuts.insert(kv.second.vConvOut);
      }
      for(auto& kv : ropeSites) {
        string c = findConvOut(kv.first);
        if(!c.empty()) faFeedConvOuts.insert(c);
      }
      for(auto& kv : faSites) {
        string cq = findConvOut(kv.second.qName);
        string ck = findConvOut(kv.second.kName);
        string cv = findConvOut(kv.second.vName);
        if(!cq.empty()) faFeedConvOuts.insert(cq);
        if(!ck.empty()) faFeedConvOuts.insert(ck);
        if(!cv.empty()) faFeedConvOuts.insert(cv);
        // R2 (MMPREC='3'): also the out-projection 1x1 Conv consuming the fa sv
        // output, via its Transpose/Reshape chain.
        const char* mmp = getenv("KATAGO_VK_MMPREC_F32");
        if(mmp != nullptr && mmp[0] == '3') {
          string cur = kv.second.svOut;
          for(int hop = 0; hop < 5; hop++) {
            int next = -1;
            for(int i = 0; i < (int)nodes.size(); i++) {
              const onnx::NodeProto& pn = nodes[i];
              if(pn.input_size() < 1 || pn.input(0) != cur)
                continue;
              if(pn.op_type() == "Conv") {
                const onnx::TensorProto* w = findInitializer(pn.input(1));
                if(w != NULL && w->dims_size() == 4 && w->dims(2) == 1 && w->dims(3) == 1)
                  faFeedConvOuts.insert(pn.output(0));
                next = -2;
                break;
              }
              next = i;
              break;
            }
            if(next < 0)
              break;
            cur = nodes[next].output(0);
          }
        }
      }
    }
  }

  // Conv weight with the RMSNorm gamma absorbed by input channel (Sequence ①).
  // F16 storage like the plain path; cached per initializer name.
  ggml_tensor* convWeightWithGamma(const string& initName) {
    auto itG = gammaAbsorbOf.find(initName);
    if(itG == gammaAbsorbOf.end())
      return initializerTensor(initName, GGML_TYPE_F16);
    const string name = initName + "@gamma";
    auto it = tensors.find(name);
    if(it != tensors.end())
      return it->second;
    const onnx::TensorProto* wt = findInitializer(initName);
    testAssert(wt != NULL);
    vector<float> wf = initializerFloats(*wt);
    vector<float> gv = initializerFloats(*findInitializer(itG->second));
    const int64_t IC = (int64_t)gv.size();
    testAssert((int64_t)wf.size() % IC == 0);
    vector<uint16_t> bytes(wf.size());
    for(size_t i = 0; i < wf.size(); i++)
      bytes[i] = ggml_fp32_to_fp16(wf[i] * gv[i % (size_t)IC]);
    int64_t ne[4] = {1, 1, IC, (int64_t)(wf.size() / (size_t)IC)};
    return hostConstant(name, GGML_TYPE_F16, ne, bytes.data());
  }

  void buildFusedProjections(ProjFusion& f) {
    // Shared input must already be built (CS F32).
    auto itX = annTensors.find(f.xSrc);
    if(itX == annTensors.end())
      return;   // not built yet -> the normal conv path will handle it
    AnnotatedTensor annX = itX->second;
    ggml_tensor* x = ensure(annX, Layout::L_CS, GGML_TYPE_F32);
    const int64_t S = x->ne[1], N = x->ne[2];
    testAssert(x->ne[0] == f.IC);
    // Fused weight: ggml [IC, 3*OC] F16, flat(ic, seg*OC+oc) = W_seg[oc*IC+ic]
    // (ONNX [OC,IC,1,1] row-major: flat[oc*IC+ic]).
    string wname = f.qInit + "+" + f.kInit + "+" + f.vInit + "@fused";
    if(tensors.count(wname) == 0) {
      auto rd = [&](const string& initName) {
        const onnx::TensorProto* t = findInitializer(initName);
        testAssert(t != NULL);
        return initializerFloats(*t);
      };
      vector<float> wq = rd(f.qInit), wk = rd(f.kInit), wv = rd(f.vInit);
      testAssert((int64_t)wq.size() == f.IC * f.OC);
      // Sequence ①: absorb each segment's RMSNorm gamma by input channel.
      auto gammaOf = [&](const string& initName) {
        vector<float> g;
        auto itG = gammaAbsorbOf.find(initName);
        if(itG != gammaAbsorbOf.end())
          g = initializerFloats(*findInitializer(itG->second));
        return g;
      };
      vector<float> gq = gammaOf(f.qInit), gk = gammaOf(f.kInit), gv = gammaOf(f.vInit);
      const float* gs[3] = {gq.empty() ? nullptr : gq.data(),
                            gk.empty() ? nullptr : gk.data(),
                            gv.empty() ? nullptr : gv.data()};
      vector<uint16_t> bytes((size_t)f.IC * 3 * f.OC);
      uint16_t* dst = bytes.data();
      const float* segs[3] = {wq.data(), wk.data(), wv.data()};
      for(int seg = 0; seg < 3; seg++)
        for(int64_t oc = 0; oc < f.OC; oc++)
          for(int64_t ic = 0; ic < f.IC; ic++) {
            float v = segs[seg][oc * f.IC + ic];
            if(gs[seg])
              v *= gs[seg][ic];
            dst[ic + f.IC * (seg * f.OC + oc)] = ggml_fp32_to_fp16(v);
          }
      int64_t ne[4] = {f.IC, 3 * f.OC, 1, 1};
      hostConstant(wname, GGML_TYPE_F16, ne, bytes.data());
    }
    ggml_tensor* W = ggml_reshape_2d(ctx, tensors[wname], f.IC, 3 * f.OC);  // view
    ggml_tensor* x2 = ggml_reshape_2d(ctx, x, f.IC, S * N);                 // view
    ggml_tensor* Y = ggml_mul_mat(ctx, W, x2);                              // [3*OC, S*N] F32
    if(faEnabled())
      faFeedGemmPtrs.insert(Y);   // C-N34 R1: fused qkv GEMM feeds fa directly
    const size_t es = ggml_element_size(Y);
    const int64_t segNe[4] = {f.OC, S, N, 1};
    struct { const char* out; } segs[3] = {{f.qConvOut.c_str()}, {f.kConvOut.c_str()}, {f.vConvOut.c_str()}};
    // N57 slice-1 (2026-09-12): hand the strided qkv window straight to the
    // head-split view instead of materializing it here. The only consumer of
    // these three tensors is the [0,H,D,S] head-split Reshape below, whose own
    // result is consumed by either the RoPE cont(permute(...)) or the FA
    // ggml_view_4d -- both accept arbitrary strides -- and which is turned into
    // a stride-preserving view when its input is not contiguous. The copy
    // removed here is 42.4% of all CONT bytes (census13: 90 dispatches = 3 x 30
    // sites). Measured on the 1660 (A-B-B-A, bit-exact, margins identical):
    // +2.84% / +3.45% / +2.49% end-to-end (c384/c512/b11) -- the strided read
    // makes the 125 remaining copies ~38% slower, which eats ~45% of the copy
    // that disappears. Rollback: KATAGO_VK_QKV_NOCONT=0.
    static const char* katago_qkv_nocont_env = getenv("KATAGO_VK_QKV_NOCONT");
    // Legality: the FA feed leans on vec4 row strides being 8*16B aligned when it
    // takes the "aligned" variant (flash_attn_cm2.comp:196-204 masks stride bits).
    // The strided window's row stride is Y->nb[1] = 3*OC*4 B, so require 128B
    // alignment (== OC % 32 == 0 for these F32 CS tensors); otherwise keep the
    // copy -- loudly, never silently.
    const bool qkvNoContLegal = (Y->nb[1] % 128) == 0;
    const bool qkvNoContWanted = !(katago_qkv_nocont_env != nullptr && katago_qkv_nocont_env[0] == '0');
    const bool qkvNoCont = qkvNoContWanted && qkvNoContLegal;
    static bool katago_qkv_nocont_logged = false;
    if(qkvNoContWanted && !qkvNoContLegal && !katago_qkv_nocont_logged) {
      katago_qkv_nocont_logged = true;
      fprintf(stderr, "N57|REJECTED|Y->nb[1]=%zu not 128B aligned (OC=%lld) -- falling back to the copy\n",
              (size_t)Y->nb[1], (long long)f.OC);
    }
    for(int seg = 0; seg < 3; seg++) {
      ggml_tensor* out;
      if(qkvNoCont) {
        // [OC,S,N] with the (s,n) pair still merged into Y's row stride: the
        // fork's CS algebra is flat(c,s,n) = c + C*s + C*S*n (s fastest, n
        // slowest; l2b3-rope-verify.py:56), so S has stride nb1 and N has
        // stride nb1*S -- the same element order the contiguous reshape_3d
        // produces. (Getting this backwards is still a pure permutation of the
        // same span: silent wrong values, caught only by the bit-eq gate.)
        out = ggml_view_3d(ctx, Y, f.OC, S, N, Y->nb[1], Y->nb[1] * (size_t)S, (size_t)seg * f.OC * es);
        if(!katago_qkv_nocont_logged) {
          katago_qkv_nocont_logged = true;
          fprintf(stderr, "N57|qkv split copy removed|Y ne0=%lld ne1=%lld nb1=%zu nb0=%zu|OC=%lld S=%lld N=%lld|seg_off=%zu\n",
                  (long long)Y->ne[0], (long long)Y->ne[1], (size_t)Y->nb[1], (size_t)Y->nb[0],
                  (long long)f.OC, (long long)S, (long long)N, (size_t)f.OC * es);
        }
      } else {
        ggml_tensor* view = ggml_view_2d(ctx, Y, f.OC, S * N, Y->nb[1], (size_t)seg * f.OC * es);
        ggml_tensor* c = ggml_cont(ctx, view);                                 // [OC, S*N] contiguous
        out = ggml_reshape_3d(ctx, c, f.OC, S, N);                             // [OC,S,N] view
      }
      tensors[segs[seg].out] = out;
      annTensors[segs[seg].out] = {out, Layout::L_CS, GGML_TYPE_F32};
    }
    f.built = true;
  }

  void prescanMaskReassoc() {
    const auto& nodes = model.graph().node();

    for(int i = 0; i < (int)nodes.size(); i++) {
      const onnx::NodeProto& n = nodes[i];
      if(n.op_type() != "Add" || n.input_size() != 2)
        continue;
      // The masked branch must be input(1): Mul(proj, InputMask).
      auto itMul = producerIndex.find(n.input(1));
      if(itMul == producerIndex.end())
        continue;
      const onnx::NodeProto& mulN = nodes[itMul->second];
      if(mulN.op_type() != "Mul" || mulN.input_size() != 2)
        continue;
      int maskIdx = -1, projIdx = -1;
      for(int k = 0; k < 2; k++) {
        if(mulN.input(k) == "InputMask")
          maskIdx = k;
        else
          projIdx = k;
      }
      if(maskIdx < 0 || projIdx < 0)
        continue;
      // B must be produced by a Conv/MatMul (a real projection) and A must be
      // a computed tensor, not a graph input or initializer.
      auto itProj = producerIndex.find(mulN.input(projIdx));
      if(itProj == producerIndex.end())
        continue;
      const onnx::NodeProto& projN = nodes[itProj->second];
      if(projN.op_type() != "Conv" && projN.op_type() != "MatMul")
        continue;
      if(producerIndex.find(n.input(0)) == producerIndex.end())
        continue;   // A is a graph input: not a residual chain
      MaskReassoc f;
      f.active = true;
      f.resSrc = n.input(0);
      f.projSrc = mulN.input(projIdx);
      maskReassocs[n.output(0)] = f;
      absorbedMaskMulNodes.insert(mulN.name());
    }
  }

  void prescanRms() {
    const auto& nodes = model.graph().node();

    for(int i = 0; i < (int)nodes.size(); i++) {
      const onnx::NodeProto& n = nodes[i];
      if(n.op_type() != "Div")
        continue;
      // Div(x, rmsstd): rmsstd = Sqrt(Add(ReduceMean(Pow(x,2)), eps))
      auto itRms = producerIndex.find(n.input(1));
      if(itRms == producerIndex.end())
        continue;
      const onnx::NodeProto& sqrtN = nodes[itRms->second];
      if(sqrtN.op_type() != "Sqrt")
        continue;
      auto itAdd = producerIndex.find(sqrtN.input(0));
      if(itAdd == producerIndex.end())
        continue;
      const onnx::NodeProto& addN = nodes[itAdd->second];
      if(addN.op_type() != "Add")
        continue;
      // Add(input(0)) is the ReduceMean(Pow(x,2)) output.
      auto itMean = producerIndex.find(addN.input(0));
      if(itMean == producerIndex.end())
        continue;
      const onnx::NodeProto& meanN = nodes[itMean->second];
      if(meanN.op_type() != "ReduceMean")
        continue;
      auto itSq = producerIndex.find(meanN.input(0));
      if(itSq == producerIndex.end())
        continue;
      const onnx::NodeProto& sqN = nodes[itSq->second];
      if(sqN.op_type() != "Pow")
        continue;
      // x must be the same tensor feeding both Pow and Div
      if(sqN.input(0) != n.input(0))
        continue;
      // ---- structural guards: only fuse the exact RMSNorm pattern, otherwise
      // fall back to the original chain (avoid silently wrong results on other
      // model variants) ----
      // 1) Pow exponent must be exactly 2
      {
        const onnx::TensorProto* powInit = findInitializer(sqN.input(1));
        if(powInit == NULL)
          continue;
        vector<float> pv = initializerFloats(*powInit);
        if(pv.size() != 1 || pv[0] != 2.0f)
          continue;
      }
      // 2) ReduceMean must reduce exactly the channel axis (ONNX axis 1) with
      // keepdims. The axes come from an initializer; also accept the common
      // attribute form.
      {
        const onnx::TensorProto* axInit = NULL;
        for(int a = 0; a < meanN.input_size(); a++)
          if(findInitializer(meanN.input(a)) != NULL)
            axInit = findInitializer(meanN.input(a));
        if(axInit != NULL) {
          vector<int64_t> ax = initializerInt64(*axInit);
          if(ax.size() != 1 || ax[0] != 1)
            continue;
        }
        bool keepdims = true;
        for(const onnx::AttributeProto& at : meanN.attribute())
          if(at.name() == "keepdims")
            keepdims = at.i() != 0;
        if(!keepdims)
          continue;
      }
      // 3) eps must be a scalar initializer
      const onnx::TensorProto* epsInit = findInitializer(addN.input(1));
      if(epsInit == NULL)
        continue;
      vector<float> ev = initializerFloats(*epsInit);
      if(ev.size() != 1)
        continue;
      RmsFusion f;
      f.active = true;
      f.xSrc = n.input(0);
      f.eps = ev[0];
      // The weight Mul follows the Div: find the consumer Mul(node output, weight)
      // and record the weight initializer name and the Mul node name. The
      // weight must be a per-channel [1,1,C,1] tensor, not a scalar.
      for(int j = 0; j < (int)nodes.size(); j++) {
        const onnx::NodeProto& c = nodes[j];
        if(c.op_type() != "Mul" || c.input(0) != n.output(0))
          continue;
        const onnx::TensorProto* wInit = findInitializer(c.input(1));
        if(wInit == NULL)
          continue;
        // weight dims: expect [1,1,C,1] (ggml ne [1,1,C,1]); reject scalars
        int64_t ne4[4] = {1, 1, 1, 1};
        for(int d = 0; d < wInit->dims_size() && d < 4; d++)
          ne4[3 - d] = wInit->dims(d);
        if(ne4[0] == 1 && ne4[1] == 1 && ne4[2] == 1 && ne4[3] == 1)
          continue;   // scalar weight: keep the original chain
        f.weightSrc = c.input(1);
        f.mulOut = c.output(0);
        absorbedRmsNodes.insert(c.name());
        break;
      }
      if(f.weightSrc.empty())
        continue;   // no weight Mul: keep the original chain
      rmsFusions[n.output(0)] = f;
      absorbedRmsNodes.insert(sqN.name());
      absorbedRmsNodes.insert(meanN.name());
      absorbedRmsNodes.insert(addN.name());
      absorbedRmsNodes.insert(sqrtN.name());
    }
  }

  void prescanSilu() {
    const auto& nodes = model.graph().node();

    for(int i = 0; i < (int)nodes.size(); i++) {
      const onnx::NodeProto& n = nodes[i];
      if(n.op_type() != "Mul")
        continue;
      // final swiglu Mul: one input is the silu product, other is the gate
      for(int inIdx = 0; inIdx < 2; inIdx++) {
        auto itProd = producerIndex.find(n.input(inIdx));
        if(itProd == producerIndex.end())
          continue;
        const onnx::NodeProto& prod = nodes[itProd->second];
        if(prod.op_type() != "Mul")
          continue;
        // prod = a * sigmoid(a): find the sigmoid among prod's inputs
        string sigTensorName, sigInput, sigNodeName;
        for(int pIdx = 0; pIdx < 2; pIdx++) {
          auto itSig = producerIndex.find(prod.input(pIdx));
          if(itSig == producerIndex.end())
            continue;
          if(nodes[itSig->second].op_type() == "Sigmoid") {
            sigTensorName = prod.input(pIdx);
            sigInput = nodes[itSig->second].input(0);
            sigNodeName = nodes[itSig->second].name();
          }
        }
        if(sigTensorName.empty())
          continue;
        // the other prod input must equal the sigmoid input
        const string& aName = prod.input(0) == sigTensorName ? prod.input(1) : prod.input(0);
        if(aName != sigInput)
          continue;
        SiluFusion f;
        f.active = true;
        f.siluInput = aName;
        f.gateSrc = n.input(1 - inIdx);
        // Guard: the gate must be produced by a compute node (Conv/MatMul),
        // not a graph input or initializer. Otherwise plain swish activations
        // (a*sigmoid(a)) followed by a mask Mul get misidentified as SwiGLU.
        {
          auto itGate = producerIndex.find(f.gateSrc);
          if(itGate == producerIndex.end())
            continue;   // gate is a graph input (e.g. InputMask): not a SwiGLU
          const onnx::NodeProto& gateN = nodes[itGate->second];
          if(gateN.op_type() != "Conv" && gateN.op_type() != "MatMul")
            continue;
        }
        siluFusions[n.output(0)] = f;
        absorbedSiluNodes.insert(prod.name());
        absorbedSiluNodes.insert(sigNodeName);
        break;
      }
    }
  }

  // One shared full-shape reference tensor per distinct target shape, reused
  // by ggml_repeat for broadcasts (repeat reads only the small operand; the
  // reference only supplies the shape, so a handful of them per handle is all
  // it takes and the transient repeat outputs reuse graph memory).
  // S1-④ (N-2): the reference content is never read -- ggml-vulkan's repeat
  // dispatch binds src1 (the shape reference) as nullptr (GV:13186) -- so the
  // tensors no longer need ggml_set_input. Letting gallocr own them lets the
  // buffer participate in liveness-based reuse instead of permanently pinning
  // ~8.3MB per shape (mask ref [S,S,1,N]); aliased overwrites are harmless.
  unordered_map<uint64_t, ggml_tensor*> shapeRefs;

  // B2a (next-temp §10.11): when true, FA sites are built WITHOUT a mask tensor
  // (skip repeat+cast, pass NULL). Valid only when the batch mask is guaranteed
  // all-ones (requireExactNNLen handle) or under the test-only KATAGO_VK_FA_NULL
  // env with full-board inputs; additive mask all-zero is equivalent to NULL.
  const bool faNoMask;

  OnnxToGgml(ggml_context* c, ggml_context* wc, int maxBatch, int xLen, int yLen, const onnx::ModelProto& m, bool noMaskFA)
    : ctx(c), wctx(wc), maxBatchSize(maxBatch), nnXLen(xLen), nnYLen(yLen), model(m), faNoMask(noMaskFA)
  {}

  ggml_tensor* shapeRefFor(const int64_t ne[4]) {
    uint64_t key = ((uint64_t)ne[0] << 48) | ((uint64_t)ne[1] << 32) | ((uint64_t)ne[2] << 16) | (uint64_t)ne[3];
    auto it = shapeRefs.find(key);
    if(it != shapeRefs.end())
      return it->second;
    ggml_tensor* t = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ne[0], ne[1], ne[2], ne[3]);
    // S1-④: no ggml_set_input -- content is never read (repeat dispatches src1
    // as nullptr, GV:13186), so gallocr may reuse the buffer via liveness
    // analysis. A/B verified: no perf impact vs set_input (901-902 nnEval/s).
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
  // data for the post-allocation upload. The cache is keyed by name only, so
  // the first request fixes the type: assert a conflicting later request
  // instead of silently returning a tensor of the wrong type.
  ggml_tensor* initializerTensor(const string& name, ggml_type type) {
    auto it = tensors.find(name);
    if(it != tensors.end()) {
      testAssert(it->second->type == type);
      return it->second;
    }
    const onnx::TensorProto* t = findInitializer(name);
    if(t == NULL)
      throw StringError("VULKAN backend: initializer '" + name + "' not found in the ONNX graph");
    int64_t ne[4];
    dimsToNe(*t, ne);
    if(type == GGML_TYPE_F32 || type == GGML_TYPE_F16) {
      vector<float> data = initializerFloats(*t);
      testAssert((int64_t)data.size() == ne[0] * ne[1] * ne[2] * ne[3]);
      vector<uint8_t> bytes;
      if(type == GGML_TYPE_F16) {
        // B1: keep matmul/conv weights natively in F16. ggml-vulkan converts
        // F32 operands to F16 on every eval on coopmat2 devices (and im2col
        // emits F16 anyway), so pre-converting here (same RTNE rounding as the
        // device-side to_fp16) removes the per-eval weight conversion with
        // bit-identical numerics.
        bytes.resize(data.size() * 2);
        ggml_fp16_t* dst = (ggml_fp16_t*)bytes.data();
        for(size_t i = 0; i < data.size(); i++)
          dst[i] = ggml_fp32_to_fp16(data[i]);
      }
      else {
        bytes.resize(data.size() * 4);
        memcpy(bytes.data(), data.data(), bytes.size());
      }
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

  // Host-generated constant (C27-3): create a weight-context tensor plus its
  // upload bytes from arbitrary host data, bypassing ONNX initializers. Used
  // for L2-B RoPE tables: ids[32,S,N] I32 (swapidx broadcast) and the CS-layout
  // cos/sin tables. Cached by name like initializerTensor.
  ggml_tensor* hostConstant(const string& name, ggml_type type, const int64_t ne[4], const void* data) {
    auto it = tensors.find(name);
    if(it != tensors.end()) {
      testAssert(it->second->type == type);
      return it->second;
    }
    size_t es = ggml_type_size(type);
    size_t nbytes = es;
    for(int i = 0; i < 4; i++)
      nbytes *= (size_t)ne[i];
    vector<uint8_t> bytes(nbytes);
    memcpy(bytes.data(), data, nbytes);
    initData[name] = std::move(bytes);
    ggml_tensor* t = ggml_new_tensor_4d(wctx, type, ne[0], ne[1], ne[2], ne[3]);
    ggml_set_name(t, name.c_str());
    tensors[name] = t;
    return t;
  }

  // ---- shape helpers ----

  // 2D variant for the efficient RoPE path: table laid out as [H*S*N, D]
  // (flat(i0, d) = i0 + H*S*N*d, i0 = h + H*s + H*S*n), so the multiply
  // operates on the same [H*S*N, D] tensor as get_rows' output.
  ggml_tensor* ropeCosTable2D(const RopeSite& site, bool isSin) {
    const string& initName = isSin ? site.sinInit : site.cosInit;
    const string name = initName + (isSin ? "@2d_sin" : "@2d_cos");
    auto it = tensors.find(name);
    if(it != tensors.end())
      return it->second;
    const onnx::TensorProto* t = findInitializer(initName);
    testAssert(t != NULL);
    vector<float> src = initializerFloats(*t);
    const int64_t H = site.H, D = site.D, S = site.S;
    const int64_t N = ropeIdsN;   // batch from the shared ids table setup
    testAssert(N > 0);
    vector<float> dst((size_t)H * S * N * D);
    for(int64_t h = 0; h < H; h++)
      for(int64_t s = 0; s < S; s++)
        for(int64_t n = 0; n < N; n++) {
          const int64_t i0 = h + H * s + H * S * n;
          for(int64_t d = 0; d < D; d++)
            dst[i0 + H * S * N * d] = src[h * D * S + d * S + s];
        }
    int64_t ne[4] = {H * S * N, D, 1, 1};
    ggml_tensor* out = hostConstant(name, GGML_TYPE_F32, ne, dst.data());
    return out;
  }

  // All-ones column for reductions via mul_mat: [len, 1] F32.
  ggml_tensor* onesTable(int64_t len) {
    const string name = "ones_" + Global::intToString(len);
    auto it = tensors.find(name);
    if(it != tensors.end())
      return it->second;
    vector<float> ones((size_t)len, 1.0f);
    int64_t ne[4] = {len, 1, 1, 1};
    return hostConstant(name, GGML_TYPE_F32, ne, ones.data());
  }

  // Layout conversion edges (L2 起生效; L1 阶段恒等,不插转换边):
  // NCHW->CS: cont(permute(x, 1,2,0,3)) + reshape_3d(C,S,N)  (v2.5 穷举验证)
  // CS->NCHW: reshape_4d(x,C,W,H,N) + permute(2,0,1,3) + cont (v2.5 穷举验证)
  ggml_tensor* toCS(ggml_tensor* x) {
    GGML_ASSERT(ggml_n_dims(x) <= 4);
    const int64_t C = x->ne[2], S = x->ne[0] * x->ne[1], N = x->ne[3];
    if(S == 1 && ggml_is_contiguous(x)) {
      // S3-8/O8 identity short-circuit: NCHW [1,1,C,N] and CS [C,1,N] share
      // the same memory layout; permute+cont would be an identity copy
      // (observed: all 1x1xCx1 bias/scale parameter edges). Return a
      // zero-copy reshape view -- same ne/nb semantics as the materialized
      // result, no CONT node, no dispatch.
      return ggml_reshape_3d(ctx, x, C, S, N);
    }
    ggml_tensor* p = ggml_permute(ctx, x, 1, 2, 0, 3);
    ggml_tensor* c = ggml_cont(ctx, p);
    return ggml_reshape_3d(ctx, c, C, S, N);
  }
  ggml_tensor* toNCHW(ggml_tensor* x) {
    GGML_ASSERT(ggml_n_dims(x) <= 4);
    // x is CS [C, S, N] (3D) or [C, S, N, 1]. S = W*H must be split with the
    // board dims: reshape to [C, W, H, N] then permute to [W, H, C, N].
    const int64_t C = x->ne[0];
    const int64_t S = x->ne[1];
    const int64_t N = ggml_n_dims(x) >= 3 ? x->ne[2] : 1;
    // S==1 covers the heads' global-pooled [C,1,1] outputs; otherwise the CS
    // spatial dim is the full board W*H (sub-board positions appear as 0/1
    // masked inputs, the tensor is still W*H wide).
    int64_t W, H;
    if(S == 1) { W = 1; H = 1; }
    else {
      GGML_ASSERT(S == (int64_t)nnXLen * nnYLen);
      W = nnXLen; H = nnYLen;
    }
    if(S == 1 && ggml_is_contiguous(x)) {
      // S3-8/O8 identity short-circuit: CS [C,1,N] and NCHW [1,1,C,N] share
      // the same memory layout; reshape+permute+cont would be an identity
      // copy. Zero-copy view with identical ne/nb semantics.
      return ggml_reshape_4d(ctx, x, 1, 1, C, N);
    }
    // N57 safety net: a strided CS tensor (the qkv window handed downstream when
    // KATAGO_VK_QKV_NOCONT=1) cannot go through the reshape below -- ggml_reshape_4d
    // asserts contiguity (ggml.c:3705). Materialize exactly like the pre-N57 path
    // would have; for every contiguous input this guard is a no-op.
    if(!ggml_is_contiguous(x)) { x = ggml_cont(ctx, x); }
    ggml_tensor* r4 = ggml_reshape_4d(ctx, x, C, W, H, N);
    ggml_tensor* p = ggml_permute(ctx, r4, 2, 0, 1, 3);
    return ggml_cont(ctx, p);
  }
  ggml_tensor* toF16(ggml_tensor* x) {
    return ggml_cast(ctx, x, GGML_TYPE_F16);
  }
  ggml_tensor* toF32(ggml_tensor* x) {
    return ggml_cast(ctx, x, GGML_TYPE_F32);
  }

  // Ensure a tensor is in the wanted layout/dtype, inserting conversion edges
  // if not (rule R1). Conversion results are cached per source tensor so
  // shared inputs (InputMask, stem outputs) only pay once.
  ggml_tensor* ensure(AnnotatedTensor& ann, Layout wantLayout, ggml_type wantDtype) {
    ggml_tensor* t = ann.t;
    if(ann.layout != wantLayout) {
      if(wantLayout == Layout::L_CS) {
        GGML_ASSERT(ann.layout == Layout::L_NCHW);
        auto& cached = convertCacheToCS[t];
        if(cached == NULL) {
          cached = toCS(t);
          nConvToCS++;
          if(getenv("KATAGO_VK_LAYOUTDBG_EDGE") != nullptr) {
            // S3-8 step-0 edge audit: one line per materialized conversion edge.
            fprintf(stderr, "LAYOUT-EDGE|op=%s|dir=toCS|srcNe=%lldx%lldx%lldx%lld|n=%d\n",
                currentOpName.c_str(), (long long)t->ne[0], (long long)t->ne[1],
                (long long)t->ne[2], (long long)t->ne[3], nConvToCS);
          }
        }
        t = cached;
      } else if(wantLayout == Layout::L_NCHW) {
        GGML_ASSERT(ann.layout == Layout::L_CS);
        auto& cached = convertCacheToNCHW[t];
        if(cached == NULL) {
          cached = toNCHW(t);
          nConvToNCHW++;
          if(getenv("KATAGO_VK_LAYOUTDBG_EDGE") != nullptr) {
            fprintf(stderr, "LAYOUT-EDGE|op=%s|dir=toNCHW|srcNe=%lldx%lldx%lldx%lld|n=%d\n",
                currentOpName.c_str(), (long long)t->ne[0], (long long)t->ne[1],
                (long long)t->ne[2], (long long)t->ne[3], nConvToNCHW);
          }
        }
        t = cached;
      }
      // L_MAT conversions are never inserted (C8-5): layout-orthogonal domain.
      ann.layout = wantLayout;
      ann.t = t;
    }
    if(ann.dtype != wantDtype) {
      if(wantDtype == GGML_TYPE_F16) {
        GGML_ASSERT(ann.dtype == GGML_TYPE_F32);
        t = toF16(t);
      } else {
        GGML_ASSERT(wantDtype == GGML_TYPE_F32 && ann.dtype == GGML_TYPE_F16);
        t = toF32(t);
      }
      ann.dtype = wantDtype;
      ann.t = t;
    }
    return t;
  }

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
    // Elementwise op with native ggml broadcasting (no explicit ggml_repeat):
    // the ggml binary kernels handle src1 with per-dim ne==1 by indexing, so
    // broadcasts cost nothing extra. ggml broadcasts src1 up to src0's shape,
    // so the elementwise-larger tensor must be src0. For commutative ops
    // (Add/Mul) that is always possible; for Div/Sub the operand order matters
    // (a/b != b/a), so when a is the smaller one it is materialized via
    // ggml_repeat first and then used as src0.
    bool aHas1 = false, bHas1 = false;
    for(int i = 0; i < 4; i++) {
      if(a->ne[i] == 1 && b->ne[i] != 1)
        aHas1 = true;
      if(b->ne[i] == 1 && a->ne[i] != 1)
        bHas1 = true;
      if(a->ne[i] != 1 && b->ne[i] != 1 && a->ne[i] != b->ne[i])
        throw StringError("VULKAN backend: shape mismatch in broadcast a ne=" +
          Global::intToString(a->ne[0]) + "," + Global::intToString(a->ne[1]) + "," + Global::intToString(a->ne[2]) + "," + Global::intToString(a->ne[3]) +
          " b ne=" + Global::intToString(b->ne[0]) + "," + Global::intToString(b->ne[1]) + "," + Global::intToString(b->ne[2]) + "," + Global::intToString(b->ne[3]));
    }
    testAssert(!(aHas1 && bHas1));  // one side must be elementwise-larger
    if(!aHas1 && !bHas1) {
      // same shape
      switch(op) {
      case 0: return ggml_add(ctx, a, b);
      case 1: return ggml_mul(ctx, a, b);
      case 2: return ggml_div(ctx, a, b);
      case 3: return ggml_sub(ctx, a, b);
      default: throw StringError("VULKAN backend: unknown binary op");
      }
    }
    if(bHas1) {
      // b is the smaller one: b broadcasts against a. Keep operand order.
      switch(op) {
      case 0: return ggml_add(ctx, a, b);
      case 1: return ggml_mul(ctx, a, b);
      case 2: return ggml_div(ctx, a, b);
      case 3: return ggml_sub(ctx, a, b);
      default: throw StringError("VULKAN backend: unknown binary op");
      }
    }
    // a is the smaller one (aHas1). Commutative ops: use b as src0. Non-
    // commutative (Div/Sub): materialize a to b's shape first, then a-op-b.
    switch(op) {
    case 0: return ggml_add(ctx, b, a);
    case 1: return ggml_mul(ctx, b, a);
    default: break;
    }
    int64_t targetNe[4];
    for(int i = 0; i < 4; i++)
      targetNe[i] = b->ne[i];
    ggml_tensor* ba = ggml_repeat(ctx, a, shapeRefFor(targetNe));
    switch(op) {
    case 2: return ggml_div(ctx, ba, b);
    case 3: return ggml_sub(ctx, ba, b);
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
    ggml_tensor* red;
    if(reduced >= 4096) {
      // Long rows: ggml's sum_rows shader reduces each row with a single
      // workgroup, serialized for long rows (the trunk-final mean-square has
      // reduced=138624 -> 305us). Two-stage reduction instead: reshape
      // [reduced, rest] as [chunk, reduced/chunk, rest] (zero-copy view),
      // sum_rows along the small chunk dim -> [1, reduced/chunk, rest] (that
      // many workgroups run in parallel), then reshape + sum_rows again.
      // Keeps exact F32 arithmetic (a mul_mat with an all-ones column would be
      // faster but the coopmat2 F16 conversion pushed topPolicyDelta 99% over
      // its limit).
      int64_t chunk = -1;
      for(int64_t cand : {1024LL, 768LL, 512LL, 384LL, 256LL, 192LL, 128LL, 96LL, 64LL}) {
        if(reduced % cand == 0) {
          chunk = cand;
          break;
        }
      }
      if(chunk > 0) {
        ggml_tensor* v = ggml_reshape_3d(ctx, flat, chunk, reduced / chunk, rest);
        ggml_tensor* s1 = isSum ? ggml_sum_rows(ctx, v) : ggml_mean(ctx, v);   // [1, n, rest]
        ggml_tensor* s2v = ggml_reshape_2d(ctx, s1, reduced / chunk, rest);
        red = isSum ? ggml_sum_rows(ctx, s2v) : ggml_mean(ctx, s2v);           // [1, rest]
      }
      else {
        red = isSum ? ggml_sum_rows(ctx, flat) : ggml_mean(ctx, flat);
      }
    }
    else {
      red = isSum ? ggml_sum_rows(ctx, flat) : ggml_mean(ctx, flat);  // [1, rest]
    }
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
    if(ggml_nelements(x) != dims[0] * dims[1] * dims[2] * dims[3])
      fprintf(stderr, "RESHAPE-DBG op=%s xne=%lld,%lld,%lld,%lld nl=%lld dims=%lld,%lld,%lld,%lld\n",
        currentOpName.c_str(), (long long)x->ne[0], (long long)x->ne[1], (long long)x->ne[2], (long long)x->ne[3],
        (long long)ggml_nelements(x), (long long)dims[0], (long long)dims[1], (long long)dims[2], (long long)dims[3]);
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
    for(auto& kv : inputs) {
      tensors[kv.first] = kv.second;
      // Graph inputs are uploaded in NCHW/F32 (F10): L1 default domain.
      annTensors[kv.first] = {kv.second, Layout::L_NCHW, GGML_TYPE_F32};
    }

    {
      // Built once for every prescan (see the member comment).  clear() first: a member outlives a
      // single build(), so stale entries would otherwise be invisible.
      producerIndex.clear();
      const auto& nodes = model.graph().node();
      for(int i = 0; i < (int)nodes.size(); i++)
        for(int j = 0; j < nodes[i].output_size(); j++)
          producerIndex[nodes[i].output(j)] = i;
    }

    prescanSilu();
    prescanSwish();
    prescanRms();
    prescanMaskReassoc();
    prescanSoftmaxScale();
    // C-N34: fa is opt-in via KATAGO_VK_FA (default off = bit-identical to
    // the previous build). Must run BEFORE prescanRope: prescanFa builds
    // faSites (q/k/v names) which prescanRope consumes to decide rope for v.
    if(faEnabled())
      prescanFa();  // EXPERIMENT b'
    prescanRope();
    prescanProjFusion();
    // prescanGammaAbsorb();  // sequence-1 path A: margin gate fails (1.04x), disabled
    // Pre-build every RMS weight as a standalone [C,1,1,1] constant (not a
    // reshape view) so the cgraph order becomes rms_norm -> mul with no node
    // between; ggml-vulkan's RMS_NORM_MUL fusion requires mul to directly
    // follow rms_norm. A reshape view would be expanded between them by
    // ggml_build_forward (mul's src0 is expanded before src1).
    for(auto& kv : rmsFusions) {
      if(!kv.second.active)
        continue;
      const onnx::TensorProto* t = findInitializer(kv.second.weightSrc);
      testAssert(t != NULL);
      vector<float> wv = initializerFloats(*t);
      int64_t ne[4] = {(int64_t)wv.size(), 1, 1, 1};
      rmsWeightViews[kv.second.weightSrc] =
        hostConstant(kv.second.weightSrc + "@c1", GGML_TYPE_F32, ne, wv.data());
    }
    // Batch dim for the shared RoPE 2D tables: N = max batch (from InputSpatial).
    {
      auto itS = tensors.find("InputSpatial");
      if(itS != tensors.end())
        ropeIdsN = itS->second->ne[3];
    }

    for(const onnx::NodeProto& node : model.graph().node()) {
      const string& op = node.op_type();
      const string& outName = node.output_size() > 0 ? node.output(0) : string();
      testAssert(node.output_size() == 1);
      currentOpName = node.name().empty() ? op : node.name();

      // S1 fusion: the Sigmoid/Mul pieces of a SwiGLU, and the Pow/ReduceMean/
      // Add/Sqrt pieces of an RMSNorm chain, are skipped (rebuilt as single ops).
      // Skip only named nodes: an empty name would otherwise absorb every
      // following unnamed node.
      if(!node.name().empty() &&
         (absorbedSiluNodes.count(node.name()) > 0 || absorbedRmsNodes.count(node.name()) > 0 ||
          absorbedMaskMulNodes.count(node.name()) > 0 || absorbedRopeNodes.count(node.name()) > 0 ||
          absorbedSoftmaxNodes.count(node.name()) > 0))
        continue;

      auto in = [&](int i) -> AnnotatedTensor {
        testAssert(i < node.input_size());
        const string& name = node.input(i);
        if(name.empty())
          throw StringError("VULKAN backend: op " + op + " is missing an optional input the interpreter requires");
        auto it = annTensors.find(name);
        if(it != annTensors.end())
          return it->second;
        // Inputs that are initializers (weights, scalars, RoPE tables) are
        // materialized as F32 tensors on first use.
        if(findInitializer(name) != NULL)
          return {initializerTensor(name, GGML_TYPE_F32), Layout::L_NCHW, GGML_TYPE_F32};
        throw StringError("VULKAN backend: tensor '" + name + "' used by op " + op + " was not produced yet (graph not topologically ordered?)");
      };
      auto registerOut = [&](ggml_tensor* t, Layout lay = Layout::L_NCHW, ggml_type dt = GGML_TYPE_F32) {
        if(layoutHintEnabled()) {
          // S3-8 production-side domain assignment: a hint overrides the
          // caller's hardcoded output domain for this tensor name.
          auto it = layoutHints.find(outName);
          if(it != layoutHints.end())
            lay = it->second;
        }
        tensors[outName] = t;
        annTensors[outName] = {t, lay, dt};
      };
      // Follow the layout/dtype of an input tensor for ops that preserve them.
      auto registerFollow = [&](const AnnotatedTensor& ann) {
        registerOut(ann.t, ann.layout, ann.dtype);
      };

      // L2 domain unification for elementwise binaries: both operands must live
      // in the same domain. If exactly one side is L_CS, the NCHW side is
      // converted (trunk target domain is CS; InputMask's CS copy is cached and
      // shared by all ~40 mask sites).
      auto unifyDomains = [&](AnnotatedTensor& a0, AnnotatedTensor& a1) {
        if(a0.layout == a1.layout)
          return;
        if(a0.layout == Layout::L_CS && a1.layout == Layout::L_NCHW)
          ensure(a1, Layout::L_CS, GGML_TYPE_F32);
        else if(a0.layout == Layout::L_NCHW && a1.layout == Layout::L_CS)
          ensure(a0, Layout::L_CS, GGML_TYPE_F32);
        else
          throw StringError("VULKAN backend: binary op across incompatible domains (" + currentOpName + ")");
      };

      if(op == "Conv") {
        // L2-B4: q/k/v projections of a blockstack are merged into one GEMM
        // (built on first encounter of any of the three Conv nodes).
        auto itF = projSegOf.find(outName);
        if(itF != projSegOf.end()) {
          ProjFusion& f = projFusions[itF->second];
          if(!f.built)
            buildFusedProjections(f);
          testAssert(f.built && tensors.count(outName) > 0);
          continue;
        }
        AnnotatedTensor annX = in(0);
        // B1: conv weights are stored natively in F16 (ggml-vulkan converts
        // F32 operands to F16 every eval on coopmat2 devices, and im2col
        // emits F16 regardless). Sequence ①: if this conv consumes a norm's
        // masked output, its gamma is absorbed by input channel here.
        ggml_tensor* w = convWeightWithGamma(node.input(1));
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
        if(kX == 1 && kY == 1) {
          // L2 §4.1: 1x1 conv (the linear-layer workhorse) -> direct mul_mat in
          // the CS domain: no im2col materialization, no conv-output cont.
          //   W [1,1,IC,OC] -> reshape_2d(IC, OC)          (zero-copy view)
          //   x [C,S,N]     -> reshape_2d(IC, S*N)         (zero-copy view)
          //   Y = mul_mat(W, x2) -> [OC, S*N], OC fastest
          //   out = reshape_3d(Y, OC, S, N)                (zero-copy view)
          ggml_tensor* x = ensure(annX, Layout::L_CS, GGML_TYPE_F32);
          const int64_t IC = w->ne[2], OC = w->ne[3];
          testAssert(x->ne[0] == IC);
          const int64_t S = x->ne[1], N = x->ne[2];
          ggml_tensor* W = ggml_reshape_2d(ctx, w, IC, OC);
          ggml_tensor* x2 = ggml_reshape_2d(ctx, x, IC, S * N);
          ggml_tensor* Y = ggml_mul_mat(ctx, W, x2);            // [OC, S*N]
          // C-N34 R1: unfused q/k/v projection GEMM that feeds fa.
          if(faEnabled() && faFeedConvOuts.count(outName) > 0)
            faFeedGemmPtrs.insert(Y);
          ggml_tensor* out = ggml_reshape_3d(ctx, Y, OC, S, N); // [OC, S, N]
          registerOut(out, Layout::L_CS, GGML_TYPE_F32);
        }
        else {
          // k>1: im2col needs NCHW, and the input may currently be in the CS domain.
          //
          // C1 REGRESSION FIX.  L2-A introduced the CS domain and left this branch reading `annX.t`
          // raw, under the assumption written in its comment: "k>1 (stem conv_spatial only)".  That
          // assumption was true for EVERY net in the tree at the time (b10c384 / b10c512 / b11c768
          // each have exactly ONE k>1 conv -- the stem), and false in general: a classic
          // two-3x3-per-block trunk (kata1-zhizi-b28c512nbt) has 122 of them, so from the second one
          // on the input is CS [C,S,N] and ggml_im2col's `is_2D` clause (ggml.c:4481,
          // a->ne[2] == b->ne[2]) aborted the process during graph build.  Before L2-A there was no
          // CS domain at all (every tensor was NCHW), which is why the same net ran then, only
          // slower (1x1 convs paid im2col too).
          //
          // `ensure` is a NO-OP when the tensor already has the wanted layout (see ensure()), so
          // single-spatial-conv nets -- all of them before this one -- keep a bit-identical graph;
          // the conversion is cached per source tensor, and both directions are already counted
          // (nConvToCS / nConvToNCHW, observable with KATAGO_VK_LAYOUTDBG_EDGE), so the price of the
          // classic trunk can be accounted instead of guessed.
          ggml_tensor* x = ensure(annX, Layout::L_NCHW, GGML_TYPE_F32);
          // Loud, in our own words: a channel mismatch here would otherwise surface as an upstream
          // GGML_ASSERT abort with no hint of which conv or which layout produced it.
          if(x->ne[2] != w->ne[2])
            throw StringError("VULKAN backend: conv channel mismatch at " + currentOpName
                              + " (input channels " + std::to_string((long long)x->ne[2])
                              + " vs kernel " + std::to_string((long long)w->ne[2])
                              + ", input layout id " + std::to_string((int)annX.layout)
                              + ") -- im2col needs the input in NCHW");
          registerOut(ggml_conv_2d_s1_ph(ctx, w, x), Layout::L_NCHW, GGML_TYPE_F32);
        }
      }
      else if(op == "MatMul") {
        AnnotatedTensor annA = in(0), annB = in(1);
        if(annA.layout == Layout::L_MAT || annB.layout == Layout::L_MAT) {
          // L_MAT domain attention (C18 verified):
          //  scores = qT·k: qT [D,S_q,H,N] (transpose output, K=D at ne0) x
          //    k [D,H,S,N] -> permute k to [D,S,H,N]; mul_mat -> [S_q,S_k,H,N].
          //  sv = probs·vT: probs [S_q,S_k,H,N] x vT [D,S,H,N] (transpose
          //    output). mul_mat(A,B) = A^T B with A [K,M], B [K,N]:
          //    A = permute(v, 1,2,0,3) -> [S_k, D, H, N]
          //    B = permute(probs, 1,0,2,3) -> [S_k, S_q, H, N]
          //    -> [D, S_q, H, N].
          GGML_ASSERT(annA.layout == annB.layout);   // L_MAT on both sides
          if(outName.find("/scores") != string::npos || outName.find("scores") == 0) {
            auto itFa = faSites.find(outName);
            if(itFa != faSites.end() && itFa->second.active) {
              // M1: build flash_attn_ext instead of scores matmul + softmax + sv.
              // q/k/v are [D,H,S,N] L_MAT; fa wants [D,S,H,N] (ne1=seq,
              // ne2=heads). Zero-dispatch strided views via ggml_view_4d with
              // swapped strides (fa shader uses p.nb01/nb02/nb03).
              const FaSite& f = itFa->second;
              auto itQ = tensors.find(f.qName), itK = tensors.find(f.kName), itV = tensors.find(f.vName);
              testAssert(itQ != tensors.end() && itK != tensors.end() && itV != tensors.end());
              ggml_tensor* q = itQ->second;   // [D,H,S,N]
              ggml_tensor* k = itK->second;
              ggml_tensor* v = itV->second;
              const int64_t D = q->ne[0], H = q->ne[1], S = q->ne[2], N = q->ne[3];
              testAssert(k->ne[0] == D && v->ne[0] == D);
              // [D,S,H,N] views: ne1=S (stride = old ne2 stride), ne2=H (old ne1)
              ggml_tensor* qv = ggml_view_4d(ctx, q, D, S, H, N, q->nb[2], q->nb[1], q->nb[3], 0);
              ggml_tensor* kv = ggml_view_4d(ctx, k, D, S, H, N, k->nb[2], k->nb[1], k->nb[3], 0);
              ggml_tensor* vv = ggml_view_4d(ctx, v, D, S, H, N, v->nb[2], v->nb[1], v->nb[3], 0);
              // mask: F16 [S_k,S_q,1,N] from the same source as the fallback
              // (separate cache per review: fallback keeps its F32 cache).
              // B2a: with faNoMask, skip the repeat+cast entirely and pass NULL
              // (ggml_flash_attn_ext accepts mask==NULL, ggml.c:5433). The ONNX
              // maskBias producer chain stays as orphan ggml tensors (built but
              // unreachable from the graph outputs => zero eval cost, §10.11).
              ggml_tensor* faMask = NULL;
              if(!faNoMask) {
                auto itM = annTensors.find(f.maskSrc);
                testAssert(itM != annTensors.end());
                if(faMaskF16Shared == NULL || faMaskF16SK != S || faMaskF16SQ != S || faMaskF16N != N) {
                  int64_t targetNe[4] = {S, S, 1, N};
                  ggml_tensor* full = ggml_repeat(ctx, itM->second.t, shapeRefFor(targetNe));   // F32
                  faMaskF16Shared = ggml_cast(ctx, full, GGML_TYPE_F16);
                  faMaskF16SK = S; faMaskF16SQ = S; faMaskF16N = N;
                }
                faMask = faMaskF16Shared;
              }
              ggml_tensor* r = ggml_flash_attn_ext(ctx, qv, kv, vv, faMask,
                                                    f.scale, 0.0f, 0.0f);   // [D,H,S,N]
              // Force F32 accumulation: fa defaults to f16 acc on fp16 devices
              // (GV:10864 f32acc = !fp16 || prec==F32 || bf16), which costs
              // precision -- b10c384 margin 1.08x fails. F32 acc restores the
              // hand-written chain's accuracy (margin gate AD-5).
              ggml_flash_attn_ext_set_prec(r, GGML_PREC_F32);
              // Register under scores/softmax/sv output names; downstream
              // attnnchw consumes svOut. The softmax/sv nodes are skipped.
              tensors[f.scoresOut] = r;
              annTensors[f.scoresOut] = {r, Layout::L_MAT, GGML_TYPE_F32};
              tensors[f.softmaxOut] = r;
              annTensors[f.softmaxOut] = {r, Layout::L_MAT, GGML_TYPE_F32};
              tensors[f.svOut] = r;
              annTensors[f.svOut] = {r, Layout::L_MAT, GGML_TYPE_F32};
              continue;
            }
            // scores[i,j] = sum_d q[d,i]*k[d,j]; softmax is along S_k (ONNX
            // axis 3), which must sit at ne0. mul_mat(a,b) computes
            // dst[m,n] = sum_k a[k,m]*b[k,n] -> [a.ne1, b.ne1]. With
            //   a = permute(k, 0,2,1,3) [D,S,H,N]  (K=D, ne1=S_k)
            //   b = permute(q, 0,2,1,3) [D,S,H,N]  (K=D, ne1=S_q)
            // dst = [S_k, S_q, H, N], and dst[m,n] = scores_LMAT where
            // ONNX scores[n,m] lives at (m=S_k,n=S_q): softmax along ne0=S_k.
            // mul_mat needs contiguous (non-transposed) operands, so the
            // permuted views are materialized.
            ggml_tensor* a = annB.t;   // RAW-OK: L_MAT-by-assert (both operands asserted L_MAT at the branch head)
            if(a->ne[1] < a->ne[2])
              a = ggml_cont(ctx, ggml_permute(ctx, a, 0, 2, 1, 3));   // -> [D,S,H,N]
            ggml_tensor* b = annA.t;   // RAW-OK: L_MAT-by-assert (qT [D,H,S,N], identity in L_MAT)
            if(b->ne[1] < b->ne[2])
              b = ggml_cont(ctx, ggml_permute(ctx, b, 0, 2, 1, 3));   // -> [D,S,H,N]
            // EXPERIMENT b': F16 operands -> F16 tensor-core GEMM path
            // (coopmat2 consumes F16 natively; F32 inputs get converted
            // device-side anyway, GV:9246 -- this makes it explicit and skips
            // the conversion; softmax below stays the exact soft_max_ext).
            a = ggml_cast(ctx, a, GGML_TYPE_F16);
            b = ggml_cast(ctx, b, GGML_TYPE_F16);
            ggml_tensor* r = ggml_mul_mat(ctx, a, b);   // [S_k, S_q, H, N] F32 out
            registerOut(r, Layout::L_MAT, GGML_TYPE_F32);
          }
          else if(outName.find("/sv") != string::npos) {
            if(faAbsorbedSv.count(outName) > 0)
              continue;   // M1: already produced by flash_attn_ext in the scores branch
            // sv[d,i] = sum_j probs[i,j] * v[d,j] (j = S_k contraction).
            // vT (transpose) is identity in L_MAT, so v is [D,H,S,N].
            // mul_mat(A,B) = A^T B with A [K,M], B [K,N]:
            //   A = permute(v, 1,2,0,3) -> [S_k, D, H, N]  (K=S_k) [contiguous]
            //   B = probs as-is [S_k, S_q, H, N]
            //   -> [D, S_q, H, N].
            ggml_tensor* a = ggml_cont(ctx, ggml_permute(ctx, annB.t, 1, 2, 0, 3)); // v [D,H,S,N]->[S_k,D,H,N]
            ggml_tensor* b = annA.t;   // RAW-OK: L_MAT-by-assert (probs [S_k,S_q,H,N] as-is)
            // EXPERIMENT b': F16 operands for the tensor-core path (see scores)
            a = ggml_cast(ctx, a, GGML_TYPE_F16);
            b = ggml_cast(ctx, b, GGML_TYPE_F16);
            ggml_tensor* r = ggml_mul_mat(ctx, a, b);   // [D, S_q, H, N]
            registerOut(r, Layout::L_MAT, GGML_TYPE_F32);
          }
          else {
            throw StringError("VULKAN backend: L_MAT MatMul with unrecognized role (" + outName + ")");
          }
        }
        else {
          // B1: when the right operand is a stored weight, keep it natively F16
          // to skip the per-eval F32->F16 conversion in ggml_vk_mul_mat. Live
          // (non-initializer) operands stay F32.
          ggml_tensor* b;
          if(findInitializer(node.input(1)) != NULL)
            b = initializerTensor(node.input(1), GGML_TYPE_F16);
          else
            b = annB.t;   // RAW-OK: GEMM-any-layout (mul_mat consumes either domain; shape is the annotation's)
          registerOut(matmul(annA.t, b));
        }
      }
      else if(op == "Add") {
        auto itM = maskReassocs.find(outName);
        if(itM != maskReassocs.end() && itM->second.active) {
          // C19-1: Add(A, Mul(B, InputMask)) -> Mul(Add(A, B), InputMask).
          // A is the masked residual, B the unmasked projection output; the
          // reassociation is exact for 0/1 masks and linear 1x1 convs, and it
          // lets the Add fuse into B's MUL_MAT epilogue.
          const MaskReassoc& f = itM->second;
          AnnotatedTensor annA = annTensors.count(f.resSrc) > 0 ? annTensors[f.resSrc] : in(0);
          AnnotatedTensor annB = annTensors.count(f.projSrc) > 0 ? annTensors[f.projSrc] : in(1);
          unifyDomains(annA, annB);
          ggml_tensor* sum = binary(annA.t, annB.t, 0);
          // The mask must be applied in the unified domain: re-mask with InputMask.
          AnnotatedTensor annMask = annTensors.count("InputMask") > 0 ? annTensors["InputMask"] : in(0);
          // Mask is the same tensor on both sides; pick a domain via unify with sum.
          Layout sumLay = annA.layout;
          AnnotatedTensor annM = annTensors["InputMask"];
          if(annM.layout != sumLay)
            ensure(annM, sumLay, GGML_TYPE_F32);
          ggml_tensor* out = binary(sum, annM.t, 1);
          registerOut(out, sumLay, GGML_TYPE_F32);
        }
        else {
          AnnotatedTensor a0 = in(0), a1 = in(1);
          unifyDomains(a0, a1);
          registerOut(binary(a0.t, a1.t, 0), a0.layout, GGML_TYPE_F32);
        }
      }
      else if(op == "Mul") {
        auto itSwish = swishFusions.find(outName);
        if(itSwish != swishFusions.end() && itSwish->second.active) {
          // F-4: plain swish x*sigmoid(x) == ggml_silu(x) (single op).
          auto itX = annTensors.find(itSwish->second.xSrc);
          testAssert(itX != annTensors.end());
          registerOut(ggml_silu(ctx, itX->second.t), itX->second.layout, GGML_TYPE_F32);
        }
        else {
        auto itSilu = siluFusions.find(outName);
        if(itSilu != siluFusions.end() && itSilu->second.active) {
          // SwiGLU: silu(a) * gate, fused into a single swiglu kernel
          ggml_tensor* a = tensors.count(itSilu->second.siluInput) > 0 ? tensors[itSilu->second.siluInput] : initializerTensor(itSilu->second.siluInput, GGML_TYPE_F32);
          ggml_tensor* gate = tensors.count(itSilu->second.gateSrc) > 0 ? tensors[itSilu->second.gateSrc] : initializerTensor(itSilu->second.gateSrc, GGML_TYPE_F32);
          // The gate and silu inputs share the layout of the linear layers'
          // outputs; L2 trunk runs in CS so both are CS.
          Layout lay = Layout::L_NCHW;
          auto itA = annTensors.find(itSilu->second.siluInput);
          if(itA != annTensors.end())
            lay = itA->second.layout;
          registerOut(ggml_swiglu_split(ctx, a, gate), lay, GGML_TYPE_F32);
        }
        else {
          AnnotatedTensor a0 = in(0), a1 = in(1);
          unifyDomains(a0, a1);
          registerOut(binary(a0.t, a1.t, 1), a0.layout, GGML_TYPE_F32);
        }
        }
      }
      else if(op == "Div") {
        auto itR = rmsFusions.find(outName);
        if(itR != rmsFusions.end() && itR->second.active) {
          // RMSNorm fused, L2 CS direct: x is [C, S, N] (CS) with the channel
          // dim at ne0, so ggml_rms_norm reduces the right axis with no permute
          // round-trip at all (design §4.2). weight is [1,1,C,1] -> [C,1,1,1].
          // The weight view is created BEFORE the rms_norm so the cgraph order
          // is rms_norm -> mul with nothing in between; ggml-vulkan's
          // RMS_NORM_MUL fusion (ggml_vk_can_fuse) requires the mul node to
          // directly follow the rms_norm node.
          const RmsFusion& f = itR->second;
          AnnotatedTensor annX = in(0);
          ggml_tensor* x = ensure(annX, Layout::L_CS, GGML_TYPE_F32);   // [C,S,N]
          ggml_tensor* rn = ggml_rms_norm(ctx, x, f.eps);               // [C,S,N]
          ggml_tensor* sc;
          if(gammaAbsorbedNorm.count(outName) > 0) {
            // Sequence ①: gamma absorbed into the downstream conv weights; the
            // next node is the mask Mul, and ggml-vulkan's RMS_NORM_MUL fusion
            // eats rms_norm -> mul(mask) (mask is spatial broadcast, H-2).
            sc = rn;
          }
          else {
            // The [C,1,1,1] weight view was pre-built before the node loop so
            // the cgraph order is rms_norm -> mul with nothing in between
            // (enables ggml-vulkan's RMS_NORM_MUL fusion).
            auto itW = rmsWeightViews.find(f.weightSrc);
            testAssert(itW != rmsWeightViews.end());
            sc = ggml_mul(ctx, rn, itW->second);                        // [C,S,N]
          }
          // The weight Mul was absorbed too, so register this result under both
          // the Div output and the Mul output names.
          tensors[outName] = sc;
          annTensors[outName] = {sc, Layout::L_CS, GGML_TYPE_F32};
          if(!f.mulOut.empty()) {
            tensors[f.mulOut] = sc;
            annTensors[f.mulOut] = {sc, Layout::L_CS, GGML_TYPE_F32};
          }
        }
        else {
          AnnotatedTensor a0 = in(0), a1 = in(1);
          unifyDomains(a0, a1);
          registerOut(binary(a0.t, a1.t, 2), a0.layout, GGML_TYPE_F32);
        }
      }
      else if(op == "Sub") {
        AnnotatedTensor a0 = in(0), a1 = in(1);
        unifyDomains(a0, a1);
        registerOut(binary(a0.t, a1.t, 3), a0.layout, GGML_TYPE_F32);
      }
      else if(op == "Relu") {
        AnnotatedTensor ann = in(0);
        registerOut(ggml_relu(ctx, ann.t), ann.layout, GGML_TYPE_F32);
      }
      else if(op == "Sigmoid") {
        AnnotatedTensor ann = in(0);
        registerOut(ggml_sigmoid(ctx, ann.t), ann.layout, GGML_TYPE_F32);
      }
      else if(op == "Tanh") {
        AnnotatedTensor ann = in(0);
        registerOut(ggml_tanh(ctx, ann.t), ann.layout, GGML_TYPE_F32);
      }
      else if(op == "Softplus") {
        AnnotatedTensor ann = in(0);
        registerOut(ggml_softplus(ctx, ann.t), ann.layout, GGML_TYPE_F32);
      }
      else if(op == "Sqrt") {
        AnnotatedTensor ann = in(0);
        registerOut(ggml_sqrt(ctx, ann.t), ann.layout, GGML_TYPE_F32);
      }
      else if(op == "Identity") {
        registerFollow(in(0));
      }
      else if(op == "Pow") {
        // Only Pow(x, const 2) is used (RMSNorm squaring).
        const onnx::TensorProto* e = findInitializer(node.input(1));
        testAssert(e != NULL);
        vector<float> expo = initializerFloats(*e);
        testAssert(expo.size() == 1 && expo[0] == 2.0f);
        AnnotatedTensor ann = in(0);
        registerOut(ggml_mul(ctx, ann.t, ann.t), ann.layout, GGML_TYPE_F32);
      }
      else if(op == "ReduceMean" || op == "ReduceSum" || op == "ReduceMax") {
        const onnx::TensorProto* axes = findInitializer(node.input(1));
        testAssert(axes != NULL);
        vector<int64_t> ax = initializerInt64(*axes);
        // Reductions (trunk-tip global means, policy/value head stats) run in
        // the NCHW domain (P1); CS inputs convert back once (few sites, §4.5).
        AnnotatedTensor ann = in(0);
        ggml_tensor* x = ensure(ann, Layout::L_NCHW, GGML_TYPE_F32);
        if(op == "ReduceMax")
          registerOut(reduceMaxAxes(x, ax));
        else
          registerOut(reduceAxes(x, ax, op == "ReduceSum"));
      }
      else if(op == "Gather") {
        int64_t axis = 0;
        if(hasAttr(node, "axis"))
          axis = getAttr(node, "axis").i();
        // L2-B: if the gather's data input is a rope-site reshape output and it
        // lives in L_MAT (built by the Reshape branch), rebuild the entire rope
        // subgraph with 3D ids -- zero materializations (C18).
        auto itRope = ropeSites.find(node.input(0));
        if(itRope != ropeSites.end() && itRope->second.active) {
          const RopeSite& site = itRope->second;
          testAssert(axis == 2);
          AnnotatedTensor ann = in(0);      // [D,H,S,N] L_MAT
          GGML_ASSERT(ann.layout == Layout::L_MAT);
          ggml_tensor* q = ann.t;           // RAW-OK: L_MAT-by-assert (see the GGML_ASSERT above)
          const int64_t D = site.D, H = site.H, S = site.S;
          const int64_t N = q->ne[3];
          // Layout the tensor as [H,S,N,D] contiguous so get_rows' ne0 is
          // large (its shader uses ne0 as the x dimension; ne0=H=6 wastes
          // 506/512 lanes -- that was the 393us regression). Then flatten to
          // [H*S*N, D]: rows are (h,s,n) with h fastest, columns are d.
          // get_rows selects along ne1 (=D) with a 1D swapidx table:
          //   dst[i0][i1] = a[i0][swapidx[i1]]  (row-internal channel swap).
          ggml_tensor* hsnd = ggml_cont(ctx, ggml_permute(ctx, q, 3, 0, 1, 2)); // [H,S,N,D]
          ggml_tensor* q2 = ggml_reshape_2d(ctx, hsnd, H * S * N, D);          // [H*S*N, D]
          // 2) 1D swapidx as the row index (all sites share the same table)
          ggml_tensor* ids = initializerTensor(site.swapIdxInit, GGML_TYPE_I32);
          // 3) sw = get_rows(q2, ids) -> [H*S*N, D]
          ggml_tensor* sw2 = ggml_get_rows(ctx, q2, ids);
          // 4) rotate in 2D: t1 = q2*cos2, t2 = sw2*sin2, out = t1+t2, where
          //    cos2/sin2 are the same [H*S*N, D] layout.
          ggml_tensor* cos2 = ropeCosTable2D(site, false);
          ggml_tensor* sin2 = ropeCosTable2D(site, true);
          ggml_tensor* t1 = ggml_mul(ctx, q2, cos2);
          ggml_tensor* t2 = ggml_mul(ctx, sw2, sin2);
          ggml_tensor* out2 = ggml_add(ctx, t1, t2);                    // [H*S*N, D]
          // 5) back to [D,H,S,N]: reshape [H,S,N,D] then permute to [D,H,S,N]
          ggml_tensor* out4 = ggml_reshape_4d(ctx, out2, H, S, N, D);   // view
          ggml_tensor* ret = ggml_cont(ctx, ggml_permute(ctx, out4, 1, 2, 3, 0)); // [D,H,S,N]
          // Register under the rope_out (Add output) name; the gather's own
          // output name (rope_swap) is only an intermediate.
          tensors[site.addOut] = ret;
          annTensors[site.addOut] = {ret, Layout::L_MAT, GGML_TYPE_F32};
        }
        else {
          ggml_tensor* idx = initializerTensor(node.input(1), GGML_TYPE_I32);
          // RoPE swaps run inside the NCHW attention chain (L2-A keeps the
          // hand-written attention in NCHW; the 3D-ids zero-materialization is
          // L2-B). Convert CS inputs back.
          AnnotatedTensor ann = in(0);
          ggml_tensor* x = ensure(ann, Layout::L_NCHW, GGML_TYPE_F32);
          registerOut(gather(x, idx, axis));
        }
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
        AnnotatedTensor ann = in(0);
        if(ann.layout == Layout::L_MAT) {
          // L_MAT domain: ONNX Transpose [0,1,3,2] swaps the D and S axes of
          // the underlying [N,H,D,S] tensor.
          //  * qT/vT inputs are [D,H,S,N] (head-major, ne1=H < ne2=S): the
          //    swap is absorbed by the CS layout -> identity (verified).
          //  * svT input is sv [D,S_q,H,N] (ne1=S > ne2=H): swap S<->H ->
          //    permute(0,2,1,3) view -> [D,H,S,N].
          testAssert(perm.size() == 4 && perm[0] == 0 && perm[1] == 1 && perm[2] == 3 && perm[3] == 2);
          if(ann.t->ne[1] > ann.t->ne[2]) {
            ggml_tensor* v = ggml_permute(ctx, ann.t, 0, 2, 1, 3);
            registerOut(v, Layout::L_MAT, GGML_TYPE_F32);
          }
          else {
            registerOut(ann.t, Layout::L_MAT, GGML_TYPE_F32);
          }
        }
        else {
          // attention-internal transpose: keep NCHW semantics (permutation of
          // the NCHW tensor; see transposeOp which handles any 4D shape).
          ggml_tensor* x = ensure(ann, Layout::L_NCHW, GGML_TYPE_F32);
          registerOut(transposeOp(x, perm));
        }
      }
      else if(op == "Reshape") {
        const onnx::TensorProto* shape = findInitializer(node.input(1));
        testAssert(shape != NULL);
        vector<int64_t> target = initializerInt64(*shape);
        AnnotatedTensor ann = in(0);
        // L2-B: q/k/v head-split reshape [0,H,D,S] with a CS input is a
        // zero-copy reshape_4d(x, D, H, S, N) -> [D,H,S,N] (head-major
        // c=h*D+d; verified in l2b3-rope-verify.py).
        bool headSplit = ann.layout == Layout::L_CS && target.size() >= 4 &&
                         target[0] == 0 && target[3] > 1 &&
                         target[1] * target[2] == ann.t->ne[0] &&
                         target[3] == ann.t->ne[1];
        if(headSplit) {
          const int64_t D = target[2], H = target[1], S = target[3], N = ann.t->ne[2];
          // N57 slice-1: a non-contiguous L_CS source (the strided qkv window)
          // cannot go through ggml_reshape_4d (GGML_ASSERT(ggml_is_contiguous),
          // ggml.c:3705); express the same head-major c = h*D+d split as a
          // stride-preserving 4D view instead (identical tensor for a
          // contiguous source, so that path keeps using RESHAPE).
          ggml_tensor* v = ggml_is_contiguous(ann.t)
                             ? ggml_reshape_4d(ctx, ann.t, D, H, S, N)
                             : ggml_view_4d(ctx, ann.t, D, H, S, N,
                                            ann.t->nb[0] * (size_t)D, ann.t->nb[1], ann.t->nb[2], 0);
          registerOut(v, Layout::L_MAT, GGML_TYPE_F32);
        }
        else if(ann.layout == Layout::L_MAT && target.size() >= 4 && target[0] == 0 &&
                target[1] == ann.t->ne[0] * ann.t->ne[1]) {
          // attnnchw reshape: [D,H,S,N] L_MAT -> [C,S,N] CS (c = h*D+d).
          const int64_t C = ann.t->ne[0] * ann.t->ne[1];
          const int64_t S = ann.t->ne[2];
          const int64_t N = ann.t->ne[3];
          ggml_tensor* c = ggml_is_contiguous(ann.t) ? ann.t : ggml_cont(ctx, ann.t);
          ggml_tensor* out = ggml_reshape_3d(ctx, c, C, S, N);
          registerOut(out, Layout::L_CS, GGML_TYPE_F32);
        }
        else {
          // L2-A: reshape keeps NCHW semantics (attention chain, mask reshape);
          // CS inputs convert back once per source.
          ggml_tensor* x = ensure(ann, Layout::L_NCHW, GGML_TYPE_F32);
          registerOut(reshapeOp(x, target));
        }
      }
      else if(op == "Concat") {
        int64_t axis = getAttr(node, "axis").i();
        testAssert(axis >= 0 && axis < 4);
        // C1 audit: ggml_concat works on whatever it is given, and the axis arithmetic below
        // (ggmlAxis = 3 - ONNX axis) is a statement about NCHW axes.  A CS input would therefore
        // concatenate along a different logical axis than the ONNX node asked for -- and it would do
        // so SILENTLY, because the resulting ggml shape stays self-consistent.  All inputs must be in
        // ONE domain, so the requirement is asserted instead of assumed (this is the same class of
        // bug as the conv one this batch fixes: an implicit layout assumption).
        int ggmlAxis = 3 - (int)axis;
        const Layout accLayout = in(0).layout;
        ggml_tensor* acc = in(0).t;   // RAW-OK: same-domain-concat (asserted in the loop below)
        for(int i = 1; i < node.input_size(); i++) {
          testAssert(in(i).layout == accLayout);
          acc = ggml_concat(ctx, acc, in(i).t, ggmlAxis);   // RAW-OK: same-domain-concat
        }
        registerOut(acc);
      }
      else if(op == "Softmax") {
        if(faAbsorbedSoftmax.count(outName) > 0)
          continue;   // M1: already produced by flash_attn_ext in the scores branch
        int64_t axis = getAttr(node, "axis").i();
        if(axis != 3)
          throw StringError("VULKAN backend: Softmax is only supported on the last ONNX axis");
        auto itS = softmaxAbsorbs.find(outName);
        if(itS != softmaxAbsorbs.end() && itS->second.active) {
          // C19-2: Softmax(Add(Mul(scores, scale), mask)) ->
          // soft_max_ext(scores, mask_full, scale). scores is [S_k, S_q, H, N]
          // (L_MAT) or [S_k,S_q,H,N] (NCHW from matmul); mask is materialized
          // [S_k,S_q,1,N] (F21). In L_MAT the mask layout is orthogonal.
          const SoftmaxScaleAbsorb& f = itS->second;
          AnnotatedTensor annS = annTensors.count(f.scoresSrc) > 0 ? annTensors[f.scoresSrc] : in(0);
          AnnotatedTensor annM = annTensors.count(f.maskSrc) > 0 ? annTensors[f.maskSrc] : in(1);
          GGML_ASSERT(annS.layout == Layout::L_NCHW || annS.layout == Layout::L_MAT);
          const int64_t SK = annS.t->ne[0], SQ = annS.t->ne[1], N = annS.t->ne[3];
          if(maskFullShared == NULL || maskFullSK != SK || maskFullSQ != SQ || maskFullN != N) {
            int64_t targetNe[4] = {SK, SQ, 1, N};
            maskFullShared = ggml_repeat(ctx, annM.t, shapeRefFor(targetNe));
            maskFullSK = SK; maskFullSQ = SQ; maskFullN = N;
          }
          registerOut(ggml_soft_max_ext(ctx, annS.t, maskFullShared, f.scale, 0.0f),
                      annS.layout, GGML_TYPE_F32);
        }
        else {
          AnnotatedTensor ann = in(0);
          registerOut(ggml_soft_max(ctx, ann.t), ann.layout, GGML_TYPE_F32);
        }
      }
      else {
        throw StringError("VULKAN backend: unsupported ONNX op '" + op + "'");
      }
    }
    if(getenv("KATAGO_VULKAN_LAYOUTDBG") != nullptr) {
      fprintf(stderr, "LAYOUT-DBG: convToCS=%d convToNCHW=%d ropeSites=%d\n", nConvToCS, nConvToNCHW, (int)ropeSites.size());
      for(auto& kv : ropeSites) {
        const RopeSite& s = kv.second;
        fprintf(stderr, "  rope %s D=%lld H=%lld S=%lld cos=%s sin=%s\n",
          kv.first.c_str(), (long long)s.D, (long long)s.H, (long long)s.S,
          s.cosInit.c_str(), s.sinInit.c_str());
      }
    }
  }
};

//--------------------------------------------------------------

struct ComputeContext {
  int nnXLen;
  int nnYLen;
  enabled_t useFP16Mode;

  // T3 (t3-shell-plan.md section 6, step 1): the tuner cache lives under homeDataDir
  // (mirroring OpenCL's <homeDataDir>/opencltuning/), so the override has to survive
  // createComputeContext -- it used to be discarded here.
  std::string homeDataDirOverride;

  // T3 user-facing tuner mode (cfg `vulkanTune`, t3-shell-plan.md section 2.2).  Read HERE
  // because createComputeContext owns the ConfigParser; the shell must not read one.  A bad
  // value is a hard error: it is the operator's own config, and silently falling back to auto
  // would hide a typo behind "it tuned anyway".
  VulkanTuner::TunePolicy tunePolicy;

  // ONE tune decision per (evaluator, device), taken by whichever NN server thread
  // creates its handle first; the others reuse it and only perform their own per-context
  // install (section 2).  Without the single flight, N threads would race for the same
  // cache file; without the per-context install, only one context would get the rows and
  // the rest would silently fall back to stock -- i.e. fabricated evidence.
  std::mutex tuneMutex;
  std::map<int, std::shared_ptr<VulkanTuner::Decision>> tuneDecisions;

  ComputeContext(int xLen, int yLen, enabled_t fp16Mode, const std::string& homeDataDir)
    : nnXLen(xLen), nnYLen(yLen), useFP16Mode(fp16Mode), homeDataDirOverride(homeDataDir)
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

  // T3 user-facing mode (t3-shell-plan.md section 2.2): cfg `vulkanTune` = auto (default) or
  // off (read-only).  The dev master switch `KATAGO_VK_TUNE=0` is resolved inside the shell and
  // is stronger; the env-overrides-cfg rule therefore lives there, not here.
  string vulkanTune = "auto";
  if(cfg.contains("vulkanTune"))
    vulkanTune = cfg.getString("vulkanTune");
  if(vulkanTune != "auto" && vulkanTune != "off")
    throw StringError("Vulkan backend: vulkanTune must be \"auto\" or \"off\", got \"" + vulkanTune + "\"");

  // scale8 (1/8 activation rescaling) is applied here, before the graph is
  // built, exactly like the ONNX/TensorRT backends do. Transformer models are
  // not eligible and are skipped automatically.
  loadedModel->maybeApplyScale8(false, logger);

  // S1-③: keep the requested FP16 mode on the context so isUsingFP16 can
  // report it truthfully instead of hard-coding false (the Vulkan backend
  // itself does not gate on it -- ggml picks the fp16 path by device
  // capability -- but the request must be observable for tooling/tests).
  ComputeContext* ctx = new ComputeContext(nnXLen, nnYLen, useFP16Mode, homeDataDirOverride);
  ctx->tunePolicy.off = (vulkanTune == "off");
  ctx->tunePolicy.modeText = ctx->tunePolicy.off ? "off(cfg)" : "auto(cfg)";
  if(logger != NULL)
    logger->write("VULKAN backend: creating compute context for " +
                  Global::intToString(nnXLen) + "x" + Global::intToString(nnYLen) +
                  (homeDataDirOverride.empty() ? string() : (" (homeDataDir=" + homeDataDirOverride + ")")) +
                  " (vulkanTune=" + vulkanTune + ")");
  return ctx;
}

void NeuralNet::freeComputeContext(ComputeContext* computeContext) {
  delete computeContext;
}

//--------------------------------------------------------------

// ONE implementation of "force f32 accumulation on the F16-src0 mul_mats".  Both the env override
// (KATAGO_VK_MMPREC_F32) and the REQUESTED precision come through here, so they cannot drift into
// two copies: the first version of F1 had this loop twice (same predicate, same API, but one edit
// could have changed only one of them, and "两处拷贝靠纪律" is exactly what this project refuses).
static void forceF32AccOnF16MulMats(ggml_cgraph* g, bool global,
                                    const unordered_set<const ggml_tensor*>* faFeedOnly) {
  const int nNodes = ggml_graph_n_nodes(g);
  for(int i = 0; i < nNodes; i++) {
    ggml_tensor* node = ggml_graph_node(g, i);
    if(node->op != GGML_OP_MUL_MAT || node->src[0]->type != GGML_TYPE_F16)
      continue;
    if(global || (faFeedOnly != nullptr && faFeedOnly->count(node) > 0))
      ggml_mul_mat_set_prec(node, GGML_PREC_F32);
  }
}

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
    // B2a (next-temp §10.11): FA sites drop the mask tensor when the handle is
    // requireExactNNLen (mask provably all-ones, nneval hard-asserts full board).
    // TEST-ONLY override: KATAGO_VK_FA_NULL unset = flag-driven (production);
    //   = "1"/any = force on (only valid with full-board inputs);
    //   = "0"      = force off (A/B legs + rollback under the same binary).
    const char* faNullEnv = getenv("KATAGO_VK_FA_NULL");
    const bool faNoMask =
      (faNullEnv != nullptr && !(faNullEnv[0] == '0' && faNullEnv[1] == '\0'))
        ? true
        : (faNullEnv != nullptr ? false : requireExactNNLen);

    // Debug cross-check: KATAGO_VULKAN_CPU=1 runs the same graph on the ggml
    // CPU backend instead of Vulkan (validates the interpreter in isolation).
    const bool cpucrossCheck =
      getenv("KATAGO_VULKAN_CPU") != nullptr && getenv("KATAGO_VULKAN_CPU")[0] == '1';
    if(cpucrossCheck) {
      backend = ggml_backend_cpu_init();
      if(backend == NULL)
        throw StringError("VULKAN backend: could not initialize the ggml CPU backend");
    }
    else {
      // C-N33: serialize first-time device initialization. Two NN server
      // threads concurrently calling ggml_backend_vk_init raced on device
      // enumeration (p0_nst2_rawnn crash: dual "Found 2 Vulkan devices" prints
      // then vkCreateFence Invalid device, exit 0xC0000409; 0/15 reproductions
      // since -- race is timing-dependent, so fix defensively). The mutex
      // scopes only the init call; graph building below stays parallel.
      static std::mutex vkInitMutex;
      std::lock_guard<std::mutex> vkInitLock(vkInitMutex);
      backend = ggml_backend_vk_init(deviceIdx >= 0 ? (size_t)deviceIdx : 0);
      if(backend == NULL)
        throw StringError("VULKAN backend: could not initialize Vulkan backend for device " + Global::intToString(deviceIdx));
    }

    // --- T3 tuner hook (t3-shell-plan.md section 6, step 1) --------------------------
    // Placement is a hard requirement, not a convenience. It must be
    //   * after the backend exists -- the fingerprint and the install both need it, and
    //     the fingerprint is what names the cache file;
    //   * before anything can select a pipeline, i.e. before the ONNX/graph build below,
    //     because the pre-record path stops SELECTING once it replays (section 5) and a
    //     row installed after that would never be read.
    // The single flight (per evaluator+device) and the per-context install are what make
    // "1 decision, N installs" true instead of hoped for.  Nothing between here and the
    // install may query the selector, so the install cannot race the freeze.
    // Skipped on the KATAGO_VULKAN_CPU cross-check path: there is no Vulkan device to
    // describe and no Vulkan pipeline to tune.
    if(!cpucrossCheck) {
      const int effectiveDeviceIdx = deviceIdx >= 0 ? deviceIdx : 0;
      // The model half of the cache's outer key (t1-schema.md section 4): two models can
      // share the device, board size and batch size and still need different tiles, because
      // the model decides the (m,k) shapes.  Uses the model's OWN sha256 (already computed
      // when the model was loaded -- no extra 64-355 MB hash on every startup).  The sha goes
      // into the file NAME truncated to 16 hex characters (an identity, not a checksum:
      // mv+name are the backstop), the name only into the header.
      const ModelDesc& md = loadedModel->modelDesc;
      VulkanTuner::ModelIdentity modelKey;
      modelKey.version = md.modelVersion;
      modelKey.sha16 = md.sha256.empty() ? "none" : md.sha256.substr(0, 16);
      modelKey.name = md.name;
      std::shared_ptr<VulkanTuner::Decision> tuneDecision;
      {
        // The decision map is per ComputeContext and NN server threads are created
        // concurrently, so the single flight needs this lock.
        std::lock_guard<std::mutex> tuneLock(ctx->tuneMutex);
        std::map<int, std::shared_ptr<VulkanTuner::Decision>>::iterator it =
          ctx->tuneDecisions.find(effectiveDeviceIdx);
        if(it == ctx->tuneDecisions.end()) {
          it = ctx->tuneDecisions.emplace(
            effectiveDeviceIdx,
            VulkanTuner::makeDecision(
              backend, effectiveDeviceIdx, ctx->homeDataDirOverride,
              ctx->nnXLen, ctx->nnYLen, maxBatchSize, ctx->useFP16Mode, modelKey, ctx->tunePolicy)
          ).first;
        }
        tuneDecision = it->second;
      }
      if(tuneDecision != nullptr)
        VulkanTuner::installForHandle(tuneDecision, backend);
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
    OnnxToGgml builder(ggmlCtx, wctx, maxBatchSize, ctx->nnXLen, ctx->nnYLen, model, faNoMask);
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
    // getOutput decodes in NCHW (channel-strided), so CS-domain head outputs
    // get a CS->NCHW conversion edge here (design §11.3 "heads 出口插转换边").
    auto getOut = [&](const char* name) -> ggml_tensor* {
      auto it = builder.tensors.find(name);
      if(it == builder.tensors.end())
        throw StringError(string("VULKAN backend: graph is missing expected output ") + name);
      ggml_tensor* t = it->second;
      auto aIt = builder.annTensors.find(name);
      if(aIt != builder.annTensors.end() && aIt->second.layout == OnnxToGgml::Layout::L_CS) {
        // outPolicyPass is [N,2,1,1]-ish flat; converting keeps getOutput intact.
        t = builder.toNCHW(t);
      }
      return t;
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

    // KATAGO_VK_MMPREC_F32 (default off = bit-identical): force F32 accumulation
    // on mul_mat with an F16 src0 (f16acc -> f32acc, F16 storage kept).
    //   '1' = global: every F16-src0 mul_mat (stage-1 scalar-path error sources:
    //         fused QKV proj, 1x1-conv trunk, b' scores/sv all fall to f16acc
    //         under PREC_DEFAULT+fp16, GV:7729-7743).
    //   '2' = C-N34 R1: only the fa-feed qkv projection GEMMs (needs KATAGO_VK_FA;
    //         empty set otherwise = no-op).
    //   '3' = C-N34 R2: R1 + the fa out-projection conv GEMM.
    // F32xF16 nodes from matmul() are unconditionally f32acc (GV:7723/4886) and
    // correctly left untouched; future Q8_0 nodes have non-F16 src0 and skip.
    // --- Precision class (F1): ONE rule, ONE implementation, ONE report -------------------------
    // Priority, highest first:
    //   '1'  (env, a FULL class)   -> f32acc on every F16-src0 mul_mat          source=env
    //   the  REQUESTED precision   -> f32acc on every F16-src0 mul_mat          source=requested
    //   '2'/'3' (env, a SUBSET)    -> f32acc on the fa-feed GEMMs only          source=env(subset)
    //   nothing names a class      -> ggml-vulkan's own choice (f16acc on fp16 devices)
    //
    // Why the request is NOT the env's else-branch (review finding, 2026-09-17): with a subset env
    // ('2'/'3') or an unrecognized value ('0'), a request of "no fp16" was skipped entirely, so the
    // "fp32" evaluator testgpuerror builds stayed f16acc/mixed -- F1's disease, revived, with the
    // margin gate still green because it only checks that a reference file was passed.  A request for
    // the full class now wins over any subset, and the source string says so.
    //
    // `mode=` in the report is the mode AFTER Setup's `disableFP16` rewrite (setup.cpp:342), i.e. the
    // value this backend actually receives; setup.cpp's own "useFP16 X" line prints the request
    // BEFORE that rewrite, so the two lines can legitimately differ (they say different things).
    const char* mmpEnv = getenv("KATAGO_VK_MMPREC_F32");
    const char mmp = (mmpEnv != nullptr && mmpEnv[0] != '\0') ? mmpEnv[0] : '\0';
    const bool reqF32 = (ctx->useFP16Mode == enabled_t::False);
    bool precF32Acc = false;
    bool precMixed = false;
    const char* precSource = "capability";
    if(mmp == '1') {
      forceF32AccOnF16MulMats(graph, true, &builder.faFeedGemmPtrs);
      precF32Acc = true;
      precSource = "env";
    }
    else if(reqF32) {
      forceF32AccOnF16MulMats(graph, true, &builder.faFeedGemmPtrs);
      precF32Acc = true;
      precSource = (mmp == '2' || mmp == '3') ? "requested(subsumes env subset)"
                 : (mmpEnv != nullptr ? "requested(env named no class)" : "requested");
    }
    else if(mmp == '2' || mmp == '3') {
      forceF32AccOnF16MulMats(graph, false, &builder.faFeedGemmPtrs);
      precMixed = true;
      precSource = "env(subset)";
    }
    else if(mmpEnv != nullptr) {
      // An unrecognized value is NOT silently reinterpreted as "on", but it is also no longer allowed
      // to hide the request above: it only means "the env named no class".
      precSource = "env(no class named)";
    }
    const char* precClass = precF32Acc ? "f32acc"
                          : (precMixed ? "mixed(env subset)"
                                       : "ggml-default(f16acc on fp16 devices)");

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
                    Global::uint64ToString((uint64_t)ggml_graph_n_nodes(graph)) + " ggml nodes) [prec] accumulation=" +
                    string(precClass) + " source=" + string(precSource) + " mode=" + ctx->useFP16Mode.toString());
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
  // S1-③: report the requested mode truthfully (was hard-coded false).
  return handle->ctx->useFP16Mode != enabled_t::False;
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

  // N-1: per-eval result staging buffers, allocated once here and reused by
  // every getOutput call (sizes are fixed by maxBatchSize under FixedShape
  // policy, so resize() below never reallocates after the first eval).
  vector<float> policyPassResultData;
  vector<float> policyResultData;
  vector<float> valueResultData;
  vector<float> scoreValueResultData;
  vector<float> ownershipResultData;

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

  // N-1: reuse the per-eval staging buffers owned by InputBuffers instead of
  // allocating five vectors every call. resize() to the same fixed size is a
  // no-op on the heap (capacity retained), so steady-state cost is zero.
  vector<float>& policyPassData = inputBuffers->policyPassResultData;
  vector<float>& policyData = inputBuffers->policyResultData;
  vector<float>& valueData = inputBuffers->valueResultData;
  vector<float>& scoreValueData = inputBuffers->scoreValueResultData;
  vector<float>& ownershipData = inputBuffers->ownershipResultData;
  policyPassData.resize(inputBuffers->singlePolicyPassResultElts * runBatchSize);
  policyData.resize(inputBuffers->singlePolicyResultElts * runBatchSize);
  valueData.resize(inputBuffers->singleValueResultElts * runBatchSize);
  scoreValueData.resize(inputBuffers->singleScoreValueResultElts * runBatchSize);
  ownershipData.resize(inputBuffers->singleOwnershipResultElts * runBatchSize);

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

