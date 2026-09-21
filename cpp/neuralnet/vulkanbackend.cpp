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
#include "onnxtoggml.h"
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

