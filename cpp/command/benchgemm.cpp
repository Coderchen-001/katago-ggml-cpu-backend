// ============================================================================
// benchgemm.cpp -- Production-regime per-shape GEMM microbenchmark
// (development tool, NOT part of GTP/analysis/selfplay paths).
//
// Purpose (O-hw-17, 2026-09-02; Q1b extension O-hw-25):
//   The PERF_LOGGER op-table "ruler" is regime-distorted (serial + cold L2 +
//   per-op sync, GV:16040-16043) vs production (batched submits, back-to-back
//   kernels, hot L2). The binary experiment proved the sign flips (1660: E
//   "fast" in op-table, E slower ~34% in production). This tool measures
//   per-shape GEMM cost in the PRODUCTION regime: N identical mul_mat nodes
//   in one graph -> batched submit -> fence + wall clock -> /N.
//
//   Q1b (q80-design.md §6, O-q80-4 baseline): -dtype q8f32 measures the
//   int8-dot (mmq) path vs the PRODUCTION baseline matmul_f16_f32 (f32acc).
//   A = Q8_0 weights (quantized on host via ggml_quantize_chunk), B = F32
//   activations -> quantize_y -> matmul_q8_0_q8_1 (mmq) when the device has
//   integer_dot_product and is NOT coopmat2 (mul_mmq.comp only compiles for
//   scalar, vulkan-shaders-gen.cpp:626). Do NOT baseline against matmul_f32:
//   scalar-f32 is slower than f16_f32 and would systematically overstate the
//   dp4a ratio (false positive). Q1b shape set = trunk 1x1 conv shapes only
//   (FA deletes the scores/sv attention buckets, so attention shapes are not
//   FA-invariant); K must be %32==0 (Q8_0 block constraint, q80 K4).
//
// Design (v1, O-hw-17):
//   D1  N identical mul_mat nodes, INDEPENDENT A (weight) + D (output) per
//       node, SHARED B (activation). No WAW -> no inserted sync -> all in one
//       submit batch (N <= max_nodes_per_submit=100).
//   D3  warmup W + timed T iterations: graph_compute -> synchronize -> wall
//       clock; per_gemm = median(total)/N. Wall clock + fence (NOT timestamp
//       query: query goes through the logger regime we proved distorted).
//   D4  Bit-equality probe: compare GPU output vs CPU reference (sampled +
//       NaN/Inf scan). err==0 => bit-equal => no margin gate needed.
//       (f16f32/f16f16 only; q8f32 is non-bit-equal by construction.)
//   D5  dtype pairs: f16xf32 (trunk/heads), f16xf16 (attention), q8f32 (Q1b).
//   G3  ALL candidates incl. reference config m must go through
//       KATAGO_VK_MM_TILE=custom (uniform injection, no baseline asymmetry).
//
// Build notes:
//   - Uses only ggml public API (ggml.h / ggml-backend.h / ggml-vulkan.h).
//   - Does NOT touch ggml-vulkan.cpp or any .comp (N1-ext avoidance).
//   - One config per process (tile env read at pipeline creation); the outer
//     script loops over configs.
//
// Usage:
//   katago benchgemm -M 512 -K 192 -batch 3 -n 50 -iterations 7 -warmup 2
//                    [-dtype f16f32|f16f16|q8f32] [-seed 0] [-cpu-ref]
// ============================================================================

#include "../core/global.h"
#include "../core/rand.h"
#include "../main.h"

#include "../external/ggml/include/ggml.h"
#include "../external/ggml/include/ggml-backend.h"
#ifdef USE_VULKAN_BACKEND
#include "../external/ggml/include/ggml-vulkan.h"
#endif
#include "../external/ggml/include/ggml-alloc.h"

#include <cmath>
#include <cstring>
#include <chrono>
#include <vector>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>

// Compile-time git revision (build-generated program/gitinfo.h, same mechanism
// as main.cpp:13). Unlike a runtime `git rev-parse` (whose cwd may be outside
// the repo, e.g. running from a build dir -> "?"), this records the exact
// commit the binary was built from -- the value the CSV "git" column needs.
#ifdef NO_GIT_REVISION
#define BENCHGEMM_GIT_REVISION "<omitted>"
#else
#include <program/gitinfo.h>
#define BENCHGEMM_GIT_REVISION GIT_REVISION
#endif

using namespace std;

namespace {
  // ---------------- tiny arg parser (KataGo-style) ----------------
  struct BenchArgs {
    int64_t M = 512;
    int64_t K = 192;
    int64_t batch = 3;
    int N = 50;            // identical nodes per graph
    int iterations = 7;    // timed iterations (median taken)
    int warmup = 2;        // warmup iterations (discarded)
    int seed = 0;
    int dtype = 0;         // 0 = f16 x f32 (trunk, production); 1 = f16 x f16 (attention); 2 = q8f32 (Q1b: Q8_0 x F32 -> mmq)
    bool cpuRef = false;   // compare vs CPU reference (bit-equal probe; f16f32/f16f16 only)
    bool verbose = false;
    string op = "matmul";  // "matmul" = the original path; otherwise a single-op benchmark (see benchOneOp)
    bool strided = false;  // feed NON-CONTIGUOUS (permuted-view) inputs: the real graph has 100+ such
                           // nodes (CONT/VIEW/PERMUTE/FA/ADD/SUB/CONCAT consumers), which clean 2D
                           // benchmarks cannot reproduce -- and that is the one class left in the
                           // ISA-tier investigation after every contiguous op came out flat.
    int bcast = 0;         // 0 = same shape; 1 = src1 is [rows,1] (per-channel bias, the [C,1,1,1]
                           // pattern); 2 = src1 is [1,tokens] (per-position mask, the [1,S,N]
                           // pattern). The graph has 393 ADD/MUL nodes and most are broadcasts, which
                           // are DIFFERENT kernels from the same-shape elementwise ones.
  };

  bool getInt64(const vector<string>& args, const string& key, int64_t& out) {
    for(size_t i = 0; i + 1 < args.size(); i++) {
      if(args[i] == key) { out = Global::stringToInt64(args[i+1]); return true; }
    }
    return false;
  }
  bool getInt(const vector<string>& args, const string& key, int& out) {
    int64_t v; if(getInt64(args, key, v)) { out = (int)v; return true; } return false;
  }
  bool getBool(const vector<string>& args, const string& key, bool& out) {
    for(size_t i = 0; i < args.size(); i++) {
      if(args[i] == key) { out = true; return true; }
    }
    return false;
  }

  // ---------------- CPU reference (bit-equal probe, D4) ----------------
  // C[m][n] = sum_k A[k][m] * B[k][n]  (ggml mul_mat layout: A is [K,M], B is [K,N])
  void cpuMulMatF32(
    const float* A, const float* B, float* C,
    int64_t K, int64_t M, int64_t N
  ) {
    for(int64_t m = 0; m < M; m++)
      for(int64_t n = 0; n < N; n++) {
        float acc = 0.0f;
        for(int64_t k = 0; k < K; k++)
          acc += A[k + m*K] * B[k + n*K];
        C[m + n*M] = acc;
      }
  }

  // Sample-based comparison: full scan for NaN/Inf, sampled max abs/rel err.
  void compareOutputs(
    const float* gpu, const float* cpu, int64_t count,
    double& maxAbsErr, double& maxRelErr, bool& hasNan
  ) {
    maxAbsErr = 0.0; maxRelErr = 0.0; hasNan = false;
    const int64_t sampleStep = std::max<int64_t>(1, count / 4096);
    for(int64_t i = 0; i < count; i++) {
      float g = gpu[i];
      if(std::isnan(g) || std::isinf(g)) { hasNan = true; continue; }
      if(i % sampleStep != 0)
        continue;
      double absErr = std::fabs((double)g - (double)cpu[i]);
      if(absErr > maxAbsErr) maxAbsErr = absErr;
      double denom = std::max(std::fabs((double)cpu[i]), 1e-12);
      double relErr = absErr / denom;
      if(relErr > maxRelErr) maxRelErr = relErr;
    }
  }

  // ---------------- single-op benchmark (ISA-tier attribution) ----------------
  // Why this mode exists: on a 9800X3D the AVX-512 ggml-cpu tier ("cascadelake", what the loader
  // picks by feature score) is 22-40% SLOWER than the AVX2 tiers on transformer models, while the
  // classic conv net is unaffected. Measured eliminations before writing this:
  //   * not mul_mat        -- this file, all shapes tried, flat across tiers (incl. attention-like
  //                           m=361 k=64 and m=64 k=361),
  //   * not flash-attn     -- KATAGO_VK_FA=0 keeps (grows) the gap,
  //   * not the RMS fusion -- GGML_CPU_DISABLE_FUSION=1 leaves it unchanged,
  //   * not frequency/power-- mean %ProcessorPerformance 109 in BOTH tiers, 8 samples each,
  //   * not softmax/attention-dominated -- the extra time scales LINEARLY with board area
  //                           (19.2 / 40.3 / 89.7 ms at 9x9 / 13x13 / 19x19), not with S^2.
  // What is left is the family of element-wise / reduction / layout kernels, which is what this
  // mode measures one at a time. Shapes come from -M (ne0) and -batch (ne1 = batch*361); -n is the
  // number of identical nodes per graph, and the reported rate is elements/s (not GFLOPS).
  int benchOneOp(ggml_backend_t backend, const char* devName, const BenchArgs& a, const string& op) {
    const int64_t rows   = a.M > 0 ? a.M : 384;
    const int64_t tokens = a.batch > 0 ? a.batch * 361 : 361;
    const int64_t elems  = rows * tokens;

    const bool twoInput = (op == "add" || op == "mul" || op == "swiglu");
    const bool wantRms  = (op == "rmsnorm");
    const bool wantCont = (op == "cont");
    const bool wantMask = (op == "softmaxext");
    const bool inF16    = (op == "cast_f32");
    if(op != "silu" && op != "gelu" && op != "swiglu" && op != "rmsnorm" && op != "softmax" &&
       op != "add" && op != "mul" && op != "cont" && op != "cast_f16" && op != "cast_f32" &&
       op != "softmaxext" && op != "scale") {
      cerr << "benchgemm: unknown -op \"" << op << "\" (matmul|silu|gelu|swiglu|rmsnorm|softmax|softmaxext|cast_f16|cast_f32|scale|add|mul|cont)" << endl;
      return 1;
    }

    // A node count above 16 would just multiply memory traffic here (each node owns its inputs),
    // and the point is per-node cost, not occupancy.
    const int nNodes = (a.N > 0 && a.N <= 16) ? a.N : 8;

    // +2 tensors per node when -strided (a base plus its permuted view), and the mask/rms helpers.
    const size_t ctxSize = ggml_tensor_overhead() * (size_t)(4 * nNodes + 32) +
                           ggml_graph_overhead_custom((size_t)(nNodes + 16), false) + 2048;
    struct ggml_init_params iparams = {
      /*.mem_size   =*/ ctxSize,
      /*.mem_buffer =*/ nullptr,
      /*.no_alloc   =*/ true,
    };
    ggml_context* ctx = ggml_init(iparams);
    if(ctx == nullptr) { cerr << "benchgemm: ggml_init failed" << endl; return 1; }

    vector<ggml_tensor*> X(nNodes), Y(nNodes), Out(nNodes), XBase(nNodes), YBase(nNodes);
    ggml_tensor* W = nullptr;
    ggml_tensor* Mask = nullptr;
    const int64_t yRows   = (a.bcast == 1) ? 1 : rows;
    const int64_t yTokens = (a.bcast == 2) ? 1 : tokens;
    const int64_t yElems  = yRows * yTokens;
    if(a.bcast != 0 && !(op == "add" || op == "mul")) {
      cerr << "benchgemm: -bcast is only meaningful for -op add|mul (swiglu/others do not broadcast)" << endl;
      return 1;
    }
    if(wantRms)
      W = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, rows, 1);
    if(wantMask)
      Mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, rows, tokens);
    for(int i = 0; i < nNodes; i++) {
      if(a.strided) {
        // ROW-CONTIGUOUS with a 3x gap between rows: nb[0] stays sizeof(elem) (so ADD/MUL/unary are
        // legal -- they assert nb00 == sizeof(src0_t) / is_contiguous_rows), but ggml_is_contiguous
        // is false. This is the strided shape the real graph actually feeds: the builder's
        // QKV_NOCONT path hands downstream a window whose row stride is 3*OC*4 while ne0 = OC.
        // (A full transpose is NOT a legal input for these ops -- ggml asserts -- which is why the
        // graph carries 90 CONT nodes; that is a different, already-flat case.)
        XBase[i] = ggml_new_tensor_2d(ctx, inF16 ? GGML_TYPE_F16 : GGML_TYPE_F32, rows * 3, tokens);
        X[i] = ggml_view_2d(ctx, XBase[i], rows, tokens,
                            (size_t)rows * 3 * (inF16 ? sizeof(ggml_fp16_t) : sizeof(float)), 0);
        if(twoInput) {
          if(a.bcast != 0) {
            Y[i] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, yRows, yTokens);
            YBase[i] = Y[i];
          } else {
            YBase[i] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, rows * 3, tokens);
            Y[i] = ggml_view_2d(ctx, YBase[i], rows, tokens, (size_t)rows * 3 * sizeof(float), 0);
          }
        }
      } else {
        X[i] = ggml_new_tensor_2d(ctx, inF16 ? GGML_TYPE_F16 : GGML_TYPE_F32, rows, tokens);
        XBase[i] = X[i];
        if(twoInput) {
          Y[i] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, yRows, yTokens);
          YBase[i] = Y[i];
        }
      }
    }
    ggml_cgraph* graph = ggml_new_graph_custom(ctx, (size_t)(nNodes + 16), false);
    for(int i = 0; i < nNodes; i++) {
      if(op == "silu")          Out[i] = ggml_silu(ctx, X[i]);
      else if(op == "gelu")     Out[i] = ggml_gelu(ctx, X[i]);
      else if(op == "swiglu")   Out[i] = ggml_swiglu_split(ctx, X[i], Y[i]);
      else if(op == "rmsnorm")  Out[i] = ggml_mul(ctx, ggml_rms_norm(ctx, X[i], 1e-5f), W);
      else if(op == "softmax")  Out[i] = ggml_soft_max(ctx, X[i]);
      else if(op == "softmaxext") Out[i] = ggml_soft_max_ext(ctx, X[i], Mask, 1.0f, 0.0f);
      else if(op == "cast_f16") Out[i] = ggml_cast(ctx, X[i], GGML_TYPE_F16);
      else if(op == "cast_f32") Out[i] = ggml_cast(ctx, X[i], GGML_TYPE_F32);
      else if(op == "scale")    Out[i] = ggml_scale(ctx, X[i], 0.5f);
      else if(op == "add")      Out[i] = ggml_add(ctx, X[i], Y[i]);
      else if(op == "mul")      Out[i] = ggml_mul(ctx, X[i], Y[i]);
      else if(op == "cont")     Out[i] = ggml_cont(ctx, ggml_transpose(ctx, X[i]));
      ggml_build_forward_expand(graph, Out[i]);
    }

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if(buf == nullptr) { cerr << "benchgemm: alloc failed" << endl; return 1; }

    {
      // Fill the WHOLE base, not just the viewed window: uninitialized floats can be denormals/NaN
      // and a denormal penalty would silently pollute the timing.
      vector<float> data((size_t)(a.strided ? 3 * elems : elems));
      uint64_t rngState = 99991 + (uint64_t)a.seed * 1000003;
      for(int i = 0; i < nNodes; i++) {
        for(size_t k = 0; k < data.size(); k++) {
          rngState ^= rngState << 13; rngState ^= rngState >> 7; rngState ^= rngState << 17;
          data[k] = ((float)(rngState & 0xFFFF) / 32767.5f) - 1.0f;
        }
        ggml_backend_tensor_set(XBase[i], data.data(), 0, data.size() * sizeof(float));
        if(twoInput)
          ggml_backend_tensor_set(YBase[i], data.data(), 0, (size_t)yElems * sizeof(float));
      }
      if(inF16) {
        vector<ggml_fp16_t> h((size_t)elems);
        for(size_t k = 0; k < h.size(); k++)
          h[k] = ggml_fp32_to_fp16(data[k]);
        for(int i = 0; i < nNodes; i++)
          ggml_backend_tensor_set(XBase[i], h.data(), 0, h.size() * sizeof(ggml_fp16_t));
      }
      if(wantRms) {
        vector<float> w((size_t)rows, 1.0f);
        ggml_backend_tensor_set(W, w.data(), 0, w.size() * sizeof(float));
      }
      if(wantMask) {
        vector<float> m((size_t)elems, 0.0f);
        ggml_backend_tensor_set(Mask, m.data(), 0, m.size() * sizeof(float));
      }
    }

    auto nowMs = []() { return std::chrono::duration_cast<std::chrono::microseconds>(
      std::chrono::high_resolution_clock::now().time_since_epoch()).count() / 1000.0; };
    for(int w = 0; w < a.warmup; w++) {
      ggml_backend_graph_compute(backend, graph);
      ggml_backend_synchronize(backend);
    }
    vector<double> totalsMs;
    for(int t = 0; t < a.iterations; t++) {
      const double t0 = nowMs();
      ggml_backend_graph_compute(backend, graph);
      ggml_backend_synchronize(backend);
      totalsMs.push_back(nowMs() - t0);
    }
    sort(totalsMs.begin(), totalsMs.end());
    const double medTotal = totalsMs[totalsMs.size() / 2];
    const double perNodeUs = medTotal * 1000.0 / (double)nNodes;
    const double spread = (totalsMs.back() - totalsMs.front()) / medTotal * 100.0;
    const double melemPerSec = ((double)elems / (perNodeUs / 1e6)) / 1e6;
    const string gitHead = BENCHGEMM_GIT_REVISION;

    cout << (devName ? devName : "?") << ","
         << "op=" << op << ",rows=" << rows << ",tokens=" << tokens << ",batch=" << a.batch << ","
         << op << ",(opbench" << (a.strided ? ";rowstrided" : "") << ";bcast=" << a.bcast << "),"
         << fixed << setprecision(3) << perNodeUs << ","
         << setprecision(2) << melemPerSec << ","
         << setprecision(2) << spread << ",-,-,-,"
         << nNodes << "," << a.iterations << "," << a.seed << ","
         << gitHead << endl;
    if(a.verbose)
      cerr << "benchgemm: op=" << op << " per_node=" << perNodeUs << " us  " << melemPerSec
           << " Melem/s  spread=" << spread << "%" << endl;

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return 0;
  }

  // ---------------- main bench ----------------
  int benchGemmMain(const vector<string>& args) {
    BenchArgs a;
    getInt64(args, "-M", a.M);
    getInt64(args, "-K", a.K);
    getInt64(args, "-batch", a.batch);
    getInt(args, "-n", a.N);
    getInt(args, "-iterations", a.iterations);
    getInt(args, "-warmup", a.warmup);
    getInt(args, "-seed", a.seed);
    getBool(args, "-cpu-ref", a.cpuRef);
    getBool(args, "-strided", a.strided);
    getInt(args, "-bcast", a.bcast);
    if(a.strided && a.cpuRef) {
      cerr << "benchgemm: -cpu-ref ignored with -strided (non-contiguous readback is not supported)" << endl;
      a.cpuRef = false;
    }
    getBool(args, "-verbose", a.verbose);
    for(size_t i = 0; i < args.size(); i++) {
      if(args[i] == "-dtype" && i + 1 < args.size()) {
        const string& d = args[i+1];
        if(d == "f16f16") a.dtype = 1;
        else if(d == "q8f32") a.dtype = 2;
        else a.dtype = 0;   // f16f32
      }
    }
    if(a.dtype == 2 && a.cpuRef) {
      // Q8_0 path is non-bit-equal by construction; CPU reference meaningless.
      cerr << "benchgemm: warning: -cpu-ref ignored for q8f32 (quantized path is not bit-equal)" << endl;
      a.cpuRef = false;
    }

    const int64_t M = a.M;
    const int64_t K = a.K;
    const int64_t Ndim = a.batch * 361;   // KataGo N = batch * board cells

    // ---- init backend ----
    // Backend-agnostic since the CPU backend exists: on the CPU the same mul_mat shapes run
    // through ggml-cpu, which is what makes this command usable for the f16-vs-q8 kernel
    // question without a GPU. ggml_backend_load_all() is a no-op in a static build and loads
    // the highest-scoring ISA variant dll in a GGML_CPU_ALL_VARIANTS build.
#if defined(USE_CPUGGML_BACKEND)
    ggml_backend_load_all();
    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, NULL);
    if(backend == nullptr) {
      cerr << "benchgemm: no ggml CPU backend available" << endl;
      return 1;
    }
#else
    ggml_backend_t backend = ggml_backend_vk_init(0);
    if(backend == nullptr) {
      cerr << "benchgemm: ggml_backend_vk_init(0) failed" << endl;
      return 1;
    }
#endif
    const char* devName = ggml_backend_name(backend);
    if(a.verbose)
      cerr << "benchgemm: backend=" << (devName ? devName : "?") << endl;

    for(size_t i = 0; i + 1 < args.size(); i++) {
      if(args[i] == "-op")
        a.op = args[i+1];
    }
    if(a.op != "matmul") {
      const int rc = benchOneOp(backend, devName, a, a.op);
      ggml_backend_free(backend);
      return rc;
    }

    // ---- context + N identical mul_mat nodes ----
    // A = weights; B = activations. dtype 2 (q8f32, Q1b): A=Q8_0 -> quantize_y
    // -> matmul_q8_0_q8_1 (mmq) on integer_dot && !coopmat2 devices (vulkan-
    // shaders-gen.cpp:626); B stays F32 (quantize_y requires src1==F32).
    const bool isQ8 = (a.dtype == 2);
    const char* dtypeLabel = isQ8 ? "q8f32" : (a.dtype == 1 ? "f16f16" : "f16f32");
    ggml_type aType = isQ8 ? GGML_TYPE_Q8_0 : GGML_TYPE_F16;
    ggml_type bType = (a.dtype == 1) ? GGML_TYPE_F16 : GGML_TYPE_F32;
    if(isQ8 && K % 32 != 0) {
      cerr << "benchgemm: q8f32 requires K % 32 == 0 (Q8_0 block size, q80-design K4); got K=" << K << endl;
      return 1;
    }
    // A: [K, M] ; B: [K, N] ; D: [M, N]  (ggml mul_mat layout)
    size_t ctxSize = ggml_tensor_overhead() * (size_t)(3 * a.N + 16) + ggml_graph_overhead_custom((size_t)(a.N + 16), false) + 2048;
    struct ggml_init_params iparams = {
      /*.mem_size   =*/ ctxSize,
      /*.mem_buffer =*/ nullptr,
      /*.no_alloc   =*/ true,
    };
    ggml_context* ctx = ggml_init(iparams);
    if(ctx == nullptr) { cerr << "benchgemm: ggml_init failed" << endl; return 1; }

    // B shared across nodes (K x N); with -strided it is a permuted view, i.e. a non-contiguous
    // src1, which is what the real graph feeds mul_mat in several places.
    ggml_tensor* BBase = nullptr;
    ggml_tensor* B = nullptr;
    if(a.strided) {
      // Row-strided (3x gap), i.e. what the QKV_NOCONT window looks like: nb[0] == elem size, nb[1]
      // three times wider than ne0 needs.
      BBase = ggml_new_tensor_2d(ctx, bType, K * 3, Ndim);
      B = ggml_view_2d(ctx, BBase, K, Ndim, (size_t)K * 3 * (bType == GGML_TYPE_F16 ? 2 : 4), 0);
    } else {
      B = ggml_new_tensor_2d(ctx, bType, K, Ndim);
      BBase = B;
    }
    // A_i per node; R_i = mul_mat dst returned by ggml_mul_mat (must be kept:
    // the bit-equal probe reads R[0], and R[] tensors are what the graph writes)
    vector<ggml_tensor*> A(a.N), R(a.N);
    for(int i = 0; i < a.N; i++)
      A[i] = ggml_new_tensor_2d(ctx, aType, K, M);

    ggml_cgraph* graph = ggml_new_graph_custom(ctx, (size_t)(a.N + 16), false);
    for(int i = 0; i < a.N; i++) {
      R[i] = ggml_mul_mat(ctx, A[i], B);
      ggml_build_forward_expand(graph, R[i]);
    }

    // ---- allocate ----
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if(buf == nullptr) { cerr << "benchgemm: alloc failed" << endl; return 1; }

    // ---- fill data ----
    // A: [K,M] f16 or Q8_0; B: [K,N] f16/f32; deterministic from seed
    uint64_t rngState = 12345 + (uint64_t)a.seed * 1000003;
    auto rnd = [&]() -> float {
      // xorshift-ish deterministic noise in [-1,1]
      rngState ^= rngState << 13; rngState ^= rngState >> 7; rngState ^= rngState << 17;
      return ((float)(rngState & 0xFFFF) / 32767.5f) - 1.0f;
    };
    {
      if(isQ8) {
        // Host-quantize f32 weights to Q8_0 via the public ggml API.
        // ggml_quantize_chunk(type, src, dst, start_row, nrows, n_per_row, imatrix);
        vector<float> aF32((size_t)(K * M));
        for(size_t i = 0; i < aF32.size(); i++) aF32[i] = rnd();
        vector<uint8_t> qData(ggml_nbytes(A[0]));
        const size_t written = ggml_quantize_chunk(GGML_TYPE_Q8_0, aF32.data(), qData.data(),
                                                   0, M, K, nullptr);
        if(written != qData.size()) {
          cerr << "benchgemm: quantize_chunk wrote " << written << " bytes, expected " << qData.size() << endl;
          return 1;
        }
        ggml_backend_tensor_set(A[0], qData.data(), 0, qData.size());
      } else {
        vector<uint16_t> aData((size_t)(K * M));
        for(size_t i = 0; i < aData.size(); i++)
          aData[i] = ggml_fp32_to_fp16(rnd());
        ggml_backend_tensor_set(A[0], aData.data(), 0, aData.size() * 2);
      }
      // all A_i share the same weight data (independent buffers, same content)
      for(int i = 1; i < a.N; i++)
        ggml_backend_tensor_copy(A[0], A[i]);

      const size_t bElems = (size_t)(a.strided ? 3 : 1) * (size_t)K * (size_t)Ndim;
      vector<uint8_t> bData(bElems * (bType == GGML_TYPE_F16 ? 2 : 4));
      if(bType == GGML_TYPE_F16) {
        uint16_t* p = (uint16_t*)bData.data();
        for(size_t i = 0; i < bElems; i++) p[i] = ggml_fp32_to_fp16(rnd());
      } else {
        float* p = (float*)bData.data();
        for(size_t i = 0; i < bElems; i++) p[i] = rnd();
      }
      ggml_backend_tensor_set(BBase, bData.data(), 0, bData.size());
    }

    // ---- CPU reference (D4; f16f32/f16f16 only - q8f32 forced cpuRef=false above) ----
    // A[0] is [K,M] f16 -> convert to f32; B f16/f32 -> f32
    vector<float> aF32, bF32;
    if(a.cpuRef) {
      aF32.resize((size_t)(K * M)); bF32.resize((size_t)(K * Ndim));
      vector<uint16_t> aRaw((size_t)(K * M));
      ggml_backend_tensor_get(A[0], aRaw.data(), 0, aRaw.size() * 2);
      for(size_t i = 0; i < aRaw.size(); i++) aF32[i] = ggml_fp16_to_fp32(aRaw[i]);
      if(bType == GGML_TYPE_F16) {
        vector<uint16_t> bRaw((size_t)(K * Ndim));
        ggml_backend_tensor_get(B, bRaw.data(), 0, bRaw.size() * 2);
        for(size_t i = 0; i < bRaw.size(); i++) bF32[i] = ggml_fp16_to_fp32(bRaw[i]);
      } else {
        ggml_backend_tensor_get(B, bF32.data(), 0, bF32.size() * 4);
      }
    }

    // ---- warmup + timed iterations ----
    auto nowMs = []() { return std::chrono::duration_cast<std::chrono::microseconds>(
      std::chrono::high_resolution_clock::now().time_since_epoch()).count() / 1000.0; };

    for(int w = 0; w < a.warmup; w++) {
      ggml_backend_graph_compute(backend, graph);
      ggml_backend_synchronize(backend);
    }

    vector<double> totalsMs;
    for(int t = 0; t < a.iterations; t++) {
      double t0 = nowMs();
      ggml_backend_graph_compute(backend, graph);
      ggml_backend_synchronize(backend);
      double t1 = nowMs();
      totalsMs.push_back(t1 - t0);
    }
    sort(totalsMs.begin(), totalsMs.end());
    double medTotal = totalsMs[totalsMs.size() / 2];
    double perGemmUs = medTotal * 1000.0 / (double)a.N;   // per node
    // spread: (max-min)/median
    double spread = (totalsMs.back() - totalsMs.front()) / medTotal * 100.0;
    double gflops = (2.0 * M * K * Ndim) / (perGemmUs / 1e6) / 1e9;

    // ---- bit-equal probe (D4): check R[0] (the real mul_mat dst) vs CPU ----
    double maxAbsErr = -1, maxRelErr = -1; bool hasNan = false; bool bitEqual = false;
    if(a.cpuRef) {
      vector<float> dGpu((size_t)(M * Ndim));
      ggml_backend_tensor_get(R[0], dGpu.data(), 0, dGpu.size() * 4);
      vector<float> dCpu((size_t)(M * Ndim));
      cpuMulMatF32(aF32.data(), bF32.data(), dCpu.data(), K, M, Ndim);
      compareOutputs(dGpu.data(), dCpu.data(), (int64_t)dGpu.size(), maxAbsErr, maxRelErr, hasNan);
      bitEqual = (maxAbsErr == 0.0 && !hasNan);
    }

    // ---- output CSV row ----
    string tileEnv = getenv("KATAGO_VK_MM_TILE") != nullptr ? getenv("KATAGO_VK_MM_TILE") : "(default)";
    string gitHead = BENCHGEMM_GIT_REVISION;

    cout << (devName ? devName : "?") << ","
         << "M=" << M << ",K=" << K << ",N=" << Ndim << ",batch=" << a.batch << ","
         << dtypeLabel << ","
         << tileEnv << ","
         << fixed << setprecision(2) << perGemmUs << ","
         << setprecision(1) << gflops << ","
         << setprecision(2) << spread << ","
         << (a.cpuRef ? (bitEqual ? "1" : "0") : "-") << ","
         << (a.cpuRef ? to_string(maxAbsErr) : "-") << ","
         << (a.cpuRef ? (hasNan ? "1" : "0") : "-") << ","
         << a.N << "," << a.iterations << "," << a.seed << ","
         << gitHead << endl;

    if(a.verbose) {
      if(isQ8)
        cerr << "benchgemm: q8f32 -> mmq (matmul_q8_0_q8_1) iff integer_dot_product && !coopmat2 (scalar, vulkan-shaders-gen.cpp:626); else dequant fallback (Mode B)" << endl;
      cerr << "benchgemm: per_gemm=" << perGemmUs << " us  gflops=" << gflops
           << "  spread=" << spread << "%  (N=" << a.N << ")" << endl;
      if(a.cpuRef)
        cerr << "benchgemm: bit_equal=" << bitEqual << " maxAbsErr=" << maxAbsErr
             << " maxRelErr=" << maxRelErr << " hasNan=" << hasNan << endl;
    }

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return 0;
  }
}

int MainCmds::benchgemm(const vector<string>& args) {
  return benchGemmMain(args);
}
