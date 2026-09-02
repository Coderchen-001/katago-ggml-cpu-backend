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
#include "../external/ggml/include/ggml-vulkan.h"
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
    ggml_backend_t backend = ggml_backend_vk_init(0);
    if(backend == nullptr) {
      cerr << "benchgemm: ggml_backend_vk_init(0) failed" << endl;
      return 1;
    }
    const char* devName = ggml_backend_name(backend);
    if(a.verbose)
      cerr << "benchgemm: backend=" << (devName ? devName : "?") << endl;

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
    size_t ctxSize = ggml_tensor_overhead() * (size_t)(3 * a.N + 8) + ggml_graph_overhead_custom((size_t)(a.N + 16), false) + 1024;
    struct ggml_init_params iparams = {
      /*.mem_size   =*/ ctxSize,
      /*.mem_buffer =*/ nullptr,
      /*.no_alloc   =*/ true,
    };
    ggml_context* ctx = ggml_init(iparams);
    if(ctx == nullptr) { cerr << "benchgemm: ggml_init failed" << endl; return 1; }

    // B shared across nodes (K x N)
    ggml_tensor* B = ggml_new_tensor_2d(ctx, bType, K, Ndim);
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

      vector<uint8_t> bData((size_t)(K * Ndim) * (bType == GGML_TYPE_F16 ? 2 : 4));
      if(bType == GGML_TYPE_F16) {
        uint16_t* p = (uint16_t*)bData.data();
        for(size_t i = 0; i < K * Ndim; i++) p[i] = ggml_fp32_to_fp16(rnd());
      } else {
        float* p = (float*)bData.data();
        for(size_t i = 0; i < K * Ndim; i++) p[i] = rnd();
      }
      ggml_backend_tensor_set(B, bData.data(), 0, bData.size());
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
