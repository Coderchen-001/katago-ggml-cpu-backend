// CPU backend for KataGo (ggml-cpu).
//
// Derived from vulkanbackend.cpp: the same ONNX -> ggml interpreter (onnxtoggml.h,
// zero backend coupling), the same fixed-shape graph, the same IO tensor protocol,
// and the same host-side post-processing -- but the kernels are ggml-cpu's. The
// model is a .onnx file dumped by 'katago dumponnx' (or a .bin.gz, converted to the
// same ONNX graph in memory at load time). Batch policy is FixedShape, so short
// batches are padded up to maxBatchSize (see onnxbackend.cpp).
//
// The IO tensor protocol is identical to the ONNX/TensorRT/Vulkan path: four NCHW
// float32 inputs declared in the order InputSpatial, InputGlobal, InputMeta,
// InputMask, and five NCHW float32 outputs OutputPolicyPass / OutputPolicy /
// OutputValue / OutputScoreValue / OutputOwnership, all raw logits.
// getOutput below reproduces the ONNX backend's post-processing exactly
// (per-row optimism blend, inverse-symmetry, version-branched score-value decode).
//
// Distribution shape: the kernels come from ONE ISA variant dll (ggml-cpu-<tag>.dll)
// picked at startup by ggml_backend_load_all() -- see globalInitialize -- so a single
// release covers every x86-64 CPU from SSE4.2 up to AVX-512 VNNI. Nothing in this file
// knows which variant was chosen except the log line printDevices writes.

#ifdef USE_CPUGGML_BACKEND

#include "../neuralnet/nninterface.h"
#include "../neuralnet/nneval.h"
#include "../neuralnet/nninputs.h"
#include "../neuralnet/modelversion.h"
#include "../neuralnet/onnxmodelbuilder.h"
#include "onnxtoggml.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"

#include "onnx.pb.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <intrin.h>   // __cpuidex
#elif defined(__linux__)
#include <pthread.h>  // pthread_{get,set}affinity_np (E-core probe)
#include <sched.h>
#include <unistd.h>
#if defined(__x86_64__) || defined(__i386__)
#include <cpuid.h>    // __get_cpuid_count
#endif
#endif

using namespace std;

//--------------------------------------------------------------


//--------------------------------------------------------------

// CPU topology. Two different numbers with two different uses:
//   physicalCores : every core, E-cores included -> budget for the (N,T) B form (N handles x 1)
//   mathCores     : E-cores excluded             -> width of ONE pool (A/C form), because every
//                                                   ggml-cpu op ends in a barrier across the pool
//
// Not invented here where upstream already solved it. The Linux column follows llama.cpp's
// common_cpu_get_num_physical_cores() (count DISTINCT /sys/.../topology/thread_siblings, which
// needs no core_id/package bookkeeping and tolerates sparse CPU numbering) and
// common_cpu_get_num_math() (pin each CPU; cpuid leaf 0x1a reporting an Atom core means E-core,
// so skip it and its HT sibling -- "efficiency cores harm lockstep threading" is exactly this
// backend's per-op barrier failure mode). Windows has no upstream equivalent for the E-core half
// (that gate is __x86_64__ && __linux__), so it reads the OS's own field instead of guessing:
// PROCESSOR_RELATIONSHIP.EfficiencyClass.
struct CpuTopology {
  int physicalCores;
  int mathCores;
  int logicalCores;
  string physicalSource;
  string mathSource;
};

// One entry per PHYSICAL core: the logical CPU to pin work to (the lowest of its SMT siblings) and
// its sibling list, for the fingerprint. Enumerating this is what keeps the auto-generated CPU
// mask honest: on Windows logical processors are numbered with the SMT siblings ADJACENT (0 and 1
// are the two threads of core 0 on this machine), so "the lowest T logical CPUs" would put a whole
// pool onto T/2 physical cores. Measured on the 9800X3D: T=2 pinned to {0,1} costs 466 ms/batch
// against 380 ms unpinned, and two 1-thread handles pinned to 0 and 1 cost ~890 ms/eval each
// against 359 ms solo -- i.e. the naive mask halves throughput.
struct CpuCoreEntry {
  int primaryCpu;
  int efficiencyClass;   // lowest class = the capable cores; 0 on systems that do not report it
  string siblings;       // display only
};

// "0-1", "0,4", "0-3,8": lowest CPU in the list, or -1.
static int parseCpuListMin(const string& line) {
  int minCpu = -1;
  size_t i = 0;
  while(i < line.size()) {
    while(i < line.size() && (line[i] == ',' || isspace((unsigned char)line[i])))
      i++;
    if(i >= line.size())
      break;
    size_t j = i;
    while(j < line.size() && isdigit((unsigned char)line[j]))
      j++;
    if(j == i)
      break;
    const int a = atoi(line.substr(i, j - i).c_str());
    if(minCpu < 0 || a < minCpu)
      minCpu = a;
    i = j;
    if(i < line.size() && line[i] == '-') {
      size_t k = i + 1;
      while(k < line.size() && isdigit((unsigned char)line[k]))
        k++;
      i = k;
    }
  }
  return minCpu;
}

static vector<CpuCoreEntry> detectPhysicalCores() {
  vector<CpuCoreEntry> cores;
#if defined(_WIN32)
  DWORD len = 0;
  const bool ok = GetLogicalProcessorInformationEx(RelationProcessorCore, NULL, &len) != 0 ||
                  GetLastError() == ERROR_INSUFFICIENT_BUFFER;
  if(!ok)
    return cores;
  vector<char> buf(len);
  if(!GetLogicalProcessorInformationEx(
       RelationProcessorCore,
       reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buf.data()), &len))
    return cores;
  for(DWORD off = 0; off < len; ) {
    PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX info =
      reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buf.data() + off);
    if(info->Relationship == RelationProcessorCore) {
      CpuCoreEntry e;
      e.efficiencyClass = (int)info->Processor.EfficiencyClass;
      e.primaryCpu = -1;
      for(WORD g = 0; g < info->Processor.GroupCount; g++) {
        const KAFFINITY m = info->Processor.GroupMask[g].Mask;
        const int base = (int)info->Processor.GroupMask[g].Group * 64;
        for(int b = 0; b < 64; b++) {
          if(((m >> b) & 1ull) == 0)
            continue;
          const int cpu = base + b;
          if(e.primaryCpu < 0 || cpu < e.primaryCpu)
            e.primaryCpu = cpu;
          if(!e.siblings.empty())
            e.siblings += "+";
          e.siblings += Global::intToString(cpu);
        }
      }
      if(e.primaryCpu >= 0 && e.primaryCpu < GGML_MAX_N_THREADS)
        cores.push_back(e);
    }
    off += info->Size;
  }
#elif defined(__linux__)
  // Identical sibling lists = one physical core, and the numbering need not be contiguous.
  unordered_set<string> seen;
  for(unsigned cpu = 0; cpu < 4096; cpu++) {
    ifstream in("/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/topology/thread_siblings");
    if(!in.is_open())
      break;
    string line;
    if(!std::getline(in, line))
      continue;
    if(!seen.insert(line).second)
      continue;
    CpuCoreEntry e;
    e.primaryCpu = parseCpuListMin(line);
    e.efficiencyClass = 0;
    e.siblings = line;
    if(e.primaryCpu >= 0 && e.primaryCpu < GGML_MAX_N_THREADS)
      cores.push_back(e);
  }
#endif
  return cores;
}

// Pin one pool to one CPU per physical core, preferring the capable (non-E) cores. Used only when
// affinity was explicitly requested (KATAGO_CPU_AFFINITY=1) or a mask was given.
static void buildAutoMask(bool (&mask)[GGML_MAX_N_THREADS], int maxCores) {
  memset(mask, 0, sizeof(mask));
  vector<CpuCoreEntry> cores = detectPhysicalCores();
  std::stable_sort(cores.begin(), cores.end(), [](const CpuCoreEntry& a, const CpuCoreEntry& b) {
    if(a.efficiencyClass != b.efficiencyClass)
      return a.efficiencyClass < b.efficiencyClass;
    return a.primaryCpu < b.primaryCpu;
  });
  int taken = 0;
  for(size_t i = 0; i < cores.size() && taken < maxCores; i++) {
    mask[cores[i].primaryCpu] = true;
    taken++;
  }
  if(taken == 0) {
    for(int i = 0; i < maxCores && i < GGML_MAX_N_THREADS; i++)
      mask[i] = true;
  }
}

static void cpuidEx(int leaf, int subleaf, unsigned out[4]) {
#if defined(_WIN32)
  __cpuidex((int*)out, leaf, subleaf);
#elif (defined(__x86_64__) || defined(__i386__)) && !defined(_MSC_VER)
  unsigned eax = 0, ebx = 0, ecx = 0, edx = 0;
  __get_cpuid_count((unsigned)leaf, (unsigned)subleaf, &eax, &ebx, &ecx, &edx);
  out[0] = eax; out[1] = ebx; out[2] = ecx; out[3] = edx;
#else
  out[0] = out[1] = out[2] = out[3] = 0;
#endif
}

// Hardware feature bits, for the FINGERPRINT ONLY. Never gate behaviour on these: the only sound
// gate is the loaded variant's compile-time feature list (ggml_backend_get_features), because
// that is what the kernels were actually built with. See the design doc §3.12-B.
struct CpuidFeatures {
  bool avx2, avx512f, avx512dq, avx512cd, avx512bw, avx512vl;
  bool avx512Vnni, avx512Vbmi, avx512Bf16, avxVnni;
  bool amxTile, amxInt8;
};

static CpuidFeatures detectCpuidFeatures() {
  CpuidFeatures f = {};
  unsigned r[4] = {0, 0, 0, 0};
  cpuidEx(0, 0, r);
  if(r[0] < 7)
    return f;
  cpuidEx(7, 0, r);
  const unsigned maxSubleaf7 = r[0];
  f.avx2       = (r[1] >> 5)  != 0;   // leaf 7.0 EBX
  f.avx512f    = (r[1] >> 16) != 0;
  f.avx512dq   = (r[1] >> 17) != 0;
  f.avx512cd   = (r[1] >> 28) != 0;
  f.avx512bw   = (r[1] >> 30) != 0;
  f.avx512vl   = (r[1] >> 31) != 0;
  f.avx512Vbmi = (r[2] >> 1)  != 0;   // leaf 7.0 ECX
  f.avx512Vnni = (r[2] >> 11) != 0;
  f.amxTile    = (r[3] >> 24) != 0;   // leaf 7.0 EDX
  f.amxInt8    = (r[3] >> 25) != 0;
  // AVX-VNNI and AVX512-BF16 live in SUBLEAF 1 EAX (bits 4 and 5), not in subleaf 0 -- and EAX of
  // subleaf 0 is the max-subleaf count, not a feature word, so reading them from there reports 0 on
  // every CPU that actually has them (both AMD and Intel use the subleaf-1 layout). Reported only;
  // nothing gates on this, but a fingerprint that lies is worse than no fingerprint.
  if(maxSubleaf7 >= 1) {
    cpuidEx(7, 1, r);
    f.avxVnni    = (r[0] >> 4) != 0;  // leaf 7.1 EAX bit 4
    f.avx512Bf16 = (r[0] >> 5) != 0;  // leaf 7.1 EAX bit 5
  }
  return f;
}

#if defined(__linux__) && !defined(_WIN32)
#if defined(__x86_64__) && !defined(_MSC_VER)
static bool linuxIsHybridCpu() {
  unsigned r[4] = {0, 0, 0, 0};
  cpuidEx(7, 0, r);
  return ((r[3] >> 15) & 1u) != 0;   // leaf 7.0 EDX bit 15 = hybrid
}
static bool linuxOnEfficiencyCore() {
  unsigned r[4] = {0, 0, 0, 0};
  cpuidEx(0x1a, 0, r);
  const int coreType = (int)((r[0] & 0xff000000u) >> 24);
  return coreType == 0x20;           // Intel Atom = efficiency core
}
// Pin each PHYSICAL core's primary CPU in turn and ask cpuid what kind of core answers; E-cores
// are excluded (hyperthreading is not useful for linear algebra, and one pool = a barrier per op).
// Pinning the primary CPU of a core is enough to characterize the core, and it also removes
// llama.cpp's implicit assumption that logical CPUs are numbered siblings-adjacent.
static int linuxMathCores(const vector<CpuCoreEntry>& cores, string& source) {
  if(!linuxIsHybridCpu()) {
    source = "cpuid:not-hybrid";
    return 0;
  }
  cpu_set_t saved;
  if(pthread_getaffinity_np(pthread_self(), sizeof(saved), &saved) != 0) {
    source = "";
    return 0;
  }
  int math = 0;
  bool failed = false;
  for(size_t i = 0; i < cores.size(); i++) {
    cpu_set_t one;
    CPU_ZERO(&one);
    CPU_SET(cores[i].primaryCpu, &one);
    if(pthread_setaffinity_np(pthread_self(), sizeof(one), &one) != 0) {
      failed = true;
      break;
    }
    if(!linuxOnEfficiencyCore())
      math++;
  }
  pthread_setaffinity_np(pthread_self(), sizeof(saved), &saved);
  if(failed) {
    source = "";
    return 0;
  }
  source = "cpuid:leaf-0x1a";
  return math;
}
#endif
#endif // __linux__

static CpuTopology detectCpuTopology() {
  CpuTopology t;
  t.logicalCores = (int)std::thread::hardware_concurrency();
  const int logicalFallback = t.logicalCores <= 4 ? (t.logicalCores > 0 ? t.logicalCores : 1) : t.logicalCores / 2;
  t.physicalCores = 0;
  t.mathCores = 0;

  const vector<CpuCoreEntry> cores = detectPhysicalCores();
  if(!cores.empty()) {
    t.physicalCores = (int)cores.size();
#if defined(_WIN32)
    t.physicalSource = "win32:RelationProcessorCore";
#else
    t.physicalSource = "sysfs:thread_siblings";
#endif
  }
  else {
    t.physicalCores = logicalFallback;
    t.physicalSource = "fallback:logical/2";
  }

#if defined(_WIN32)
  // EfficiencyClass is per core record; 0 on systems that do not report it (all non-hybrid CPUs),
  // so "== lowest class seen" degenerates to "keep everything" exactly when it should.
  int minEff = 0xFF;
  int atMinEff = 0;
  for(size_t i = 0; i < cores.size(); i++) {
    const int eff = cores[i].efficiencyClass;
    if(eff < minEff) {
      minEff = eff;
      atMinEff = 1;
    }
    else if(eff == minEff)
      atMinEff++;
  }
  if(atMinEff > 0) {
    t.mathCores = atMinEff;
    t.mathSource = t.physicalSource + (atMinEff == t.physicalCores
      ? " (EfficiencyClass: no E-core)"
      : " (EfficiencyClass: excluded " + Global::intToString(t.physicalCores - atMinEff) + " E-core(s))");
  }
#elif defined(__linux__)
#if defined(__x86_64__) && !defined(_MSC_VER)
  string src;
  t.mathCores = linuxMathCores(cores, src);
  if(t.mathCores > 0)
    t.mathSource = src;
#endif
#endif

  if(t.mathCores <= 0) {
    t.mathCores = t.physicalCores;
    t.mathSource = t.physicalSource + " (no E-core probe)";
  }
  return t;
}

static const CpuTopology& cpuTopology() {
  static const CpuTopology t = detectCpuTopology();
  return t;
}

static int getNumMathCores() {
  return cpuTopology().mathCores;
}

// T of the (N,T) budget; N is numNNServerThreadsPerModel (setup.cpp). Each handle owns its own
// ggml threadpool and one pool can only run one graph at a time, so the invariant to respect is
// sum over handles (T) <= math cores. Default = all math cores, because the search threads are
// latency threads that block on the NN queue and do not get a reserved share.
static int getDefaultThreadsPerHandle() {
  const int phys = getNumMathCores();
  return phys > 0 ? phys : 1;
}

//--------------------------------------------------------------
// CPU affinity mask, in the shape llama.cpp exposes it (--cpu-mask / --cpu-range): hex groups,
// lowest bit = lowest CPU, group i covering CPUs [64*i, 64*i+64).
//--------------------------------------------------------------

static int maskPopcount(const bool (&mask)[GGML_MAX_N_THREADS]) {
  int n = 0;
  for(int i = 0; i < GGML_MAX_N_THREADS; i++)
    if(mask[i])
      n++;
  return n;
}

// Index of the k-th set bit (0-based), or -1.
static int maskNthSet(const bool (&mask)[GGML_MAX_N_THREADS], int k) {
  for(int i = 0; i < GGML_MAX_N_THREADS; i++) {
    if(!mask[i])
      continue;
    if(k == 0)
      return i;
    k--;
  }
  return -1;
}

static bool parseCpuMask(const string& spec, bool (&mask)[GGML_MAX_N_THREADS], string& err) {
  memset(mask, 0, sizeof(mask));
  if(spec.empty()) {
    err = "empty CPU mask";
    return false;
  }
  size_t pos = 0;
  int group = 0;
  while(true) {
    const size_t comma = spec.find(',', pos);
    string tok = spec.substr(pos, comma == string::npos ? string::npos : comma - pos);
    while(!tok.empty() && isspace((unsigned char)tok.front()))
      tok.erase(tok.begin());
    while(!tok.empty() && isspace((unsigned char)tok.back()))
      tok.pop_back();
    if(tok.empty()) {
      err = "empty group in CPU mask";
      return false;
    }
    if(group >= GGML_MAX_N_THREADS / 64) {
      err = "too many groups in CPU mask";
      return false;
    }
    if(tok.size() > 2 && tok[0] == '0' && (tok[1] == 'x' || tok[1] == 'X'))
      tok = tok.substr(2);
    if(tok.size() > 16) {
      err = "CPU mask group longer than 64 bits";
      return false;
    }
    uint64_t bits = 0;
    for(size_t i = 0; i < tok.size(); i++) {
      const char c = tok[i];
      int d;
      if(c >= '0' && c <= '9') d = c - '0';
      else if(c >= 'a' && c <= 'f') d = c - 'a' + 10;
      else if(c >= 'A' && c <= 'F') d = c - 'A' + 10;
      else {
        err = string("bad hex digit '") + c + "' in CPU mask";
        return false;
      }
      bits = (bits << 4) | (uint64_t)d;
    }
    for(int b = 0; b < 64; b++) {
      if((bits >> b) & 1ull)
        mask[group * 64 + b] = true;
    }
    group++;
    if(comma == string::npos)
      break;
    pos = comma + 1;
  }
  return true;
}

static bool parseCpuRange(const string& spec, bool (&mask)[GGML_MAX_N_THREADS], string& err) {
  memset(mask, 0, sizeof(mask));
  const size_t dash = spec.find('-');
  if(dash == string::npos) {
    err = "expected <lo>-<hi>";
    return false;
  }
  const string loStr = spec.substr(0, dash);
  const string hiStr = spec.substr(dash + 1);
  char* endLo = NULL;
  char* endHi = NULL;
  const long lo = strtol(loStr.c_str(), &endLo, 10);
  const long hi = strtol(hiStr.c_str(), &endHi, 10);
  if(endLo == loStr.c_str() || *endLo != '\0' || endHi == hiStr.c_str() || *endHi != '\0' ||
     lo < 0 || hi < lo || hi >= GGML_MAX_N_THREADS) {
    err = "bad range (want 0 <= lo <= hi < " + std::to_string((long)GGML_MAX_N_THREADS) + ")";
    return false;
  }
  for(long i = lo; i <= hi; i++)
    mask[i] = true;
  return true;
}

// Parse "low|normal|medium|high|realtime" or the numeric ggml_sched_priority value.
static bool parseSchedPrio(const string& spec, int& prioOut, string& err) {
  string s = spec;
  for(size_t i = 0; i < s.size(); i++)
    s[i] = (char)tolower((unsigned char)s[i]);
  if(s == "low")           { prioOut = GGML_SCHED_PRIO_LOW;      return true; }
  if(s == "normal")        { prioOut = GGML_SCHED_PRIO_NORMAL;   return true; }
  if(s == "medium")        { prioOut = GGML_SCHED_PRIO_MEDIUM;   return true; }
  if(s == "high")          { prioOut = GGML_SCHED_PRIO_HIGH;     return true; }
  if(s == "realtime")      { prioOut = GGML_SCHED_PRIO_REALTIME; return true; }
  char* end = NULL;
  const long v = strtol(s.c_str(), &end, 10);
  if(end != s.c_str() && *end == '\0' && v >= GGML_SCHED_PRIO_LOW && v <= GGML_SCHED_PRIO_REALTIME) {
    prioOut = (int)v;
    return true;
  }
  err = "want low|normal|medium|high|realtime (or " + std::to_string((long)GGML_SCHED_PRIO_LOW) + ".." +
         std::to_string((long)GGML_SCHED_PRIO_REALTIME) + ")";
  return false;
}

//--------------------------------------------------------------
// Thread-pool parameters for one handle. Shaped like llama.cpp's CLI (-C/--cpu-mask,
// -Cr/--cpu-range, --cpu-strict, --prio, --poll) so the same knobs mean the same thing here;
// env wins over cfg, and both are reported in the log with their source.
//--------------------------------------------------------------
struct ThreadPoolConfig {
  int poll;
  int prio;
  bool strictCpu;
  bool useMask;
  bool mask[GGML_MAX_N_THREADS];
  int numHandles;          // N (numNNServerThreadsPerModel), for per-handle mask slicing
  string source;           // "default" | "env" | "cfg" | "env+cfg"

  ThreadPoolConfig() : poll(50), prio(GGML_SCHED_PRIO_NORMAL), strictCpu(false), useMask(false), numHandles(1) {
    memset(mask, 0, sizeof(mask));
    source = "default";
  }
};

// ggml-cpu entry points that cannot be linked directly.
//
// In a GGML_CPU_ALL_VARIANTS build the kernels live in the selected ggml-cpu-<tag>.dll and are
// exported only through the backend registry's proc-address table (ggml-cpu.cpp:651-690); the
// executable has no ggml-cpu import library, so the plain symbols do not exist there. The same
// lookups work in a static build, which is why this is the single code path for both shapes.
//
// getFeatures is the important one. Its list is built with compile-time gates
// (ggml-cpu.c:3639-3645: `#if defined(__AVX512VNNI__)` else 0), so it reports what the LOADED
// VARIANT was compiled with -- not merely what the CPU supports. That is exactly the signal a
// per-ISA decision (Q8 gating) and the machine fingerprint need, and the reason nothing in this
// file calls ggml_cpu_has_* through a directly linked symbol.
struct CpuApi {
  ggml_backend_set_n_threads_t setNThreads;
  ggml_threadpool_t (*threadpoolNew)(ggml_threadpool_params*);
  void (*threadpoolFree)(ggml_threadpool_t);
  void (*setThreadpool)(ggml_backend_t, ggml_threadpool_t);
  ggml_backend_get_features_t getFeatures;

  CpuApi()
    : setNThreads(NULL), threadpoolNew(NULL), threadpoolFree(NULL),
      setThreadpool(NULL), getFeatures(NULL)
  {}
};

static CpuApi resolveCpuApi(ggml_backend_dev_t dev) {
  CpuApi api;
  ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
  if(reg == NULL)
    return api;
  api.setNThreads    = (ggml_backend_set_n_threads_t)ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
  api.threadpoolNew  = (ggml_threadpool_t (*)(ggml_threadpool_params*))ggml_backend_reg_get_proc_address(reg, "ggml_threadpool_new");
  api.threadpoolFree = (void (*)(ggml_threadpool_t))ggml_backend_reg_get_proc_address(reg, "ggml_threadpool_free");
  api.setThreadpool  = (void (*)(ggml_backend_t, ggml_threadpool_t))ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_set_threadpool");
  api.getFeatures    = (ggml_backend_get_features_t)ggml_backend_reg_get_proc_address(reg, "ggml_backend_get_features");
  return api;
}

// Loaded once, before the first device lookup. This must not depend on the caller having run
// NeuralNet::globalInitialize(): commands differ in whether they call Setup::initializeSession
// (benchmarknn did not), and a static build hides that mistake because its CPU device is
// registered at static-init. ggml_backend_load_all() is idempotent -- the registry returns the
// already-loaded backend for a repeated path -- so calling it from both places is safe, and it
// is what keeps a dl-loaded ggml-cpu-<tag>.dll backend self-sufficient.
// Directory of the running executable, in the same shape ggml's own loader builds it
// (ggml-backend-reg.cpp:420-478: GetModuleFileNameW on Windows, /proc/self/exe on Linux).
static string getExecutableDir() {
#if defined(_WIN32)
  vector<wchar_t> buf(MAX_PATH);
  DWORD len = GetModuleFileNameW(NULL, buf.data(), (DWORD)buf.size());
  while(len == buf.size()) {
    buf.resize(buf.size() * 2);
    len = GetModuleFileNameW(NULL, buf.data(), (DWORD)buf.size());
  }
  if(len == 0)
    return string();
  wstring wide(buf.data(), len);
  const size_t slash = wide.find_last_of(L"\\/");
  if(slash == wstring::npos)
    return string();
  wide.resize(slash);
  // ASCII paths only: the model/dll paths this backend deals with are ASCII in practice, and a
  // UTF-8 conversion here would need a code page dance for no gain.
  return string(wide.begin(), wide.end());
#elif defined(__linux__)
  char buf[4096];
  const ssize_t len = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
  if(len <= 0)
    return string();
  buf[len] = '\0';
  string path(buf);
  const size_t slash = path.find_last_of('/');
  return slash == string::npos ? string() : path.substr(0, slash);
#else
  return string();
#endif
}

// KATAGO_CPU_VARIANT=<tag>: force one ggml-cpu ISA variant instead of letting the loader pick the
// highest-scoring one. Why this exists (measured, 2026-09-21): "highest score" is a feature-count
// ordering, NOT a speed ordering. On this 9800X3D the AVX2 tier (haswell) beats the AVX-512 tier
// (cascadelake, what the loader picks) by 1.44x on b10c384 -- so without this knob, comparing ISA
// tiers needs copying dll sets around, and users have no way to work around a slow tier.
//
// Mechanism: ggml_backend_dev_by_type() returns the FIRST registered device of a type
// (ggml-backend-reg.cpp:355-363), so loading the wanted variant BEFORE ggml_backend_load_all()
// makes it the one the backend binds to. Loading the same path twice is harmless (the registry
// dedups by path). The env var (not a cfg key) on purpose: backends are loaded by
// Setup::initializeSession -> globalInitialize, which has no ConfigParser.
static void ensureBackendsLoadedOnce() {
  static std::once_flag once;
  std::call_once(once, []() {
    const char* want = getenv("KATAGO_CPU_VARIANT");
    if(want != NULL && want[0] != '\0') {
      const string dir = getExecutableDir();
#if defined(_WIN32)
      const string file = string("ggml-cpu-") + want + ".dll";
#else
      const string file = string("libggml-cpu-") + want + ".so";
#endif
      const string path = dir.empty() ? file : (dir + "/" + file);
      if(ggml_backend_load(path.c_str()) == NULL)
        cerr << "CPU backend: KATAGO_CPU_VARIANT=" << want
             << " could not be loaded from " << path
             << "; falling back to the loader's own choice" << endl;
    }
    ggml_backend_load_all();
  });
}

// Resolve the pool knobs once per model. Env wins over cfg (a sweep needs no config edits), and
// the source string is carried into every log line so a measurement can never be attributed to
// the wrong layer.
static ThreadPoolConfig resolveThreadPoolConfig(ConfigParser& cfg, int threadsPerHandle) {
  ThreadPoolConfig tpc;
  bool fromCfg = false;

  if(cfg.contains("cpuggmlPoll")) {
    tpc.poll = cfg.getInt("cpuggmlPoll", 0, 100);
    fromCfg = true;
  }
  if(cfg.contains("cpuggmlPrio")) {
    string err;
    if(!parseSchedPrio(cfg.getString("cpuggmlPrio"), tpc.prio, err))
      throw StringError("CPU backend: cpuggmlPrio: " + err);
    fromCfg = true;
  }
  if(cfg.contains("cpuggmlCpuStrict"))
    tpc.strictCpu = cfg.getBool("cpuggmlCpuStrict");
  if(cfg.contains("cpuggmlCpuMask")) {
    string err;
    if(!parseCpuMask(cfg.getString("cpuggmlCpuMask"), tpc.mask, err))
      throw StringError("CPU backend: cpuggmlCpuMask: " + err);
    tpc.useMask = true;
    fromCfg = true;
  }
  if(cfg.contains("cpuggmlCpuRange")) {
    string err;
    if(!parseCpuRange(cfg.getString("cpuggmlCpuRange"), tpc.mask, err))
      throw StringError("CPU backend: cpuggmlCpuRange: " + err);
    tpc.useMask = true;
    fromCfg = true;
  }
  bool strictExplicit = cfg.contains("cpuggmlCpuStrict");

  tpc.source = fromCfg ? "cfg" : "default";

  const char* env = NULL;
  auto envOver = [&]() {
    tpc.source = fromCfg ? "env+cfg" : "env";
  };

  env = getenv("KATAGO_CPU_POLL");
  if(env != NULL && env[0] != '\0') {
    char* end = NULL;
    const long v = strtol(env, &end, 10);
    if(end == env || *end != '\0' || v < 0 || v > 100)
      throw StringError("CPU backend: KATAGO_CPU_POLL must be an integer in [0,100], got \"" +
                        string(env) + "\"");
    tpc.poll = (int)v;
    envOver();
  }
  env = getenv("KATAGO_CPU_PRIO");
  if(env != NULL && env[0] != '\0') {
    string err;
    if(!parseSchedPrio(env, tpc.prio, err))
      throw StringError("CPU backend: KATAGO_CPU_PRIO: " + err);
    envOver();
  }
  env = getenv("KATAGO_CPU_MASK");
  if(env != NULL && env[0] != '\0') {
    string err;
    if(!parseCpuMask(env, tpc.mask, err))
      throw StringError("CPU backend: KATAGO_CPU_MASK: " + err);
    tpc.useMask = true;
    envOver();
  }
  env = getenv("KATAGO_CPU_RANGE");
  if(env != NULL && env[0] != '\0') {
    if(tpc.useMask)
      throw StringError("CPU backend: KATAGO_CPU_RANGE and KATAGO_CPU_MASK are mutually exclusive");
    string err;
    if(!parseCpuRange(env, tpc.mask, err))
      throw StringError("CPU backend: KATAGO_CPU_RANGE: " + err);
    tpc.useMask = true;
    envOver();
  }
  env = getenv("KATAGO_CPU_MASK_STRICT");
  if(env != NULL && env[0] != '\0') {
    tpc.strictCpu = !(env[0] == '0' && env[1] == '\0');
    strictExplicit = true;
    envOver();
  }
  // KATAGO_CPU_AFFINITY=1 with no explicit mask: pin one pool to the lowest math cores, which is
  // the "structural partition" the design asks for and is what makes the per-handle slicing below
  // testable with a single env var. OFF by default: pinning changes how the OS schedules the search
  // threads too, so it stays a measured decision, not a silent default.
  env = getenv("KATAGO_CPU_AFFINITY");
  if(!tpc.useMask && env != NULL && env[0] == '1' && env[1] == '\0') {
    // One CPU per PHYSICAL core, not the lowest N logical CPUs: on this machine logical CPUs 0 and
    // 1 are the two SMT threads of core 0, so the naive mask would put half the pool on one core.
    const int n = getNumMathCores();
    buildAutoMask(tpc.mask, n);
    tpc.useMask = true;
    envOver();
    tpc.source += "(auto-mask=one-cpu-per-core, " + std::to_string(maskPopcount(tpc.mask)) + " cpus)";
  }
  // A mask that is not also strict is a soft hint (every worker still sees every masked CPU), which
  // is NOT a partition. Default to strict when a mask was requested and nobody said otherwise.
  if(tpc.useMask && !strictExplicit)
    tpc.strictCpu = true;

  const int masked = tpc.useMask ? maskPopcount(tpc.mask) : 0;
  if(tpc.useMask && masked < threadsPerHandle && threadsPerHandle > 0)
    throw StringError("CPU backend: CPU mask selects " + std::to_string(masked) +
                      " cpus but the pool has " + std::to_string(threadsPerHandle) +
                      " threads; not enough set bits (ggml would pin two threads to one core)");
  if(tpc.useMask)
    tpc.source += "(mask=" + std::to_string(masked) + " strict=" + (tpc.strictCpu ? "1" : "0") + ")";
  return tpc;
}

struct ComputeContext {
  int nnXLen;
  int nnYLen;
  enabled_t useFP16Mode;

  // Intra-op thread count per compute handle, resolved once for the whole evaluator so that
  // every handle of this model uses the same budget. Read HERE because createComputeContext
  // owns the ConfigParser.
  int nnThreadsPerHandle;
  std::string nnThreadsSource;

  // Pool knobs (poll/prio/cpumask/strict), same lifetime and source discipline as the thread count.
  ThreadPoolConfig tpc;

  ComputeContext(int xLen, int yLen, enabled_t fp16Mode, int threadsPerHandle, const std::string& threadsSource)
    : nnXLen(xLen), nnYLen(yLen), useFP16Mode(fp16Mode),
      nnThreadsPerHandle(threadsPerHandle), nnThreadsSource(threadsSource)
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
  ensureBackendsLoadedOnce();

  // T of the (N,T) budget: env > cfg > default(math cores, E-cores excluded).  Env wins so a
  // thread sweep needs no config edits; cfg is what a shipped release reads.
  int threadsPerHandle = getDefaultThreadsPerHandle();
  string threadsSource = "default(math cores)";
  if(cfg.contains("cpuggmlThreadsPerModel")) {
    threadsPerHandle = cfg.getInt("cpuggmlThreadsPerModel", 1, 1024);
    threadsSource = "cfg";
  }
  const char* threadsEnv = getenv("KATAGO_CPU_THREADS");
  if(threadsEnv != NULL && threadsEnv[0] != '\0') {
    char* end = NULL;
    const long v = strtol(threadsEnv, &end, 10);
    if(end == threadsEnv || v < 1 || v > 1024)
      throw StringError("CPU backend: KATAGO_CPU_THREADS must be an integer in [1,1024], got \"" +
                        string(threadsEnv) + "\"");
    threadsPerHandle = (int)v;
    threadsSource = "env";
  }

  // scale8 (1/8 activation rescaling) is applied here, before the graph is built, exactly like
  // the ONNX/TensorRT backends do. Transformer models are not eligible and are skipped
  // automatically. Left at the same default as the other backends on purpose: it changes the
  // numerics, so switching it off here would break the shared margin口径.
  (void)homeDataDirOverride;
  loadedModel->maybeApplyScale8(false, logger);

  // The requested FP16 mode is kept on the context so isUsingFP16 can report it truthfully
  // instead of hard-coding false. Weight storage is F16, activations F32, accumulation f32, so
  // useFP16=auto already means "full precision arithmetic"; useFP16=false additionally forces
  // f32 accumulation (the precision class below).
  ComputeContext* ctx = new ComputeContext(nnXLen, nnYLen, useFP16Mode, threadsPerHandle, threadsSource);
  ctx->tpc = resolveThreadPoolConfig(cfg, threadsPerHandle);
  // N of the (N,T) budget. Not derivable from gpuIdxs: nneval sorts AND DEDUPLICATES it before
  // createComputeContext (nneval.cpp:136-138), so on this backend it always arrives as {0}. The
  // config key is the same one Setup used to build the server threads, so read it here.
  ctx->tpc.numHandles = cfg.contains("numNNServerThreadsPerModel")
    ? cfg.getInt("numNNServerThreadsPerModel", 1, 1024) : 1;
  if(logger != NULL) {
    const CpuTopology& topo = cpuTopology();
    logger->write("CPU backend: creating compute context for " +
                  Global::intToString(nnXLen) + "x" + Global::intToString(nnYLen) +
                  " (threadsPerHandle=" + Global::intToString(threadsPerHandle) +
                  " source=" + threadsSource + ", handles=" + Global::intToString(ctx->tpc.numHandles) +
                  ", poll=" + Global::intToString(ctx->tpc.poll) +
                  ", prio=" + Global::intToString(ctx->tpc.prio) +
                  ", poolConfig=" + ctx->tpc.source + ")");
    logger->write("CPU backend: topology physical=" + Global::intToString(topo.physicalCores) +
                  " [" + topo.physicalSource + "], math=" + Global::intToString(topo.mathCores) +
                  " [" + topo.mathSource + "], logical=" + Global::intToString(topo.logicalCores));
  }
  return ctx;
}

void NeuralNet::freeComputeContext(ComputeContext* computeContext) {
  delete computeContext;
}

//--------------------------------------------------------------

// ONE implementation of "force f32 accumulation on the F16-src0 mul_mats".  Both the env override
// (KATAGO_CPU_MMPREC_F32) and the REQUESTED precision come through here, so they cannot drift into
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

// ggml-cpu has exactly ONE graph fusion: {RMS_NORM, MUL} (ggml-cpu.c:3087-3104), and it is a
// runtime pattern match with a hard predicate -- the Mul must directly follow the RMS_NORM in the
// cgraph, all three tensors must be F32, and the weight view must be contiguous F32 with the same
// ne[0]. Whether OUR graph satisfies it is not obvious (weights are F16 everywhere else), so this
// reports the necessary conditions it can check with public APIs. ggml's own fuser additionally
// requires the RMS_NORM result to have exactly one use; the src-use count below is that check in
// the form our graphs can produce. GGML_CPU_DISABLE_FUSION=1 is the A/B arm.
static void logFusionCandidates(ggml_cgraph* g, Logger* logger) {
  if(logger == NULL)
    return;
  const int n = ggml_graph_n_nodes(g);
  int rmsNodes = 0;
  int adjacent = 0;
  int singleUse = 0;
  int predicateOk = 0;
  for(int i = 0; i + 1 < n; i++) {
    ggml_tensor* node = ggml_graph_node(g, i);
    if(node->op != GGML_OP_RMS_NORM)
      continue;
    rmsNodes++;
    ggml_tensor* mul = ggml_graph_node(g, i + 1);
    if(mul->op != GGML_OP_MUL || (mul->src[0] != node && mul->src[1] != node))
      continue;
    adjacent++;
    int uses = 0;
    for(int j = 0; j < n; j++) {
      ggml_tensor* other = ggml_graph_node(g, j);
      for(int s = 0; s < GGML_MAX_SRC; s++) {
        if(other->src[s] == NULL)
          break;
        if(other->src[s] == node)
          uses++;
      }
    }
    if(uses != 1)
      continue;
    singleUse++;
    const ggml_tensor* mulW = (mul->src[0] == node) ? mul->src[1] : mul->src[0];
    if(mulW == NULL || node->src[0] == NULL)
      continue;
    if(node->src[0]->type == GGML_TYPE_F32 && mul->type == GGML_TYPE_F32 &&
       mulW->type == GGML_TYPE_F32 && mulW->ne[0] == node->ne[0] &&
       mulW->nb[0] == sizeof(float))
      predicateOk++;
  }
  const char* disableEnv = getenv("GGML_CPU_DISABLE_FUSION");
  const bool disabled = disableEnv != NULL && atoi(disableEnv) == 1;
  logger->write("CPU backend: fusion {RMS_NORM,MUL} rmsNodes=" + Global::intToString(rmsNodes) +
                " adjacent=" + Global::intToString(adjacent) +
                " singleUse=" + Global::intToString(singleUse) +
                " predicateOk=" + Global::intToString(predicateOk) +
                (disabled ? " [fuser DISABLED via GGML_CPU_DISABLE_FUSION]" : ""));
}

// Static check for the ISA-tier investigation (zero measurement cost): which BUFFER TYPE the weights
// and the graph actually got. ggml-cpu exports its extra buffer types per COMPILED VARIANT
// (ggml_backend_cpu_get_extra_buffer_types), so a variant that offers a repack type could route the
// same tensors into a different mul_mat kernel even though the graph -- built by this executable,
// which is identical for both tiers -- is bit-identical. If this line differs between tiers, the
// "isolated GEMM is flat but the in-graph GEMM is not" puzzle is explained without a profiler.
static void logBufferTypes(ggml_backend_t backend, ggml_backend_buffer_t wbuf, Logger* logger) {
  if(logger == NULL)
    return;
  ggml_backend_buffer_type_t defBuft = ggml_backend_get_default_buffer_type(backend);
  const char* defName = defBuft != NULL ? ggml_backend_buft_name(defBuft) : "(none)";
  ggml_backend_buffer_type_t wBuft = wbuf != NULL ? ggml_backend_buffer_get_type(wbuf) : NULL;
  const char* wName = wBuft != NULL ? ggml_backend_buft_name(wBuft) : "(none)";
  logger->write("CPU backend: buffers default=" + string(defName ? defName : "?") +
                " weights=" + string(wName ? wName : "?") +
                " (weights==default means plain f16 storage, no repack type in play)");
}

// KATAGO_CPU_DUMP_GRAPH=1: one line per graph node (op / dtype / ne / nb / contiguity). The graph is
// built by the executable, which is the SAME for every ISA tier, so this dump is tier-independent by
// construction -- its value is the OPPOSITE check: it proves which nodes are non-contiguous or have
// odd shapes, i.e. exactly the nodes whose kernels a microbenchmark with clean 2D tensors cannot
// reproduce. Those are the candidates for a per-tier path difference.
static void dumpGraphNodes(ggml_cgraph* g, Logger* logger) {
  if(logger == NULL)
    return;
  const char* env = getenv("KATAGO_CPU_DUMP_GRAPH");
  if(env == NULL || env[0] != '1')
    return;
  const int n = ggml_graph_n_nodes(g);
  for(int i = 0; i < n; i++) {
    ggml_tensor* t = ggml_graph_node(g, i);
    string line = Global::strprintf(
      "CPU graph node %d %s type=%s ne=[%lld,%lld,%lld,%lld] nb=[%lld,%lld,%lld,%lld] contig=%d",
      i, ggml_op_name(t->op), ggml_type_name(t->type),
      (long long)t->ne[0], (long long)t->ne[1], (long long)t->ne[2], (long long)t->ne[3],
      (long long)t->nb[0], (long long)t->nb[1], (long long)t->nb[2], (long long)t->nb[3],
      ggml_is_contiguous(t) ? 1 : 0);
    for(int s = 0; s < GGML_MAX_SRC; s++) {
      if(t->src[s] == NULL)
        break;
      line += Global::strprintf(" src%d=%s/%s%s", s, ggml_op_name(t->src[s]->op),
                                ggml_type_name(t->src[s]->type),
                                ggml_is_contiguous(t->src[s]) ? "" : "(noncontig)");
    }
    logger->write(line);
  }
  logger->write("CPU graph node dump: " + Global::intToString(n) + " nodes");
}

struct ComputeHandle {
  ComputeContext* ctx;
  const LoadedModel* loadedModel;
  Logger* logger;
  int maxBatchSize;
  int serverThreadIdx;   // ServerThreadIdx of this handle; selects its slice of the CPU mask

  ggml_backend_t backend;
  ggml_threadpool_t threadpool;   // persistent intra-op pool, NULL when nnThreads == 1
  void (*threadpoolFreeFn)(ggml_threadpool_t);   // lives in the variant dll, resolved via the registry
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

  // Batch fill accounting: the whole point of the (N,T,batch) discussion is that a forward costs
  // the graph's batch dim, not the number of rows actually filled, so the fill rate has to be
  // reported by the layer that knows both numbers. nneval already computes it (getAvgRowsPerBatch)
  // but never logs it; these counters cost two adds per eval and make production runs
  // (gtp/selfplay) self-reporting without touching nneval.cpp.
  uint64_t numRowsFilled;
  uint64_t numRowsPadded;
  uint64_t numForwards;

  ComputeHandle(ComputeContext* context, const LoadedModel* loaded, Logger* logger,
                int maxBatch, bool requireExactNNLen, int deviceIdx, int serverThreadIdx_)
    : ctx(context),
      loadedModel(loaded),
      logger(logger),
      maxBatchSize(maxBatch),
      serverThreadIdx(serverThreadIdx_),
      backend(NULL),
      threadpool(NULL),
      threadpoolFreeFn(NULL),
      ggmlCtx(NULL),
      wctx(NULL),
      graph(NULL),
      gallocr(NULL),
      wbuf(NULL),
      inputSpatial(NULL),
      inputGlobal(NULL),
      inputMeta(NULL),
      inputMask(NULL),
      outPolicyPass(NULL),
      outPolicy(NULL),
      outValue(NULL),
      outScoreValue(NULL),
      outOwnership(NULL),
      padBatch(true),
      numRowsFilled(0),
      numRowsPadded(0),
      numForwards(0)
  {
    // B2a (next-temp §10.11): FA sites drop the mask tensor when the handle is
    // requireExactNNLen (mask provably all-ones, nneval hard-asserts full board).
    // TEST-ONLY override: KATAGO_CPU_FA_NULL unset = flag-driven (production);
    //   = "1"/any = force on (only valid with full-board inputs);
    //   = "0"      = force off (A/B legs + rollback under the same binary).
    const char* faNullEnv = getenv("KATAGO_CPU_FA_NULL");
    const bool faNoMask =
      (faNullEnv != nullptr && !(faNullEnv[0] == '0' && faNullEnv[1] == '\0'))
        ? true
        : (faNullEnv != nullptr ? false : requireExactNNLen);

    // The CPU device is registered once in globalInitialize (ggml_backend_load_all picks the
    // highest-scoring ISA variant dll). Binding the backend to that device is all that is
    // needed here; every ggml-cpu-specific entry point is resolved through the registry below.
    (void)deviceIdx;
    ggml_backend_dev_t cpuDev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if(cpuDev == NULL)
      throw StringError("CPU backend: no CPU device registered "
                        "(was NeuralNet::globalInitialize() called?)");
    backend = ggml_backend_dev_init(cpuDev, NULL);
    if(backend == NULL)
      throw StringError("CPU backend: ggml_backend_dev_init failed for the CPU device");

    // Persistent intra-op thread pool. Without one, ggml_graph_compute builds and tears down a
    // disposable pool around every forward (ggml-cpu.c:3367-3372 / :3425-3427); with GGML_OPENMP
    // off that means creating and joining T-1 real threads per evaluation. T==1 spawns no worker
    // even on the disposable path (ggml-cpu.c:3330), so a 1-thread pool is created only when it
    // buys something: pinning/raising the CALLING thread, which for this backend IS the NN server
    // thread (ggml-cpu.c:3337-3344 applies workers[0]'s mask/prio to it).
    const int nnThreads = ctx->nnThreadsPerHandle;
    const CpuApi api = resolveCpuApi(cpuDev);
    if(api.setNThreads != NULL)
      api.setNThreads(backend, nnThreads);

    // This handle's slice of the CPU mask. Handles are numbered by their server thread; slicing
    // by serverThreadIdx is what turns "one mask" into a partition. Note the trap this avoids:
    // every pool's own cpumask cursor starts at 0 (ggml-cpu.c:3328-3337), so handing the same
    // mask to N one-thread pools would pin ALL of them to the mask's first CPU.
    bool handleMask[GGML_MAX_N_THREADS];
    memset(handleMask, 0, sizeof(handleMask));
    bool haveHandleMask = false;
    if(ctx->tpc.useMask) {
      const int total = maskPopcount(ctx->tpc.mask);
      const int need = nnThreads > 0 ? nnThreads : 1;
      const int n = ctx->tpc.numHandles > 0 ? ctx->tpc.numHandles : 1;
      const int start = serverThreadIdx * need;
      if(n * need <= total && start + need <= total) {
        for(int k = 0; k < need; k++) {
          const int cpu = maskNthSet(ctx->tpc.mask, start + k);
          if(cpu >= 0)
            handleMask[cpu] = true;
        }
        haveHandleMask = true;
      }
      else {
        // Cannot partition (mask too small for N x T): every handle gets the whole mask. The
        // pools then contend for the same CPUs, which is exactly what a measurement must see.
        memcpy(handleMask, ctx->tpc.mask, sizeof(handleMask));
        haveHandleMask = true;
        if(logger != NULL && n * need > total)
          logger->write("CPU backend: WARNING mask has " + Global::intToString(total) +
                        " cpus but N x T = " + Global::intToString(n) + " x " +
                        Global::intToString(need) + "; every handle pins the same set");
      }
    }

    const bool wantPool = (nnThreads > 1) || haveHandleMask ||
                          ctx->tpc.prio != GGML_SCHED_PRIO_NORMAL;
    if(wantPool) {
      if(api.threadpoolNew == NULL || api.setThreadpool == NULL)
        throw StringError("CPU backend: the loaded ggml-cpu backend does not expose its "
                          "threadpool entry points");
      ggml_threadpool_params tpp = ggml_threadpool_params_default(nnThreads);
      // poll: ggml's default is 50 (hybrid spin before sleeping). With a pool that is idle while
      // the search threads think, that spin is pure waste, so it is a knob (KATAGO_CPU_POLL).
      tpp.poll = (uint32_t)ctx->tpc.poll;
      tpp.prio = (enum ggml_sched_priority)ctx->tpc.prio;
      if(haveHandleMask) {
        memcpy(tpp.cpumask, handleMask, sizeof(tpp.cpumask));
        // strict_cpu=true is one CPU per worker; false copies the whole mask to every worker,
        // which is a soft hint and NOT a partition (ggml-cpu.c:2689-2709).
        tpp.strict_cpu = ctx->tpc.strictCpu;
      }
      threadpool = api.threadpoolNew(&tpp);
      if(threadpool == NULL)
        throw StringError("CPU backend: ggml_threadpool_new failed for " +
                          Global::intToString(nnThreads) + " threads");
      api.setThreadpool(backend, threadpool);
      threadpoolFreeFn = api.threadpoolFree;
      if(logger != NULL) {
        string maskStr = "-";
        if(haveHandleMask) {
          maskStr = "";
          for(int i = 0; i < GGML_MAX_N_THREADS; i++)
            if(handleMask[i])
              maskStr += Global::intToString(i) + ",";
          if(!maskStr.empty())
            maskStr.erase(maskStr.size() - 1);
        }
        logger->write("CPU backend thread " + Global::intToString(serverThreadIdx) +
                      ": pool T=" + Global::intToString(nnThreads) +
                      " poll=" + Global::intToString(ctx->tpc.poll) +
                      " prio=" + Global::intToString(ctx->tpc.prio) +
                      " mask=" + maskStr +
                      " strict=" + (ctx->tpc.strictCpu ? "1" : "0"));
      }
    }

    // No tuner hook here: ggml-cpu decides its own blocking internally, so there is no
    // per-device pipeline choice to install before the graph is built.

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
        logger->write("CPU backend: building ONNX graph from model weights...");
      OnnxModelBuilder::Result onnxResult = OnnxModelBuilder::build(loadedModel->modelDesc, buildParams, logger);
      builtOnnxBytes = onnxResult.serializedModel;
      onnxBytesPtr = &builtOnnxBytes;
      auto proto = std::make_shared<onnx::ModelProto>();
      if(!proto->ParseFromArray(onnxBytesPtr->data(), (int)onnxBytesPtr->size()))
        throw StringError("CPU backend: could not parse the built ONNX graph");
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
      throw StringError("CPU backend: ggml_init failed");

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
        throw StringError("CPU backend: ggml_init(wctx) failed");
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
      throw StringError("CPU backend: graph is missing expected inputs InputSpatial/InputGlobal/InputMask");

    builder.build(inputTensors);

    // --- Resolve outputs ---
    // getOutput decodes in NCHW (channel-strided), so CS-domain head outputs
    // get a CS->NCHW conversion edge here (design §11.3 "heads 出口插转换边").
    auto getOut = [&](const char* name) -> ggml_tensor* {
      auto it = builder.tensors.find(name);
      if(it == builder.tensors.end())
        throw StringError(string("CPU backend: graph is missing expected output ") + name);
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

    // KATAGO_CPU_MMPREC_F32 (default off = bit-identical): force F32 accumulation
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
    //   nothing names a class      -> ggml-cpu's own choice (its F16 mul_mat accumulates in f32)
    //
    // CPU note: ggml-cpu never reads the mul_mat precision parameter (no GGML_PREC reference in
    // ggml-cpu.c), and its F16 mul_mat accumulates in f32 regardless, so this class is
    // informational here. It is kept for two reasons: useFP16=false must report the same
    // "f32acc"口径 as the other backends, and a future quantized path (non-F16 src0, a real
    // precision choice) needs one place to hook into.
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
    const char* mmpEnv = getenv("KATAGO_CPU_MMPREC_F32");
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
    if(getenv("KATAGO_CPU_NOALLOC") != nullptr && getenv("KATAGO_CPU_NOALLOC")[0] == '1') {
      // Debug: allocate every tensor separately (no buffer reuse) to rule out
      // gallocr liveness/aliasing issues. Allocate the weight context first so
      // views of weight tensors in the main context can resolve their buffers.
      wbuf = ggml_backend_alloc_ctx_tensors(wctx, backend);
      if(wbuf == NULL)
        throw StringError("CPU backend: weight ctx allocation failed");
      if(ggml_backend_alloc_ctx_tensors(ggmlCtx, backend) == NULL)
        throw StringError("CPU backend: ctx tensor allocation failed");
    }
    else {
      wbuf = ggml_backend_alloc_ctx_tensors(wctx, backend);
      if(wbuf == NULL)
        throw StringError("CPU backend: weight ctx allocation failed");
      gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
      if(!ggml_gallocr_alloc_graph(gallocr, graph))
        throw StringError("CPU backend: graph allocation failed (out of device memory?)");
    }
    for(auto& kv : builder.initData) {
      ggml_tensor* t = builder.tensors[kv.first];
      ggml_backend_tensor_set(t, kv.second.data(), 0, kv.second.size());
    }

    logFusionCandidates(graph, logger);
    logBufferTypes(backend, wbuf, logger);
    dumpGraphNodes(graph, logger);

    if(logger != NULL)
      logger->write("CPU backend: graph built (" + Global::uint64ToString((uint64_t)onnxBytes.size()) + " onnx bytes, " +
                    Global::uint64ToString((uint64_t)ggml_graph_n_nodes(graph)) + " ggml nodes) [prec] accumulation=" +
                    string(precClass) + " source=" + string(precSource) + " mode=" + ctx->useFP16Mode.toString());
  }

  ~ComputeHandle() {
    if(logger != NULL && numForwards > 0) {
      const double fill = numRowsPadded > 0
        ? (double)numRowsFilled / (double)numRowsPadded : 0.0;
      logger->write("CPU backend thread " + Global::intToString(serverThreadIdx) +
                    ": forwards=" + Global::uint64ToString(numForwards) +
                    " rowsFilled=" + Global::uint64ToString(numRowsFilled) +
                    " rowsPadded=" + Global::uint64ToString(numRowsPadded) +
                    " fill=" + Global::doubleToString(fill) +
                    " avgRowsPerForward=" + Global::doubleToString((double)numRowsFilled / (double)numForwards));
    }
    if(wbuf != NULL)
      ggml_backend_buffer_free(wbuf);
    if(gallocr != NULL)
      ggml_gallocr_free(gallocr);
    if(backend != NULL)
      ggml_backend_free(backend);
    if(threadpool != NULL && threadpoolFreeFn != NULL)
      threadpoolFreeFn(threadpool);
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
    throw StringError("CPU backend: inputsUseNHWC = true not supported, must use NCHW");

  if(logger != NULL)
    logger->write("CPU backend thread " + Global::intToString(serverThreadIdx) +
                  ": Model version " + Global::intToString(loadedModel->modelDesc.modelVersion));

  try {
    return new ComputeHandle(context, loadedModel, logger, maxBatchSize, requireExactNNLen,
                             gpuIdxForThisThread, serverThreadIdx);
  }
  catch(const std::exception& e) {
    // nn-server threads do not catch, so an escaping exception would become a
    // silent fastfail on Windows; log the cause first.
    string msg = string("CPU backend: compute handle creation failed: ") + e.what();
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
  // Register the backends found next to the executable. In a GGML_CPU_ALL_VARIANTS build this
  // is what loads the single highest-scoring ggml-cpu-<tag>.dll (ggml-backend-reg.cpp:504-571);
  // in a static single-variant build the CPU backend is already registered at startup and this
  // call is a no-op.
  ensureBackendsLoadedOnce();

  // KATAGO_CPU_PRINT_DEVICES=1: dispatch assertion + machine fingerprint in one line set.
  // The loader already states which dll it chose ("loaded CPU backend from <path>",
  // ggml-backend-reg.cpp:259, not NDEBUG-guarded), but that line carries no build identity and
  // can be filtered out; this prints the loaded variant's COMPILED feature list (compile-time
  // gated, so it identifies the tier), the core counts, and the device name. Sweep logs carry it
  // so a row can never be attributed to the wrong tier.
  const char* printDevEnv = getenv("KATAGO_CPU_PRINT_DEVICES");
  if(printDevEnv != NULL && printDevEnv[0] == '1')
    NeuralNet::printDevices();
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
  // Fill accounting (reported at handle teardown): a forward costs runBatchSize rows of work no
  // matter how many rows are real, so both numbers are needed to make padding visible.
  gpuHandle->numRowsFilled += (uint64_t)batchSize;
  gpuHandle->numRowsPadded += (uint64_t)runBatchSize;
  gpuHandle->numForwards++;
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
    throw StringError("CPU backend: graph compute failed");

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
  // Empty by design: the compile-time backend name already says everything a report needs, and
  // the ISA variant is deliberately NOT disclosed here -- a Q8 run must keep claiming fp16 (see
  // the design doc's precision口径 section), otherwise the same machine would report two
  // different device classes depending on a build-time or config-time detail.
  return std::string();
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

void NeuralNet::printDevices() {
  // Machine fingerprint. ensureBackendsLoadedOnce() guarantees the variant dll is loaded, so the
  // device and its feature list describe the dll that was actually selected -- the mechanical
  // proof that the variant framework works, and the machine fingerprint of the five-machine
  // matrix (run this on each machine).
  ensureBackendsLoadedOnce();
  ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
  if(dev == NULL) {
    cout << "CPU backend: no CPU device registered (globalInitialize not called?)" << endl;
    return;
  }
  ggml_backend_dev_props props;
  ggml_backend_dev_get_props(dev, &props);
  cout << "CPU device: " << (props.name != NULL ? props.name : "?")
       << " | description: " << (props.description != NULL ? props.description : "") << endl;
  {
    const char* wantVariant = getenv("KATAGO_CPU_VARIANT");
    if(wantVariant != NULL && wantVariant[0] != '\0')
      cout << "CPU variant override requested: " << wantVariant
           << " (KATAGO_CPU_VARIANT; the loader's score-based pick is bypassed)" << endl;
  }

  const CpuTopology& topo = cpuTopology();
  cout << "CPU cores: " << topo.physicalCores << " physical [" << topo.physicalSource << "]"
       << ", " << topo.mathCores << " math [" << topo.mathSource << "]"
       << ", " << topo.logicalCores << " logical" << endl;
  {
    // The core map makes the SMT numbering visible, which is what an affinity mask has to respect
    // (siblings are adjacent here, so "the lowest N CPUs" is not "N cores").
    const vector<CpuCoreEntry> coreMap = detectPhysicalCores();
    if(!coreMap.empty()) {
      cout << "CPU core map (logical cpus per physical core):";
      const size_t nShow = coreMap.size() < 16 ? coreMap.size() : 16;
      for(size_t i = 0; i < nShow; i++)
        cout << " " << coreMap[i].siblings;
      if(coreMap.size() > nShow)
        cout << " ...";
      cout << endl;
    }
  }

  // The feature list comes from the LOADED VARIANT dll and is built from compile-time gates, so
  // it identifies which tier was selected -- the device description does not (it is just "CPU").
  // This is the mechanical proof that the variant framework works in a Release build, where the
  // loader's own "loaded <path> / score:" lines are compiled out.
  //
  // Two feature sets are printed on purpose, and the rule for using them is not negotiable:
  //   1. variant  = compiled into the loaded dll  -> THE ONLY sound input to a feature gate
  //   2. cpuid    = what this machine provides    -> reporting/fingerprint only
  // Reading #2 for a gate would decide "VNNI is available" on a machine whose loaded dll was built
  // without it (see the design doc §3.12-B). The mismatch line below exists so that the upstream
  // MSVC scoring gap cannot hide: it makes "cpuid has X, the loaded variant does not" visible.
  bool variantHasVbmi = false;
  bool variantHasVnni = false;
  bool variantHasAvx512 = false;
  const CpuApi api = resolveCpuApi(dev);
  if(api.getFeatures != NULL) {
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    cout << "CPU variant features (compiled into the loaded dll):";
    for(ggml_backend_feature* f = api.getFeatures(reg); f != NULL && f->name != NULL; f++) {
      cout << " " << f->name << "=" << (f->value != NULL ? f->value : "?");
      const string name = f->name;
      if(name == "AVX512_VBMI")
        variantHasVbmi = true;
      else if(name == "AVX512_VNNI")
        variantHasVnni = true;
      else if(name == "AVX512")
        variantHasAvx512 = true;
    }
    cout << endl;
  }
  else
    cout << "CPU variant features (compiled into the loaded dll): (backend does not expose "
            "ggml_backend_get_features)" << endl;

  const CpuidFeatures hw = detectCpuidFeatures();
  cout << "CPU cpuid features (this machine, report only -- never gate on these):"
       << " AVX2=" << (hw.avx2 ? 1 : 0)
       << " AVX512=" << (hw.avx512f ? 1 : 0)
       << " AVX512_VNNI=" << (hw.avx512Vnni ? 1 : 0)
       << " AVX512_VBMI=" << (hw.avx512Vbmi ? 1 : 0)
       << " AVX512_BF16=" << (hw.avx512Bf16 ? 1 : 0)
       << " AVX_VNNI=" << (hw.avxVnni ? 1 : 0)
       << " AMX_INT8=" << (hw.amxInt8 ? 1 : 0) << endl;

  if(api.getFeatures != NULL && hw.avx512Vbmi && variantHasAvx512 && !variantHasVbmi)
    cout << "CPU feature mismatch: cpuid reports AVX512_VBMI but the loaded variant was not built "
            "with it. Known upstream MSVC scoring gap (ggml defines __AVX512VBMI__ but not "
            "GGML_AVX512_VBMI, so cpu-feats cannot award the bit); no kernel consumes VBMI today. "
            "See the design doc §9 U-1." << endl;
  if(api.getFeatures != NULL && hw.avx512Vnni && !variantHasVnni)
    cout << "CPU feature mismatch: cpuid reports AVX512_VNNI but the loaded variant was not built "
            "with it. Do NOT gate quantized paths on cpuid: use this variant feature list." << endl;
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

#endif  // USE_CPUGGML_BACKEND

