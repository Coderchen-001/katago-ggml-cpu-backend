#define _CRT_SECURE_NO_DEPRECATE // Disables "unsafe" warnings on Windows
#define _USE_MATH_DEFINES // For M_PI on MSVC

#include "ggml-backend-impl.h"
#include "ggml-backend.h"
#include "traits.h"
#include "ggml-cpu-impl.h"
#include "ggml-impl.h"
#include "quants.h"
#include "ggml-threading.h"
#include "unary-ops.h"
#include "binary-ops.h"
#include "vec.h"
#include "ops.h"
#include "ggml.h"
#include "common.h"

#if defined(_MSC_VER) || defined(__MINGW32__)
#include <malloc.h> // using malloc.h with MSC/MINGW
#elif !defined(__FreeBSD__) && !defined(__NetBSD__) && !defined(__OpenBSD__)
#include <alloca.h>
#endif

#include <assert.h>
#include <errno.h>
#include <time.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <inttypes.h>
#include <stdio.h>
#include <float.h>
#include <limits.h>
#include <stdarg.h>
#include <signal.h>
#if defined(__gnu_linux__)
#include <syscall.h>
#endif

#ifdef GGML_USE_OPENMP
#include <omp.h>
#endif

#if defined(__ARM_FEATURE_SVE) || defined(__ARM_FEATURE_MATMUL_INT8)
#undef GGML_USE_LLAMAFILE
#endif

#ifdef GGML_USE_LLAMAFILE
#include "llamafile/sgemm.h"
#endif

#ifdef GGML_USE_CPU_RISCV64_SPACEMIT
#    include "spacemit/ime.h"
#endif

// Note: once we move threading into a separate C++ file
// will use std::hardware_destructive_interference_size instead of hardcoding it here
// and we'll use C++ attribute syntax.
#define GGML_CACHE_LINE  64

#if defined(__clang__) || defined(__GNUC__)
#define GGML_CACHE_ALIGN __attribute__((aligned(GGML_CACHE_LINE)))
#endif

#if defined(__has_feature)
#if __has_feature(thread_sanitizer)
#define GGML_TSAN_ENABLED 1
#endif
#else  // __has_feature
#if defined(__SANITIZE_THREAD__)
#define GGML_TSAN_ENABLED 1
#endif
#endif // __has_feature

#define UNUSED GGML_UNUSED
#define SWAP(x, y, T) do { T SWAP = x; (x) = y; (y) = SWAP; } while (0)

// precomputed f32 table for f16 (256 KB) (simd-mappings.h)
float ggml_table_f32_f16[1 << 16];

// precomputed f32 table for e8m0 half (1 KB) (simd-mappings.h)
float ggml_table_f32_e8m0_half[1 << 8];

// precomputed f32 table for ue4m3 (1 KB) (simd-mappings.h)
float ggml_table_f32_ue4m3[1 << 8];

#if defined(__ARM_ARCH)
struct ggml_arm_arch_features_type {
    int sve_cnt;
} ggml_arm_arch_features = { 0 };
#endif

#if defined(__riscv)
struct ggml_riscv_arch_features_type {
    int rvv_vlen;
} ggml_riscv_arch_features = { 0 };
#endif

#if defined(_WIN32)

#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
    #define NOMINMAX
#endif
#include <windows.h>

#if defined(_MSC_VER) && !defined(__clang__)
#define GGML_CACHE_ALIGN __declspec(align(GGML_CACHE_LINE))

typedef volatile LONG atomic_int;
typedef atomic_int atomic_bool;
typedef atomic_int atomic_flag;

#define ATOMIC_FLAG_INIT 0

typedef enum {
    memory_order_relaxed,
    memory_order_consume,
    memory_order_acquire,
    memory_order_release,
    memory_order_acq_rel,
    memory_order_seq_cst
} memory_order;

static void atomic_store(atomic_int * ptr, LONG val) {
    InterlockedExchange(ptr, val);
}
static void atomic_store_explicit(atomic_int * ptr, LONG val, memory_order mo) {
    // TODO: add support for explicit memory order
    InterlockedExchange(ptr, val);
}
static LONG atomic_load(atomic_int * ptr) {
    return InterlockedCompareExchange(ptr, 0, 0);
}
static LONG atomic_load_explicit(atomic_int * ptr, memory_order mo) {
    // TODO: add support for explicit memory order
    return InterlockedCompareExchange(ptr, 0, 0);
}
static LONG atomic_fetch_add(atomic_int * ptr, LONG inc) {
    return InterlockedExchangeAdd(ptr, inc);
}
static LONG atomic_fetch_add_explicit(atomic_int * ptr, LONG inc, memory_order mo) {
    // TODO: add support for explicit memory order
    return InterlockedExchangeAdd(ptr, inc);
}
static atomic_bool atomic_flag_test_and_set(atomic_flag * ptr) {
    return InterlockedExchange(ptr, 1);
}
static void atomic_flag_clear(atomic_flag * ptr) {
    InterlockedExchange(ptr, 0);
}
static void atomic_thread_fence(memory_order mo) {
    MemoryBarrier();
}
#else // clang
#include <stdatomic.h>
#endif

typedef HANDLE pthread_t;

typedef DWORD thread_ret_t;
static int pthread_create(pthread_t * out, void * unused, thread_ret_t(*func)(void *), void * arg) {
    (void) unused;
    HANDLE handle = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE) func, arg, 0, NULL);
    if (handle == NULL)
    {
        return EAGAIN;
    }

    *out = handle;
    return 0;
}

static int pthread_join(pthread_t thread, void * unused) {
    (void) unused;
    int ret = (int) WaitForSingleObject(thread, INFINITE);
    CloseHandle(thread);
    return ret;
}

static int sched_yield (void) {
    Sleep (0);
    return 0;
}
#else

#include <pthread.h>
#include <stdatomic.h>
#include <sched.h>
#if defined(__FreeBSD__)
#include <pthread_np.h>
#endif

typedef void * thread_ret_t;

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#endif

typedef pthread_t ggml_thread_t;

#define GGML_THREADPOOL_N_THREADS_MASK (0xffffU)
#define GGML_THREADPOOL_N_THREADS_BITS (16)

#if defined(__APPLE__)
#include <unistd.h>
#include <mach/mach.h>
#include <TargetConditionals.h>
#endif

static const struct ggml_type_traits_cpu type_traits_cpu[GGML_TYPE_COUNT] = {
    [GGML_TYPE_F32] = {
        .from_float               = (ggml_from_float_t) ggml_cpu_fp32_to_fp32,
        .vec_dot                  = (ggml_vec_dot_t) ggml_vec_dot_f32,
        .vec_dot_type             = GGML_TYPE_F32,
        .nrows                    = 1,
    },
    [GGML_TYPE_F16] = {
        .from_float               = (ggml_from_float_t) ggml_cpu_fp32_to_fp16,
        .vec_dot                  = (ggml_vec_dot_t) ggml_vec_dot_f16,
        .vec_dot_type             = GGML_TYPE_F16,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q1_0] = {
        .from_float               = quantize_row_q1_0,
        .vec_dot                  = ggml_vec_dot_q1_0_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q2_0] = {
        .from_float               = quantize_row_q2_0,
        .vec_dot                  = ggml_vec_dot_q2_0_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q4_0] = {
        .from_float               = quantize_row_q4_0,
        .vec_dot                  = ggml_vec_dot_q4_0_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
#if defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#else
        .nrows                    = 1,
#endif
    },
    [GGML_TYPE_Q4_1] = {
        .from_float               = quantize_row_q4_1,
        .vec_dot                  = ggml_vec_dot_q4_1_q8_1,
        .vec_dot_type             = GGML_TYPE_Q8_1,
#if defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#else
        .nrows                    = 1,
#endif
    },
    [GGML_TYPE_Q5_0] = {
        .from_float               = quantize_row_q5_0,
        .vec_dot                  = ggml_vec_dot_q5_0_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q5_1] = {
        .from_float               = quantize_row_q5_1,
        .vec_dot                  = ggml_vec_dot_q5_1_q8_1,
        .vec_dot_type             = GGML_TYPE_Q8_1,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q8_0] = {
        .from_float               = quantize_row_q8_0,
        .vec_dot                  = ggml_vec_dot_q8_0_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
#if defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#else
        .nrows                    = 1,
#endif
    },
    [GGML_TYPE_Q8_1] = {
        .from_float               = quantize_row_q8_1,
        .vec_dot_type             = GGML_TYPE_Q8_1,
        .nrows                    = 1,
    },
    [GGML_TYPE_MXFP4] = {
        .from_float               = quantize_row_mxfp4,
        .vec_dot                  = ggml_vec_dot_mxfp4_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
        .nrows                    = 1,
    },
    [GGML_TYPE_NVFP4] = {
        .from_float               = quantize_row_nvfp4,
        .vec_dot                  = ggml_vec_dot_nvfp4_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q2_K] = {
        .from_float               = quantize_row_q2_K,
        .vec_dot                  = ggml_vec_dot_q2_K_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q3_K] = {
        .from_float               = quantize_row_q3_K,
        .vec_dot                  = ggml_vec_dot_q3_K_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q4_K] = {
        .from_float               = quantize_row_q4_K,
        .vec_dot                  = ggml_vec_dot_q4_K_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
#if defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#else
        .nrows                    = 1,
#endif
    },
    [GGML_TYPE_Q5_K] = {
        .from_float               = quantize_row_q5_K,
        .vec_dot                  = ggml_vec_dot_q5_K_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q6_K] = {
        .from_float               = quantize_row_q6_K,
        .vec_dot                  = ggml_vec_dot_q6_K_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
#if defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#else
        .nrows                    = 1,
#endif
    },
    [GGML_TYPE_IQ2_XXS] = {
        .from_float               = NULL,
        .vec_dot                  = ggml_vec_dot_iq2_xxs_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ2_XS] = {
        .from_float               = NULL,
        .vec_dot                  = ggml_vec_dot_iq2_xs_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ3_XXS] = {
        // NOTE: from_float for iq3 and iq2_s was removed because these quants require initialization in ggml_quantize_init
        //.from_float               = quantize_row_iq3_xxs,
        .vec_dot                  = ggml_vec_dot_iq3_xxs_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ3_S] = {
        //.from_float               = quantize_row_iq3_s,
        .vec_dot                  = ggml_vec_dot_iq3_s_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ2_S] = {
        //.from_float               = quantize_row_iq2_s,
        .vec_dot                  = ggml_vec_dot_iq2_s_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ1_S] = {
        .from_float               = NULL,
        .vec_dot                  = ggml_vec_dot_iq1_s_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ1_M] = {
        .from_float               = NULL,
        .vec_dot                  = ggml_vec_dot_iq1_m_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ4_NL] = {
        .from_float               = quantize_row_iq4_nl,
        .vec_dot                  = ggml_vec_dot_iq4_nl_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ4_XS] = {
        .from_float               = quantize_row_iq4_xs,
        .vec_dot                  = ggml_vec_dot_iq4_xs_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q8_K] = {
        .from_float               = quantize_row_q8_K,
    },
    [GGML_TYPE_BF16] = {
        .from_float               = (ggml_from_float_t) ggml_cpu_fp32_to_bf16,
        .vec_dot                  = (ggml_vec_dot_t) ggml_vec_dot_bf16,
        .vec_dot_type             = GGML_TYPE_BF16,
        .nrows                    = 1,
    },
    [GGML_TYPE_TQ1_0] = {
        .from_float               = quantize_row_tq1_0,
        .vec_dot                  = ggml_vec_dot_tq1_0_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_TQ2_0] = {
        .from_float               = quantize_row_tq2_0,
        .vec_dot                  = ggml_vec_dot_tq2_0_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_I32] = {
        .from_float               = (ggml_from_float_t) ggml_cpu_fp32_to_i32,
    },
};

const struct ggml_type_traits_cpu * ggml_get_type_traits_cpu(enum ggml_type type) {
    return &type_traits_cpu[type];
}

//
// Threading defs
//

typedef pthread_t          ggml_thread_t;

#if defined(_WIN32)

typedef CONDITION_VARIABLE ggml_cond_t;
typedef SRWLOCK            ggml_mutex_t;

#define ggml_mutex_init(m)   InitializeSRWLock(m)
#define ggml_mutex_destroy(m)
#define ggml_mutex_lock(m)   AcquireSRWLockExclusive(m)
#define ggml_mutex_unlock(m) ReleaseSRWLockExclusive(m)
#define ggml_mutex_lock_shared(m)   AcquireSRWLockShared(m)
#define ggml_mutex_unlock_shared(m) ReleaseSRWLockShared(m)

#define ggml_cond_init(c)    InitializeConditionVariable(c)
#define ggml_cond_destroy(c)
#define ggml_cond_wait(c, m) SleepConditionVariableSRW(c, m, INFINITE, CONDITION_VARIABLE_LOCKMODE_SHARED)
#define ggml_cond_broadcast(c) WakeAllConditionVariable(c)

#define ggml_thread_create pthread_create
#define ggml_thread_join   pthread_join

#else

typedef pthread_cond_t     ggml_cond_t;
typedef pthread_mutex_t    ggml_mutex_t;

#define ggml_mutex_init(m)          pthread_mutex_init(m, NULL)
#define ggml_mutex_destroy(m)       pthread_mutex_destroy(m)
#define ggml_mutex_lock(m)          pthread_mutex_lock(m)
#define ggml_mutex_unlock(m)        pthread_mutex_unlock(m)
#define ggml_mutex_lock_shared(m)   pthread_mutex_lock(m)
#define ggml_mutex_unlock_shared(m) pthread_mutex_unlock(m)

#define ggml_lock_init(x)    UNUSED(x)
#define ggml_lock_destroy(x) UNUSED(x)
#if defined(__x86_64__) || (defined(_MSC_VER) && defined(_M_AMD64))
#define ggml_lock_lock(x)    _mm_pause()
#else
#define ggml_lock_lock(x)    UNUSED(x)
#endif
#define ggml_lock_unlock(x)  UNUSED(x)

#define GGML_LOCK_INITIALIZER 0
#define ggml_cond_init(c)      pthread_cond_init(c, NULL)
#define ggml_cond_destroy(c)   pthread_cond_destroy(c)
#define ggml_cond_wait(c, m)   pthread_cond_wait(c, m)
#define ggml_cond_broadcast(c) pthread_cond_broadcast(c)

#define ggml_thread_create pthread_create
#define ggml_thread_join   pthread_join

#endif

// Threadpool def
struct ggml_threadpool {
    ggml_mutex_t mutex;       // mutex for cond.var
    ggml_cond_t  cond;        // cond.var for waiting for new work

    struct ggml_cgraph * cgraph;
    struct ggml_cplan  * cplan;

    // synchronization primitives
    atomic_int n_graph;       // updated when there is work to be done (i.e each graph) holds graph and active thread counts.
    atomic_int GGML_CACHE_ALIGN n_barrier;
    atomic_int GGML_CACHE_ALIGN n_barrier_passed;
    atomic_int GGML_CACHE_ALIGN current_chunk; // currently processing chunk during Mat_Mul, shared between all the threads.

    // these are atomic as an annotation for thread-sanitizer
    atomic_bool stop;         // Used for stopping the threadpool altogether
    atomic_bool pause;        // Used for pausing the threadpool or individual threads
    atomic_int  abort;        // Used for aborting processing of a graph

    struct ggml_compute_state * workers;   // per thread state
    int          n_threads;   // Number of threads in the pool
    int32_t      prio;        // Scheduling priority
    uint32_t     poll;        // Polling level (0 - no polling)

    enum ggml_status ec;
};

// Per-thread state
struct ggml_compute_state {
#ifndef GGML_USE_OPENMP
    ggml_thread_t thrd;
    int  last_graph;
    bool pending;
#endif
    bool cpumask[GGML_MAX_N_THREADS];
    struct ggml_threadpool * threadpool;
    int ith;
};

// Helpers for polling loops
#if defined(__aarch64__) && ( defined(__clang__) || defined(__GNUC__) )
static inline void ggml_thread_cpu_relax(void) {
    __asm__ volatile("yield" ::: "memory");
}
#elif defined(__x86_64__)
static inline void ggml_thread_cpu_relax(void) {
    _mm_pause();
}
#elif defined(__riscv)
static inline void ggml_thread_cpu_relax(void) {
    #ifdef __riscv_zihintpause
        __asm__ __volatile__ ("pause");
    #else
        /* Encoding of the pause instruction */
        __asm__ __volatile__ (".4byte 0x100000F");
    #endif
}
#else
static inline void ggml_thread_cpu_relax(void) {;}
#endif

//
// NUMA support
//

#define GGML_NUMA_MAX_NODES 8
#define GGML_NUMA_MAX_CPUS 512

struct ggml_numa_node {
    uint32_t cpus[GGML_NUMA_MAX_CPUS]; // hardware threads on this node
    uint32_t n_cpus;
};

struct ggml_numa_nodes {
    enum ggml_numa_strategy numa_strategy;
    struct ggml_numa_node nodes[GGML_NUMA_MAX_NODES];
    uint32_t n_nodes;
    uint32_t total_cpus; // hardware threads on system
    uint32_t current_node; // node on which main process is execting
#if defined(__gnu_linux__)
    cpu_set_t cpuset; // cpuset from numactl
#else
    uint32_t cpuset; // no NUMA support outside of Linux at this time. Use a portable datatype
#endif
};

//
// ggml state
//

struct ggml_state {
    struct ggml_numa_nodes numa;
};

static struct ggml_state g_state = {0};

// --- barrier instrumentation (diagnostic; off unless KATAGO_CPU_BARRIER_STATS=1) -------------------
// Counts how often threads reach the barrier and how many spin iterations they burn there. The point is to
// turn "about 14% of the profile sits in the barrier wait" into a per-node figure: measured, not inferred
// from a batch-1-versus-batch-6 comparison. The counter is a relaxed add inside a loop that already spins on
// a shared atomic, so its cost is far below one iteration of the spin itself; the totals are read once at
// teardown, so nothing here needs to be exact, only representative. Zero overhead when disabled.
static int      ggml_cpu_barrier_stats_on  = 0;
static int      ggml_cpu_barrier_stats_env = -1;
static uint64_t ggml_cpu_n_barriers        = 0;
static uint64_t ggml_cpu_n_spin_iters      = 0;

static void ggml_cpu_barrier_stats_report(void);

// Cap for the exponential backoff in the barrier wait, in cpu_relax units. 0 = disabled (stock spin), which
// is also the DEFAULT and the best setting. Resolved once from KATAGO_CPU_BARRIER_BACKOFF so one binary can be
// A/B'd. Kept small on purpose: the backoff only delays the NEXT poll, so the worst case it adds to a
// completed barrier is one cap's worth of relax instructions, against waits measured at ~2.3 us.
//
// MEASURED NEGATIVE RESULT (2026-09-29) -- do not retry this expecting a win. Interleaved A/B, 7 rounds,
// T=8, batch 6, median ms/batch: off 88.36, cap 64 88.60 (+0.27%), cap 256 89.11 (+0.85%), cap 1024 90.38
// (+2.29%), monotone and non-crossing. So touching the shared cache line less often does not help: the wait
// is genuine waiting for the slowest partner, not removable polling contention. The same reasoning retires
// the more invasive "one flag per thread" barrier -- if the contention were on the critical path, backing off
// would have helped. Only removing nodes (i.e. fewer barriers) actually reduces this cost.
static int ggml_cpu_barrier_backoff_env = -1;
static int ggml_cpu_barrier_backoff     = 0;

static void ggml_cpu_barrier_backoff_init(void) {
    if (ggml_cpu_barrier_backoff_env < 0) {
        const char * e = getenv("KATAGO_CPU_BARRIER_BACKOFF");
        ggml_cpu_barrier_backoff_env = 1;
        ggml_cpu_barrier_backoff     = (e != NULL) ? atoi(e) : 0;
        if (ggml_cpu_barrier_backoff < 0)   ggml_cpu_barrier_backoff = 0;
        if (ggml_cpu_barrier_backoff > 4096) ggml_cpu_barrier_backoff = 4096;
    }
}

// Both counters are THREAD-LOCAL while the barrier runs and folded into the shared totals once per barrier.
// This matters more than it looks. The first version added to a global atomic on every barrier entry and the
// build ran 60x slower than the same build without the flag -- 5258 ms/batch against 88 ms/batch -- with the
// stats enabled AND disabled, because the branch changed the code layout. Nine hundred nodes times sixty
// iterations times seven waiting threads is tens of thousands of contended read-modify-writes on one cache
// line, laid right on top of the barrier's own atomics. Counting must not perturb what it counts.
static __declspec(thread) uint64_t ggml_cpu_tls_spin_iters = 0;
static __declspec(thread) uint64_t ggml_cpu_tls_waits      = 0;

// --- Task 3: attribute barrier spin to the NODE the barrier was waiting for ---------------------------
// One number is wanted: the share of all barrier spin that belongs to the element-wise chain family
// (MUL/ADD/SUB/DIV/SCALE). That share decides whether "fuse element-wise chains" is worth a project.
//
// The counters above cannot answer it: ggml_barrier() knows nothing about the node, and it is called AFTER the
// kernels finish, not before. So each thread accumulates spin in TLS and drains it into its own row at the
// barrier that ends the node -- the spin at a barrier is caused by the slowest thread finishing the node just
// computed, so that is the honest label.
//
// Design constraints, each one a mistake that was considered and rejected:
//  * NOT an atexit aggregate over TLS: the pool's workers never exit individually, and every barrier zeroes the
//    TLS spin field, so nothing is left to read at teardown.
//  * NOT a thread-0 aggregate: that needs either a second barrier (which changes what is being measured) or a
//    lock on the critical path.
//  * One row PER THREAD, written only by the thread that owns it: no atomics, no barrier, no pointer swap,
//    nothing new on the critical path. Rows are padded to a cache line so no two threads share one.
//  * ACCUMULATE, do not overwrite. The kernels themselves call ggml_barrier() (23 sites in ops.cpp, 4 in
//    sgemm.cpp), so keeping only "the last barrier's spin" would silently discard every in-kernel wait.
//    Everything a thread waited on since its previous drain is charged to the node that drains it.
#define GGML_CPU_BAR_OP_SLOTS (GGML_OP_COUNT + 1)                 // +1 = the graph-end barrier column
#define GGML_CPU_BAR_ROW      ((GGML_CPU_BAR_OP_SLOTS + 7) & ~7)  // pad a row to a 64-byte multiple
static uint64_t ggml_cpu_bar_spin_by_op [GGML_MAX_N_THREADS][GGML_CPU_BAR_ROW];
static uint64_t ggml_cpu_bar_waits_by_op[GGML_MAX_N_THREADS][GGML_CPU_BAR_ROW];
static uint64_t ggml_cpu_bar_spin_unattr  = 0; // slot/ith out of range; must stay 0
static uint64_t ggml_cpu_bar_waits_unattr = 0;
static int      ggml_cpu_bar_nth          = 0; // poll size, only for the "sites" column

// Spin measured in CYCLES, and the compute phase measured in cycles by the same clock.
//
// This is the part that must not be assumed. Turning "spin iterations" into "share of runtime" needs a
// conversion, and every published conversion for this project is unusable here:
//   26%  is an identity (1.78% / 6.85%), not a measurement  (NUMBER-RECONCILIATION.md 5.7)
//   31%  was back-solved from the withdrawn -O0 build
//   15.7% is `ggml_barrier` 2.22% + `ggml_graph_compute_thread` SELF 12.03% -- and the self time of the node
//         loop is dispatch overhead, not waiting. The spin loop is inlined into ggml_barrier, so a frame
//         attribution cannot separate them either.
// So measure both halves with rdtsc in the same process: spin cycles, and the span each thread spends inside
// the node loop. Their ratio is the mean fraction of a thread's compute time spent waiting at barriers --
// no external constant, and directly comparable to the profiler's number as a cross-check.
static uint64_t ggml_cpu_bar_cycles_by_op[GGML_MAX_N_THREADS][GGML_CPU_BAR_ROW];
static uint64_t ggml_cpu_bar_span_cycles   = 0; // sum over threads and graphs of (node-loop end - start)
// Nodes per op, so "in-kernel barriers = sites - nodes" is exact instead of inferred. Same rows, same rules.
static uint64_t ggml_cpu_bar_nodes_by_op[GGML_MAX_N_THREADS][GGML_CPU_BAR_ROW];

// Site-keyed tables: WHICH barrier, not which node (see ggml-cpu-impl.h). Each barrier records its OWN spin
// here, so unlike the op rows this needs no accumulate-and-drain -- a barrier knows exactly what it waited.
static uint64_t ggml_cpu_bar_site_spin  [GGML_MAX_N_THREADS][GGML_CPU_BAR_SITE_COUNT];
static uint64_t ggml_cpu_bar_site_waits [GGML_MAX_N_THREADS][GGML_CPU_BAR_SITE_COUNT];
static uint64_t ggml_cpu_bar_site_cycles[GGML_MAX_N_THREADS][GGML_CPU_BAR_SITE_COUNT];
// Did the F32->vec_dot repack run, per MUL_MAT node? Once per node, so a plain relaxed atomic is fine.
static uint64_t ggml_cpu_mm_repack_done = 0;
static uint64_t ggml_cpu_mm_repack_skip = 0;
// Lead B's materialisation copy (KATAGO_CPU_MM_REPACK_F16): counted SEPARATELY from the F32->f16 repack,
// because they are different work and gate 4 declares one number.
static uint64_t ggml_cpu_mm_repack_f16_done  = 0;
static uint64_t ggml_cpu_mm_repack_f16_elems = 0;
// Defined in binary-ops.cpp (this TU is C, so no extern "C" here), gated by KATAGO_CPU_MUL_STORE_STATS.
// Declared up here because the report that calls it sits further down the file and this build treats an
// implicit declaration as an error -- which is how the first attempt at this line failed to compile.
void katago_mul_store_report(void);
// The repack is WORK, not waiting: it reads all of src1 and writes the converted copy, and that cost lands in
// MUL_MAT's own row where no waiting-type instrument can see it. Per-thread rows, like everything else here.
static uint64_t ggml_cpu_bar_repack_elems [GGML_MAX_N_THREADS];
static uint64_t ggml_cpu_bar_repack_cycles[GGML_MAX_N_THREADS];
// Activation-range pre-flight rows (f16 exponent window); see ggml_cpu_act_range().
static float    ggml_cpu_bar_act_maxabs[GGML_MAX_N_THREADS];
static uint64_t ggml_cpu_bar_act_over  [GGML_MAX_N_THREADS];
static uint64_t ggml_cpu_bar_act_denorm[GGML_MAX_N_THREADS];
static uint64_t ggml_cpu_bar_act_n     [GGML_MAX_N_THREADS];
// Which PRODUCER op the f16-denormal values come from, so a margin failure points at a site instead of at the
// whole graph. Atomics are fine here: one add per node, and only while the flag is on.
static uint64_t ggml_cpu_act_denorm_by_op[GGML_OP_COUNT + 1];
// Timing-only ablation switch, read ONCE (the established form: no getenv on a hot path). Its purpose is in
// G-19: a work-type counter does not automatically sit on the critical path, so the repack's cost has to be
// priced by a wall-clock ablation whose operands stay finite and whose work is provably equal.
static int ggml_cpu_mm_norepack(void) {
    static int v = -1;
    if (v < 0) {
        v = (getenv("KATAGO_CPU_MM_NOREPACK") != NULL) ? 1 : 0;
    }
    return v;
}

// KATAGO_CPU_MM_REPACK_F16=1 -- lead B: keep the MATERIALISATION, drop the CONVERSION.
//
// The R2 experiment (retype a GEMM's activation to f16 so mul_mat skips its F32->f16 repack) removes two
// things at once, and its arithmetic cannot tell them apart:
//     d(mulf16) = +1.035 - P - D
// where P is the producer's f16 store and D is the GEMM reading a graph tensor instead of the scratch the
// pool wrote one barrier earlier. P is bounded near zero by the disassembly of the store loop (three
// packed instructions per 8 elements, half the store bytes), so D carries it -- but "carries it" is not
// "measured", and the residual confound is that the retype also relays out the whole arena (15,023,376
// fewer elements = 30 MB per graph, so every activation moves).
//
// This switch separates the two. With it on, a src1 that is ALREADY in vec_dot form is still copied into
// wdata, so the layout change is the retype's (same f16 sizes), the GEMM reads a buffer the pool wrote one
// barrier earlier (no D), and the copy is a pure ROW MEMCPY -- no conversion -- which reads half the bytes
// the F32->f16 repack reads. That is what makes it a candidate optimisation and not merely a control:
//     d(B) = +1.035 - M - P      =>      d(B) - d(mulf16) = D - M
// P cancels in that difference, which is why pricing P with rdtsc confirms the story rather than gating
// the decision. NOTE it only works if the three arms share one session: two separate gate runs carry a
// session offset (+-0.6% floor, spin swings 12-20%) that would swamp D - M.
//
// The copy is a copy only under three conditions, and this REFUSES rather than guesses (G-8): the
// destination type must be F16 (so nbw0 == 2), the source's fastest stride must be one element (asserted
// above), and the block size must divide the row. When they do not hold the switch does NOTHING -- it
// falls back to the retype's behaviour, never to from_float, because from_float on an f16 source would
// convert wrong. Two separate counters: mixing the copy's elements into the F32 repack's volume would
// make the one number that gate 4 declares unreadable.
static int ggml_cpu_mm_repack_f16(void) {
    static int v = -1;
    if (v < 0) {
        v = (getenv("KATAGO_CPU_MM_REPACK_F16") != NULL) ? 1 : 0;
    }
    return v;
}

// Ablation that restores the OLD single-threaded form of the two reduction kernels (MEAN / SUM_ROWS), so the
// row split's gain can be measured at all. The split is now default-ON work, and a default-ON change with no
// way back is unfalsifiable: this switch is the way back, and it is the same shape as the A1 rule inverted
// (there, the diagnostic is default-off ahead of the change; here, it is default-off behind it).
// Work is IDENTICAL in both arms -- every row is summed by exactly one thread either way -- so this is a
// WorkRelation=equal A/B, not a fusion one.
int ggml_cpu_reduce_serial(void) {
    static int v = -1;
    if (v < 0) {
        v = (getenv("KATAGO_CPU_REDUCE_SERIAL") != NULL) ? 1 : 0;
    }
    return v;
}

// PRE-FLIGHT for "store the activations as f16" (T3/T4). f16 has 10 mantissa bits but a NARROW exponent:
// above 65504 a value becomes inf, below 6.1e-5 it lands in the denormal range and loses bits. If any
// activation leaves that window, the f16 change fails for a MECHANICAL reason rather than a numerical one, and
// that is worth knowing before spending three-model margin time on it. It is also the one measurement that can
// make bf16 interesting: bf16 carries f32's exponent range with only 7 mantissa bits, so it is strictly worse
// than f16 on precision and strictly better on range. Which of the two matters here is an empirical question
// about this model, not a matter of opinion. Read-only and flag-gated, so the default path (and the output
// hash) cannot move.
int ggml_cpu_act_range(void) {
    static int v = -1;
    if (v < 0) {
        v = (getenv("KATAGO_CPU_ACT_RANGE") != NULL) ? 1 : 0;
    }
    return v;
}

// ---------------------------------------------------------------------------------------------------------
// ACTIVATION CONSUMER CENSUS (KATAGO_CPU_ACT_CENSUS=1). This decides HOW the "store the activations as f16"
// work must be split, and it is the difference between a free win and a three-model margin run:
//
//   * A tensor whose consumers are ALL GEMM is BIT-EQUAL when stored as f16: the GEMM then reads exactly the
//     value from_float would have produced (same function, same rounding mode), so the bit gate settles it and
//     no margin is needed.
//   * A tensor with even one non-GEMM consumer (ADD / MUL / RMS_NORM / CONT / ...) changes what that consumer
//     sees, which is a precision-class change: full-tier non-bit-identical, three-model margin.
//
// The only question that matters for planning is how much of the repack's VOLUME sits in the first class, so
// this counts elements with the same expression the repack counter uses (ne10*ne11*ne12*ne13, summed over the
// MUL_MAT consumers of each base tensor), not tensor counts.
//
// Tensors are keyed by their BASE (views resolved), because making a view f16 means making its base f16, and
// a node that consumes the base directly is just as much a consumer as one that consumes a view of it.
// Read-only, one-shot, and off unless the flag is set, so it cannot affect a measured arm.
// ---------------------------------------------------------------------------------------------------------
static int ggml_cpu_census_enabled(void) {
    static int v = -1;
    if (v < 0) {
        v = (getenv("KATAGO_CPU_ACT_CENSUS") != NULL) ? 1 : 0;
    }
    return v;
}
static int ggml_cpu_census_done = 0;

static struct ggml_tensor * ggml_cpu_base_tensor(struct ggml_tensor * t) {
    while (t && t->view_src) { t = t->view_src; }
    return t;
}

// LAYOUT-TRANSPARENT ops: they rewrite how a tensor is addressed, not what values it holds, so a GEMM behind
// one still sees the same numbers. Counting them as "a non-GEMM consumer" was the first version's mistake and
// it forced every tensor into the margin class by construction -- the census reported 0% bit-equal while the
// graph quite plainly contains norm outputs whose only consumer is a GEMM. Follow them instead of stopping.
static int ggml_cpu_op_layout_transparent(enum ggml_op op) {
    switch (op) {
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_CONT:
            return 1;
        default:
            return 0;
    }
}

static void ggml_cpu_act_consumer_census(struct ggml_cgraph * cgraph) {
    int64_t  mm_nodes = 0, tens_bit = 0, tens_margin = 0, tens_leaf = 0;
    uint64_t el_all = 0, el_bit = 0, el_margin = 0, el_leaf = 0;
    // per-producer-op element totals, so the report says WHICH op's output to flip first
    uint64_t el_bit_by_op[GGML_OP_COUNT + 1];
    // GEMM consumers per tensor, 1..4+. THE SAVING IS PROPORTIONAL TO (k-1): a boundary cast costs one read
    // plus one write, which is exactly what one repack costs, so moving the conversion to a boundary is free
    // only where it DEDUPLICATES -- one conversion serving k GEMMs instead of k separate conversions.
    uint64_t el_bit_by_k[5];
    uint64_t n_bit_by_k[5];
    memset(el_bit_by_op, 0, sizeof(el_bit_by_op));
    memset(el_bit_by_k, 0, sizeof(el_bit_by_k));
    memset(n_bit_by_k, 0, sizeof(n_bit_by_k));

    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * n = cgraph->nodes[i];
        if (n->op != GGML_OP_MUL_MAT && n->op != GGML_OP_MUL_MAT_ID) { continue; }
        struct ggml_tensor * a = ggml_cpu_base_tensor(n->src[1]);
        if (a == NULL) { continue; }
        mm_nodes++;
        const uint64_t el = (uint64_t) ggml_nelements(a);

        // Value-consumer closure: walk through layout-transparent nodes instead of stopping at them.
        struct ggml_tensor * stack[1024];
        struct ggml_tensor * seen_t[1024];
        int nst = 0, nseen = 0, n_val = 0, n_gemm = 0;
        int all_gemm = 1;
        stack[nst++] = a;
        while (nst > 0 && nseen < 1024) {
            struct ggml_tensor * t = stack[--nst];
            int dup = 0;
            for (int k = 0; k < nseen; k++) { if (seen_t[k] == t) { dup = 1; break; } }
            if (dup) { continue; }
            seen_t[nseen++] = t;
            for (int k = 0; k < cgraph->n_nodes; k++) {
                struct ggml_tensor * m = cgraph->nodes[k];
                int consumes = 0;
                for (int j = 0; j < GGML_MAX_SRC; j++) {
                    if (m->src[j] != NULL && ggml_cpu_base_tensor(m->src[j]) == t) { consumes = 1; break; }
                }
                if (!consumes) { continue; }
                if (ggml_cpu_op_layout_transparent(m->op)) {
                    if (nst < 1024) { stack[nst++] = m; }
                } else {
                    n_val++;
                    if (m->op == GGML_OP_MUL_MAT || m->op == GGML_OP_MUL_MAT_ID) { n_gemm++; }
                    else { all_gemm = 0; }
                }
            }
        }

        el_all += el;
        // a is the view ROOT, so a->op is the op whose kernel WRITES these values: that is the op whose output
        // has to become f16. a->src[0]->op is the op that produced a's INPUT -- one generation too early.
        const int prod_op = (int) a->op;
        if (n_val == 0 || a->src[0] == NULL) {
            tens_leaf++; el_leaf += el;         // a leaf/input: not ours to retype
        } else if (all_gemm) {
            tens_bit++;  el_bit += el;
            el_bit_by_op[prod_op] += el;
            { const int kk = (n_gemm > 4) ? 4 : ((n_gemm < 1) ? 1 : n_gemm);
              el_bit_by_k[kk] += el; n_bit_by_k[kk]++; }
        } else {
            tens_margin++; el_margin += el;
            // Name the margin-class tensors: this is the entire margin exposure of the f16-activation plan, and
            // if that margin ever fails, this is the list to start from -- not the whole graph.
            fprintf(stderr, "[act-census]   MARGIN CLASS: tensor \"%s\"  (%lld elements)  produced by %s  "
                            "consumed by more than GEMM\n",
                    a->name, (long long) ggml_nelements(a),
                    ggml_op_name(a->op));
        }
    }

    fprintf(stderr, "[act-census] MUL_MAT nodes in this graph = %lld\n", (long long) mm_nodes);
    fprintf(stderr, "[act-census] activation volume by consumer class (elements, and %% of all repacked).\n");
    fprintf(stderr, "[act-census] the count column is MUL_MAT NODES, not tensors: a base tensor consumed by k\n");
    fprintf(stderr, "[act-census] GEMMs is visited k times here. The volume column is per node, so it is exactly\n");
    fprintf(stderr, "[act-census] the repack counter's expression and the percentages do sum to 100%%.\n");
    fprintf(stderr, "[act-census]   BIT-EQUAL  (all consumers GEMM) : %8lld nodes  %14llu  %5.1f%%\n",
            (long long) tens_bit, (unsigned long long) el_bit,
            el_all ? 100.0 * (double) el_bit / (double) el_all : 0.0);
    fprintf(stderr, "[act-census]   MARGIN     (a non-GEMM consumer): %8lld nodes  %14llu  %5.1f%%\n",
            (long long) tens_margin, (unsigned long long) el_margin,
            el_all ? 100.0 * (double) el_margin / (double) el_all : 0.0);
    fprintf(stderr, "[act-census]   LEAF/input (not an activation) : %8lld nodes  %14llu  %5.1f%%\n",
            (long long) tens_leaf, (unsigned long long) el_leaf,
            el_all ? 100.0 * (double) el_leaf / (double) el_all : 0.0);
    fprintf(stderr, "[act-census]   TOTAL                          : %8lld nodes  %14llu\n",
            (long long) (tens_bit + tens_margin + tens_leaf), (unsigned long long) el_all);
    fprintf(stderr, "[act-census]   (distinct bit-equal TENSORS = sum over k of nodes/k, below)\n");
    // which producer op would have to emit f16 for the bit-equal class
    for (int op = 0; op < GGML_OP_COUNT + 1; op++) {
        if (el_bit_by_op[op] == 0) { continue; }
        fprintf(stderr, "[act-census]   bit-equal volume produced by %-18s : %14llu  %5.1f%% of repacked\n",
                (op == GGML_OP_COUNT) ? "(no producer)" : ggml_op_name((enum ggml_op) op),
                (unsigned long long) el_bit_by_op[op],
                el_all ? 100.0 * (double) el_bit_by_op[op] / (double) el_all : 0.0);
    }
    fprintf(stderr, "[act-census] bit-equal volume by GEMM-consumer count k. This table is the PRICE of the plan:\n");
    fprintf(stderr, "[act-census] a tensor with k consumers is converted k times today, so the conversion volume\n");
    fprintf(stderr, "[act-census] here is k*elements and the saving from converting ONCE is (k-1)/k of it.\n");
    for (int k = 1; k <= 4; k++) {
        fprintf(stderr, "[act-census]   k=%s : %6lld nodes  %14llu  %5.1f%% of repacked\n",
                (k == 4) ? "4+" : (k == 1 ? "1 " : (k == 2 ? "2 " : "3 ")),
                (long long) n_bit_by_k[k], (unsigned long long) el_bit_by_k[k],
                el_all ? 100.0 * (double) el_bit_by_k[k] / (double) el_all : 0.0);
    }
    {
        // nodes/k is exact for a bucket that holds only one k; the "4+" bucket mixes k>=4, so it is excluded.
        long long td = 0;
        for (int k = 1; k <= 3; k++) { td += n_bit_by_k[k] / k; }
        fprintf(stderr, "[act-census]   => distinct bit-equal TENSORS (k=1..3) = %lld  (k=4+ bucket not separable)\n", td);
    }
    {
        double dedup = 0.0;
        for (int k = 1; k <= 4; k++) { const int kk = (k == 4) ? 4 : k; dedup += (double) el_bit_by_k[k] * (double)(kk - 1) / (double) kk; }
        fprintf(stderr, "[act-census]   => a boundary cast costs one read plus one write, which is exactly one\n");
        fprintf(stderr, "[act-census]      repack, so the only saving available is deduplication: %.1f%% of the volume\n",
                el_all ? 100.0 * dedup / (double) el_all : 0.0);
    }
    fprintf(stderr, "[act-census] => the BIT-EQUAL share is what can be taken with a bit gate alone; the rest is\n");
    fprintf(stderr, "[act-census]    a margin-class change. Quote this with the model and the DLL identity.\n");
}

static __declspec(thread) uint64_t ggml_cpu_tls_repack_elems  = 0;
static __declspec(thread) uint64_t ggml_cpu_tls_repack_cycles = 0;
// Activation-range pre-flight (f16 exponent window), per thread. See ggml_cpu_act_range().
static __declspec(thread) float    ggml_cpu_tls_act_maxabs = 0.0f;
static __declspec(thread) uint64_t ggml_cpu_tls_act_over   = 0;   // |x| > 65504  -> inf in f16
static __declspec(thread) uint64_t ggml_cpu_tls_act_denorm = 0;   // 0 < |x| < 6.1e-5 -> f16 denormal
static __declspec(thread) uint64_t ggml_cpu_tls_act_n      = 0;
// ggml_barrier() receives no ith, so the thread index is stashed once per graph by the node loop.
static __declspec(thread) int ggml_cpu_tls_bar_ith = 0;
GGML_CPU_BAR_TLS int ggml_cpu_bar_site_cur = 0;

static const char * const ggml_cpu_bar_site_name[GGML_CPU_BAR_SITE_COUNT] = {
    "node-loop", "graph-end",
    "sgemm-start", "sgemm-end",
    "mulmat-chunk", "mulmatid-chunk",
    "fa-ext-1", "fa-ext-2",
    "other"
};

static __declspec(thread) uint64_t ggml_cpu_tls_pending_cycles = 0;

static inline uint64_t ggml_cpu_bar_tsc(void) {
#if defined(__x86_64__) || defined(_M_X64)
    return __rdtsc();
#else
    return 0; // no cycle counter here; the iteration counters stay valid and the cycle share reads 0
#endif
}

static __declspec(thread) uint64_t ggml_cpu_tls_pending_spin  = 0;
static __declspec(thread) uint64_t ggml_cpu_tls_pending_waits = 0;

// Drain everything this thread has waited on since its last drain into its own row. slot is node->op, or
// GGML_OP_COUNT for the graph-end barrier.
static inline void ggml_cpu_bar_spin_drain(int ith, int slot) {
    // Fold the repack accounting FIRST: a node can carry a repack and produce no wait at all (the thread that
    // is always last arrives with waits == 0), and the early return below would then drop its bytes.
    if (ggml_cpu_tls_repack_elems || ggml_cpu_tls_repack_cycles) {
        if ((unsigned) ith < GGML_MAX_N_THREADS) {
            ggml_cpu_bar_repack_elems [ith] += ggml_cpu_tls_repack_elems;
            ggml_cpu_bar_repack_cycles[ith] += ggml_cpu_tls_repack_cycles;
        }
        ggml_cpu_tls_repack_elems  = 0;
        ggml_cpu_tls_repack_cycles = 0;
    }
    // Same reason for the activation-range pre-flight: fold it before the early return below.
    if (ggml_cpu_tls_act_n) {
        if ((unsigned) ith < GGML_MAX_N_THREADS) {
            ggml_cpu_bar_act_n     [ith] += ggml_cpu_tls_act_n;
            ggml_cpu_bar_act_over  [ith] += ggml_cpu_tls_act_over;
            ggml_cpu_bar_act_denorm[ith] += ggml_cpu_tls_act_denorm;
            if (ggml_cpu_tls_act_maxabs > ggml_cpu_bar_act_maxabs[ith]) {
                ggml_cpu_bar_act_maxabs[ith] = ggml_cpu_tls_act_maxabs;
            }
        }
        ggml_cpu_tls_act_n      = 0;
        ggml_cpu_tls_act_over   = 0;
        ggml_cpu_tls_act_denorm = 0;
        ggml_cpu_tls_act_maxabs = 0.0f;
    }
    const uint64_t spin   = ggml_cpu_tls_pending_spin;
    const uint64_t waits  = ggml_cpu_tls_pending_waits;
    const uint64_t cycles = ggml_cpu_tls_pending_cycles;
    if (spin == 0 && waits == 0) {
        return;
    }
    ggml_cpu_tls_pending_spin   = 0;
    ggml_cpu_tls_pending_waits  = 0;
    ggml_cpu_tls_pending_cycles = 0;
    if ((unsigned) ith < GGML_MAX_N_THREADS && (unsigned) slot < GGML_CPU_BAR_OP_SLOTS) {
        ggml_cpu_bar_spin_by_op  [ith][slot] += spin;
        ggml_cpu_bar_waits_by_op [ith][slot] += waits;
        ggml_cpu_bar_cycles_by_op[ith][slot] += cycles;
    } else {
        ggml_cpu_bar_spin_unattr  += spin;
        ggml_cpu_bar_waits_unattr += waits;
    }
}

// Node-visit accounting, to settle why the barrier count (1344) exceeds the number of nodes that should pay a
// barrier. ggml_op_is_empty skips NONE/RESHAPE/TRANSPOSE/VIEW/PERMUTE, and a second continue skips nodes
// without GGML_TENSOR_FLAG_COMPUTE, so the loop should only reach ggml_barrier for the remainder. If the
// counted barriers disagree with that arithmetic, one of the two continuations is not doing what the code
// appears to say -- and a stride of ~561 RESHAPE nodes each paying a barrier would be several times larger
// than every fusion candidate on the list.
static uint64_t ggml_cpu_n_nodes_seen    = 0;  // node_n iterations of the main loop
static uint64_t ggml_cpu_n_nodes_empty   = 0;  // skipped by ggml_op_is_empty
static uint64_t ggml_cpu_n_nodes_nocomp  = 0;  // skipped by the COMPUTE flag test
static uint64_t ggml_cpu_n_nodes_barr    = 0;  // reached the ggml_barrier call site
static uint64_t ggml_cpu_n_nodes_compute = 0;  // reached ggml_compute_forward / the fusion path

// A1 counters, declared up here so the teardown report can read them (the A1 code itself is much further
// down). Kept as three SEPARATE counters on purpose: hit / refuse / execute. Conflating these cost this
// project a whole debugging round once ("logged 40 hits when 1 site fused").
static uint64_t ggml_cpu_t9_hit     = 0;
static uint64_t ggml_cpu_t9_refuse  = 0;
static uint64_t ggml_cpu_t9_execute = 0;
// epilogue_chunks counts the chunk passes that actually wrote something (one per non-empty chunk of every
// executed site), so "execute>0" can be told apart from "the pass ran but covered nothing".
// path_chunk / path_early / path_fallback split the executed anchors by which GEMM path carried the fused
// pass. Measured on b10c384: 51 of 52 anchors per walk come back through the llamafile sgemm (so the pass
// runs after it, reusing that kernel's own publish barrier) and 1 goes through the ggml chunk loop. Without
// this split, "the fusion did nothing" and "the fusion ran and was wrong" look identical.
static uint64_t ggml_cpu_t9_epilogue_chunks = 0;
static uint64_t ggml_cpu_t9_path_chunk      = 0;
static uint64_t ggml_cpu_t9_path_early      = 0;
static uint64_t ggml_cpu_t9_path_fallback   = 0;

// Task 3: turn the per-thread rows into one table and one verdict.
//
// The verdict is printed ONLY when the attribution invariant holds: every spin iteration the barrier counted
// must land in exactly one row. If it does not, the table is not a measurement and the honest output is the
// residual, not a percentage.
//
// The spin share of runtime is MEASURED here (spin cycles / node-loop span cycles), not assumed. The three
// external numbers are kept only as a cross-check, because all three are unusable as the conversion:
//   15.7% -- profile: ggml_barrier 2.22% + node-loop SELF 12.03%; the self time is dispatch, not waiting
//   26%   -- an identity (1.78%/6.85%), not an independent measurement (NUMBER-RECONCILIATION.md 5.7)
//   31%   -- WITHDRAWN: back-solved from the -O0 build
#define GGML_CPU_BAR_SHARE_LB 15.7
#define GGML_CPU_BAR_SHARE_MID 26.0
#define GGML_CPU_BAR_SHARE_UB 31.0
#define GGML_CPU_BAR_VERDICT_GATE 1.0

static void ggml_cpu_bar_spin_report(void) {
    uint64_t spin_by_op  [GGML_CPU_BAR_OP_SLOTS];
    uint64_t waits_by_op [GGML_CPU_BAR_OP_SLOTS];
    uint64_t cycles_by_op[GGML_CPU_BAR_OP_SLOTS];
    uint64_t nodes_by_op [GGML_CPU_BAR_OP_SLOTS];
    uint64_t spin_by_thr [GGML_MAX_N_THREADS];
    int      order       [GGML_CPU_BAR_OP_SLOTS];
    uint64_t site_spin   [GGML_CPU_BAR_SITE_COUNT];
    uint64_t site_waits  [GGML_CPU_BAR_SITE_COUNT];
    uint64_t site_cycles [GGML_CPU_BAR_SITE_COUNT];

    memset(spin_by_op,   0, sizeof(spin_by_op));
    memset(waits_by_op,  0, sizeof(waits_by_op));
    memset(cycles_by_op, 0, sizeof(cycles_by_op));
    memset(nodes_by_op,  0, sizeof(nodes_by_op));
    memset(spin_by_thr,  0, sizeof(spin_by_thr));
    memset(site_spin,    0, sizeof(site_spin));
    memset(site_waits,   0, sizeof(site_waits));
    memset(site_cycles,  0, sizeof(site_cycles));

    uint64_t row_spin = 0;
    uint64_t row_waits = 0;
    for (int t = 0; t < GGML_MAX_N_THREADS; t++) {
        for (int s = 0; s < GGML_CPU_BAR_OP_SLOTS; s++) {
            spin_by_op  [s] += ggml_cpu_bar_spin_by_op  [t][s];
            waits_by_op [s] += ggml_cpu_bar_waits_by_op [t][s];
            cycles_by_op[s] += ggml_cpu_bar_cycles_by_op[t][s];
            nodes_by_op [s] += ggml_cpu_bar_nodes_by_op [t][s];
            spin_by_thr [t] += ggml_cpu_bar_spin_by_op  [t][s];
        }
        for (int s = 0; s < GGML_CPU_BAR_SITE_COUNT; s++) {
            site_spin  [s] += ggml_cpu_bar_site_spin  [t][s];
            site_waits [s] += ggml_cpu_bar_site_waits [t][s];
            site_cycles[s] += ggml_cpu_bar_site_cycles[t][s];
        }
        row_spin  += spin_by_thr[t];
    }
    for (int s = 0; s < GGML_CPU_BAR_OP_SLOTS; s++) {
        row_waits += waits_by_op[s];
    }

    const int64_t spin_resid = (int64_t) ggml_cpu_n_spin_iters - (int64_t) (row_spin + ggml_cpu_bar_spin_unattr);
    const int64_t wait_resid = (int64_t) ggml_cpu_n_barriers   - (int64_t) (row_waits + ggml_cpu_bar_waits_unattr);

    fprintf(stderr, "[bar-spin-by-op] nth=%d spin_iters=%llu waits=%llu sum_rows=%llu unattr=%llu residual=%lld\n",
            ggml_cpu_bar_nth,
            (unsigned long long) ggml_cpu_n_spin_iters,
            (unsigned long long) ggml_cpu_n_barriers,
            (unsigned long long) row_spin,
            (unsigned long long) ggml_cpu_bar_spin_unattr,
            (long long) spin_resid);

    // Per-thread totals: a thread that consistently waits far less than its peers is the thread that arrives
    // last, and that is what a load imbalance looks like in this data. Printed compactly, one line.
    fprintf(stderr, "[bar-spin-by-op] per-thread spin:");
    for (int t = 0; t < GGML_MAX_N_THREADS && t < 64; t++) {
        if (spin_by_thr[t] != 0) {
            fprintf(stderr, " t%d=%llu", t, (unsigned long long) spin_by_thr[t]);
        }
    }
    fprintf(stderr, "\n");

    if (spin_resid != 0 || wait_resid != 0) {
        fprintf(stderr, "[bar-spin-by-op] INVARIANT FAILED: rows+unattr != global "
                        "(spin residual=%lld, waits residual=%lld, unattr_waits=%llu)\n",
                (long long) spin_resid, (long long) wait_resid,
                (unsigned long long) ggml_cpu_bar_waits_unattr);
        fprintf(stderr, "[bar-spin-by-op] NO TABLE AND NO VERDICT -- attribution is not self-consistent\n");
        return;
    }

    for (int s = 0; s < GGML_CPU_BAR_OP_SLOTS; s++) {
        order[s] = s;
    }
    for (int i = 1; i < GGML_CPU_BAR_OP_SLOTS; i++) { // insertion sort by spin, descending
        const int key = order[i];
        int j = i - 1;
        while (j >= 0 && spin_by_op[order[j]] < spin_by_op[key]) {
            order[j + 1] = order[j];
            j--;
        }
        order[j + 1] = key;
    }

    const double total = (double) ggml_cpu_n_spin_iters;
    const int n_waiters = (ggml_cpu_bar_nth > 1) ? (ggml_cpu_bar_nth - 1) : 1;

    // The `cycles` column is NOT independent evidence: it is `spin * cyc_per_iter`, the same quantity as
    // the spin column in different units (G-27). Measured per op, M cycles: MUL 263.7 vs 262.8, FA 210.4
    // vs 207.9, ADD 105.5 vs 105.4, CONT 68.6 vs 68.2. The header says so, because one column printed
    // twice is exactly how a derived quantity gets quoted as a second, corroborating measurement -- and
    // "spin == cycles therefore the node ran on one thread" is a conclusion this column cannot support.
    // (Relabelled, not removed: run-knob-ab.ps1's Get-OpStructure parses these columns positionally.)
    fprintf(stderr, "[bar-spin-by-op] %-22s %14s %9s %16s %12s %10s %8s %8s\n",
            "op", "spin", "spin%", "cycles:spin*c/i", "waits", "sites", "nodes", "kern");
    for (int i = 0; i < GGML_CPU_BAR_OP_SLOTS; i++) {
        const int s = order[i];
        if (spin_by_op[s] == 0 && waits_by_op[s] == 0) {
            continue;
        }
        const char * name = (s == GGML_OP_COUNT) ? "graph-end" : ggml_op_name((enum ggml_op) s);
        const uint64_t sites = waits_by_op[s] / (uint64_t) n_waiters;
        // nodes_by_op is summed over threads, `sites` counts barrier INSTANCES -- divide before comparing, or
        // the difference is off by exactly nth. (This harness made that mistake once: kern read -12096 for MUL,
        // whose kernel has no barrier at all.)
        const uint64_t nodes = nodes_by_op[s] / (uint64_t) (ggml_cpu_bar_nth > 0 ? ggml_cpu_bar_nth : 1);
        fprintf(stderr, "[bar-spin-by-op] %-22s %14llu %8.2f%% %16llu %12llu %10llu %8llu %8lld\n",
                name,
                (unsigned long long) spin_by_op[s],
                total > 0.0 ? 100.0 * (double) spin_by_op[s] / total : 0.0,
                (unsigned long long) cycles_by_op[s],
                (unsigned long long) waits_by_op[s],
                (unsigned long long) sites,
                (unsigned long long) nodes,
                (long long) ((int64_t) sites - (int64_t) nodes));   // barriers inside this op's kernel

        // Per-thread split of this op's spin, with the class it implies on the SAME line (verdict and evidence
        // may not live in two places). The reading:
        //   k = 1 -> exactly one thread never waited for this op: that thread did the work  => SERIAL kernel.
        //            The spin IS the recoverable amount for that class: with a serial node taking T, the
        //            waiting is (nth-1)*T and removing it saves T - T/nth = spin/nth, which after normalising
        //            by the span is the same expression as this op's spin share.
        //   k > 1 -> several threads idle: the node was fanned out wider than it had rows => OVERSPLIT.
        //   k = 0 -> everybody waited: balanced, or a static split with a tail, or work stealing. Those three
        //            are NOT separable from one run -- cross-round reproducibility separates them (a static
        //            tail repeats to the row, a stolen tail does not).
        // NOTE: "spin anti-correlates with one's own time" is spin_i = max_j(t_j) - t_i, an IDENTITY, so it
        // can never be used as a criterion. Hence k and reproducibility, not correlation.
        {
            const int nt = ggml_cpu_bar_nth > 0 ? ggml_cpu_bar_nth : 1;
            int k = 0;
            uint64_t tmin = ~0ULL, tmax = 0;
            for (int t = 0; t < nt; t++) {
                const uint64_t v = ggml_cpu_bar_spin_by_op[t][s];
                if (v == 0) { k++; }
                if (v < tmin) { tmin = v; }
                if (v > tmax) { tmax = v; }
            }
            const char * cls = (k == 1) ? "SERIAL" : (k > 1 ? "OVERSPLIT" : "-");
            fprintf(stderr, "[bar-spin-by-op]   threads k=%d %-9s min=%llu max=%llu [",
                    k, cls, (unsigned long long) tmin, (unsigned long long) tmax);
            for (int t = 0; t < nt; t++) {
                fprintf(stderr, "%s%llu", t ? " " : "", (unsigned long long) ggml_cpu_bar_spin_by_op[t][s]);
            }
            fprintf(stderr, "]\n");
        }
    }

    // Did the F32->vec_dot repack run? It decides whether the pre-GEMM barrier waits on a memory copy or just
    // on entry spread -- the question section 12.1 could not settle from the source. It RUNS (1056 of 1064).
    // And the second line prices the repack itself, which is WORK: two published estimates of its cost differ
    // by an order of magnitude (this project's own benchgemm A/B bounded it at ~2%; a traffic count puts it
    // higher), so this number replaces both.
    {
        uint64_t rp_elems = 0, rp_cyc = 0;
        for (int t = 0; t < GGML_MAX_N_THREADS; t++) {
            rp_elems += ggml_cpu_bar_repack_elems [t];
            rp_cyc   += ggml_cpu_bar_repack_cycles[t];
        }
        const double mb_in  = 4.0 * (double) rp_elems / (1024.0*1024.0);
        const double mb_out = 2.0 * (double) rp_elems / (1024.0*1024.0);
        fprintf(stderr, "[bar-spin-by-op] MUL_MAT F32->vec_dot repack: done=%llu skipped=%llu  "
                        "elems=%llu (read %.1f MB + write %.1f MB = %.1f MB)  "
                        "cycles=%llu = %.3f%% of thread span",
                (unsigned long long) ggml_cpu_mm_repack_done,
                (unsigned long long) ggml_cpu_mm_repack_skip,
                (unsigned long long) rp_elems, mb_in, mb_out, mb_in + mb_out,
                (unsigned long long) rp_cyc,
                ggml_cpu_bar_span_cycles ? 100.0 * (double) rp_cyc / (double) ggml_cpu_bar_span_cycles : 0.0);
        if (rp_cyc > 0) {
            fprintf(stderr, "  (%.2f elems/cycle, %.2f GB/s read+write at 4 GHz)\n",
                    (double) rp_elems / (double) rp_cyc,
                    ((mb_in + mb_out) / 1024.0) / ((double) rp_cyc / 4.0e9));
        } else {
            fprintf(stderr, "\n");
        }
        if (ggml_cpu_mm_repack_f16_done) {
            // Lead B (KATAGO_CPU_MM_REPACK_F16): the materialisation copy, counted APART from the converting
            // repack because they are different work. Its element count is what proves the copy ran at all
            // -- the same reason gate 5 exists, since an arm that does nothing is also an arm that is
            // bit-identical.
            const double mb = 2.0 * (double) ggml_cpu_mm_repack_f16_elems / (1024.0*1024.0);
            fprintf(stderr, "[bar-spin-by-op] MUL_MAT f16->f16 MATERIALISE copy: done=%llu elems=%llu "
                            "(read %.1f MB + write %.1f MB = %.1f MB, no conversion)\n",
                    (unsigned long long) ggml_cpu_mm_repack_f16_done,
                    (unsigned long long) ggml_cpu_mm_repack_f16_elems, mb, mb, mb + mb);
        }
    }

    // Producer-side store cost (KATAGO_CPU_MUL_STORE_STATS=1), defined in binary-ops.cpp. Both contiguous
    // store loops are timed in ONE process -- the retype only touches some MUL nodes, so the rest keep an
    // f32 destination -- which makes P a same-process difference: no cross-arm, no cross-session and no
    // cross-DLL comparison appears anywhere in it. That matters because the cross-DLL spread on this arm
    // already measured 1.5 pp.
    katago_mul_store_report();

    // Activation-range pre-flight (only when KATAGO_CPU_ACT_RANGE=1). This is the measurement that decides
    // whether "activations in f16" is mechanically possible at all, and it is also the only thing that could
    // make bf16 interesting: bf16 trades 3 mantissa bits for f32's exponent range.
    {
        uint64_t a_n = 0, a_over = 0, a_den = 0;
        float    a_max = 0.0f;
        for (int t = 0; t < GGML_MAX_N_THREADS; t++) {
            a_n    += ggml_cpu_bar_act_n     [t];
            a_over += ggml_cpu_bar_act_over  [t];
            a_den  += ggml_cpu_bar_act_denorm[t];
            if (ggml_cpu_bar_act_maxabs[t] > a_max) { a_max = ggml_cpu_bar_act_maxabs[t]; }
        }
        if (a_n > 0) {
            fprintf(stderr, "[bar-spin-by-op] ACTIVATION RANGE (KATAGO_CPU_ACT_RANGE): max|x| = %.6g  "
                            "elements=%llu  over f16 max (|x|>65504) = %llu  f16-denormal (0<|x|<6.1e-5) = %llu\n",
                    (double) a_max, (unsigned long long) a_n,
                    (unsigned long long) a_over, (unsigned long long) a_den);
            fprintf(stderr, "[bar-spin-by-op]   => storing the activations as f16 is %s (%.4f%% of values would "
                            "saturate to inf, %.4f%% would lose bits to the denormal range). bf16 would fix the "
                            "range but costs 3 mantissa bits.\n",
                    (a_over == 0) ? "mechanically SAFE on range" : "*** BLOCKED: values exceed f16's range ***",
                    a_n ? 100.0 * (double) a_over / (double) a_n : 0.0,
                    a_n ? 100.0 * (double) a_den  / (double) a_n : 0.0);
            // WHICH producer op the denormal values come from: 0.35% of 283M is a lot of values but only a
            // handful of sites, and if the margin fails this is where to look first.
            for (int op = 0; op < GGML_OP_COUNT + 1; op++) {
                if (ggml_cpu_act_denorm_by_op[op] == 0) { continue; }
                fprintf(stderr, "[bar-spin-by-op]   f16-denormal values produced by %-18s : %12llu  (%.4f%% of all)\n",
                        (op == GGML_OP_COUNT) ? "(leaf/input)" : ggml_op_name((enum ggml_op) op),
                        (unsigned long long) ggml_cpu_act_denorm_by_op[op],
                        a_n ? 100.0 * (double) ggml_cpu_act_denorm_by_op[op] / (double) a_n : 0.0);
            }
        }
    }

    // ---- site table: WHICH barrier, not which node. This is the key that reaches into the kernels. ----
    uint64_t site_spin_total = 0, site_wait_total = 0;
    for (int s = 0; s < GGML_CPU_BAR_SITE_COUNT; s++) {
        site_spin_total += site_spin[s];
        site_wait_total += site_waits[s];
    }
    const int64_t site_spin_resid = (int64_t) ggml_cpu_n_spin_iters - (int64_t) site_spin_total;
    const int64_t site_wait_resid = (int64_t) ggml_cpu_n_barriers   - (int64_t) site_wait_total;

    fprintf(stderr, "[bar-spin-by-site] sum_spin=%llu (residual=%lld) sum_waits=%llu (residual=%lld)\n",
            (unsigned long long) site_spin_total, (long long) site_spin_resid,
            (unsigned long long) site_wait_total, (long long) site_wait_resid);
    if (site_spin_resid != 0 || site_wait_resid != 0) {
        fprintf(stderr, "[bar-spin-by-site] INVARIANT FAILED -- no site table printed\n");
    } else {
        fprintf(stderr, "[bar-spin-by-site] %-16s %14s %9s %12s %10s %10s\n",
                "site", "spin", "spin%", "waits", "sites", "cyc/iter");
        int sorder[GGML_CPU_BAR_SITE_COUNT];
        for (int s = 0; s < GGML_CPU_BAR_SITE_COUNT; s++) { sorder[s] = s; }
        for (int i = 1; i < GGML_CPU_BAR_SITE_COUNT; i++) {
            const int key = sorder[i];
            int j = i - 1;
            while (j >= 0 && site_spin[sorder[j]] < site_spin[key]) { sorder[j + 1] = sorder[j]; j--; }
            sorder[j + 1] = key;
        }
        for (int i = 0; i < GGML_CPU_BAR_SITE_COUNT; i++) {
            const int s = sorder[i];
            if (site_spin[s] == 0 && site_waits[s] == 0) { continue; }
            fprintf(stderr, "[bar-spin-by-site] %-16s %14llu %8.2f%% %12llu %10llu %10.1f\n",
                    ggml_cpu_bar_site_name[s],
                    (unsigned long long) site_spin[s],
                    site_spin_total > 0 ? 100.0 * (double) site_spin[s] / (double) site_spin_total : 0.0,
                    (unsigned long long) site_waits[s],
                    (unsigned long long) (site_waits[s] / (uint64_t) n_waiters),
                    site_spin[s] > 0 ? (double) site_cycles[s] / (double) site_spin[s] : 0.0);
        }
    }
    fprintf(stderr, "\n");

    // The one number this exercise exists for. The gate list is the pre-registered class; ADD1 is reported
    // next to it because a bias add is element-wise too but was not in the pre-registered definition.
    static const enum ggml_op ew_ops[] = { GGML_OP_MUL, GGML_OP_ADD, GGML_OP_SUB, GGML_OP_DIV, GGML_OP_SCALE };
    uint64_t ew_spin = 0;
    uint64_t ew_cycles = 0;
    for (size_t i = 0; i < sizeof(ew_ops) / sizeof(ew_ops[0]); i++) {
        ew_spin   += spin_by_op  [ew_ops[i]];
        ew_cycles += cycles_by_op[ew_ops[i]];
    }
    const uint64_t ew_spin_w_add1 = ew_spin + spin_by_op[GGML_OP_ADD1];

    uint64_t total_cycles = 0;
    for (int s = 0; s < GGML_CPU_BAR_OP_SLOTS; s++) {
        total_cycles += cycles_by_op[s];
    }

    // MEASURED conversion: spin cycles over the cycles the threads actually spent inside the node loop. Both
    // halves come from the same clock in the same process, so this is a share of runtime and not a projection.
    const double spin_share = (ggml_cpu_bar_span_cycles > 0)
                            ? 100.0 * (double) total_cycles / (double) ggml_cpu_bar_span_cycles : 0.0;
    const double cyc_per_iter = (ggml_cpu_n_spin_iters > 0)
                            ? (double) total_cycles / (double) ggml_cpu_n_spin_iters : 0.0;

    const double ew_spin_pct = total > 0.0 ? 100.0 * (double) ew_spin / total : 0.0;
    const double ew_add1_pct = total > 0.0 ? 100.0 * (double) ew_spin_w_add1 / total : 0.0;
    const double ew_measured = 100.0 * (double) ew_cycles / (double) (ggml_cpu_bar_span_cycles > 0
                                                                     ? ggml_cpu_bar_span_cycles : 1);
    const double rt_lb  = ew_spin_pct * GGML_CPU_BAR_SHARE_LB  / 100.0;
    const double rt_mid = ew_spin_pct * GGML_CPU_BAR_SHARE_MID / 100.0;
    const double rt_ub  = ew_spin_pct * GGML_CPU_BAR_SHARE_UB  / 100.0;

    fprintf(stderr, "[bar-spin-by-op] MEASURED total_spin_cycles=%llu thread_span_cycles=%llu "
                    "=> spin is %.3f%% of thread time (%.1f cycles/spin-iter)\n",
            (unsigned long long) total_cycles, (unsigned long long) ggml_cpu_bar_span_cycles,
            spin_share, cyc_per_iter);
    fprintf(stderr, "[bar-spin-by-op] ELEMENTWISE(MUL/ADD/SUB/DIV/SCALE) spin_pct=%.2f%%  ADD1_incl=%.2f%%\n",
            ew_spin_pct, ew_add1_pct);
    fprintf(stderr, "[bar-spin-by-op]   => MEASURED %.2f%% of runtime | cross-checks vs external shares: "
                    "%.2f%% (LB 15.7) / %.2f%% (identity 26) / %.2f%% (withdrawn 31)\n",
            ew_measured, rt_lb, rt_mid, rt_ub);

    // Each branch tests and prints the SAME variable (G-13): a verdict can never disagree with its number.
    if (ew_measured >= GGML_CPU_BAR_VERDICT_GATE) {
        fprintf(stderr, "[bar-spin-by-op] VERDICT: PROJECT -- measured %.2f%% of runtime clears the %.2f%% gate\n",
                ew_measured, GGML_CPU_BAR_VERDICT_GATE);
    } else {
        fprintf(stderr, "[bar-spin-by-op] VERDICT: OUT -- measured %.2f%% of runtime misses the %.2f%% gate; "
                        "the anchor table is the whole story\n", ew_measured, GGML_CPU_BAR_VERDICT_GATE);
    }
}

static void ggml_cpu_barrier_stats_report(void) {
    if (!ggml_cpu_barrier_stats_on) {
        return;
    }
    fprintf(stderr, "[barrier-stats] waits=%llu spin_iters=%llu\n",
            (unsigned long long) ggml_cpu_n_barriers,
            (unsigned long long) ggml_cpu_n_spin_iters);
    fprintf(stderr, "[barrier-stats] nodes_seen=%llu empty=%llu no_compute=%llu computed=%llu barrier_sites=%llu\n",
            (unsigned long long) ggml_cpu_n_nodes_seen,
            (unsigned long long) ggml_cpu_n_nodes_empty,
            (unsigned long long) ggml_cpu_n_nodes_nocomp,
            (unsigned long long) ggml_cpu_n_nodes_compute,
            (unsigned long long) ggml_cpu_n_nodes_barr);
    // A1 counters, printed SEPARATELY on purpose: hit (predicate matched), refuse (matched then guarded),
    // execute (kernel actually ran). Conflating these cost this project a whole debugging round once.
    fprintf(stderr, "[t9-stats] hit=%llu refuse=%llu execute=%llu epilogue_chunks=%llu "
                    "path_chunk=%llu path_early=%llu path_fallback=%llu\n",
            (unsigned long long) ggml_cpu_t9_hit,
            (unsigned long long) ggml_cpu_t9_refuse,
            (unsigned long long) ggml_cpu_t9_execute,
            (unsigned long long) ggml_cpu_t9_epilogue_chunks,
            (unsigned long long) ggml_cpu_t9_path_chunk,
            (unsigned long long) ggml_cpu_t9_path_early,
            (unsigned long long) ggml_cpu_t9_path_fallback);
    ggml_cpu_bar_spin_report();
}

static bool ggml_cpu_barrier_stats_enabled(void) {
    if (ggml_cpu_barrier_stats_env < 0) {
        const char * e = getenv("KATAGO_CPU_BARRIER_STATS");
        ggml_cpu_barrier_stats_env = (e != NULL && e[0] == '1') ? 1 : 0;
        ggml_cpu_barrier_stats_on  = ggml_cpu_barrier_stats_env;
        if (ggml_cpu_barrier_stats_on) {
            // Register ONCE. All threads of the pool reach this function on the first barrier of the first
            // graph, and the guard above is not atomic, so several of them used to register the same handler
            // and the report printed six times with identical numbers -- harmless, but it looks exactly like
            // "six measurements were taken".
            static int registered = 0;
            if (__atomic_exchange_n(&registered, 1, __ATOMIC_RELAXED) == 0) {
                atexit(ggml_cpu_barrier_stats_report);
            }
        }
    }
    return ggml_cpu_barrier_stats_on != 0;
}

void ggml_barrier(struct ggml_threadpool * tp) {
    int n_threads = atomic_load_explicit(&tp->n_graph, memory_order_relaxed) & GGML_THREADPOOL_N_THREADS_MASK;
    if (n_threads == 1) {
        return;
    }

#ifdef GGML_USE_OPENMP
    #pragma omp barrier
#else
    const bool stats = ggml_cpu_barrier_stats_enabled();
    int n_passed = atomic_load_explicit(&tp->n_barrier_passed, memory_order_relaxed);

    // enter barrier (full seq-cst fence)
    int n_barrier = atomic_fetch_add_explicit(&tp->n_barrier, 1, memory_order_seq_cst);

    if (n_barrier == (n_threads - 1)) {
        // last thread
        atomic_store_explicit(&tp->n_barrier, 0, memory_order_relaxed);

        // exit barrier (full seq-cst fence)
        atomic_fetch_add_explicit(&tp->n_barrier_passed, 1, memory_order_seq_cst);
        return;
    }

    // wait for other threads
    // Backoff, opt-in via KATAGO_CPU_BARRIER_BACKOFF (default off = stock behaviour). The wait polls
    // n_barrier_passed, and all T-1 waiting threads poll the SAME cache line, so every check pulls that line
    // across the interconnect -- measured at ~2.3 us per wait, which is far more than the ~800 relax
    // iterations would cost locally. Backing off means touching the line less often. It cannot change the
    // result: the loop still exits only when the flag actually changes, and the backoff is capped at a few
    // hundred cycles so it cannot add meaningful latency when the last thread arrives.
    if (stats) {
        ggml_cpu_tls_waits++;
        ggml_cpu_tls_spin_iters = 0;
        const uint64_t spin_t0 = ggml_cpu_bar_tsc();
        while (atomic_load_explicit(&tp->n_barrier_passed, memory_order_relaxed) == n_passed) {
            ggml_cpu_tls_spin_iters++;
            ggml_thread_cpu_relax();
        }
        const uint64_t spin_t1 = ggml_cpu_bar_tsc();
        ggml_cpu_tls_pending_cycles += spin_t1 - spin_t0;

        // Bank this wait for per-op attribution as well. It stays in TLS (thread-private) and is only written
        // into the per-op table when the node loop drains it, so this adds no shared traffic -- the two
        // shared atomics below are the only cross-thread cost, exactly as before.
        ggml_cpu_tls_pending_spin  += ggml_cpu_tls_spin_iters;
        ggml_cpu_tls_pending_waits += ggml_cpu_tls_waits;

        // Site attribution: THIS barrier's own spin, not the pending accumulator. Pending answers "which node
        // was running"; this answers "which barrier was being waited at". Two keys, both invariants checked.
        {
            const int site = ((unsigned) ggml_cpu_bar_site_cur < (unsigned) GGML_CPU_BAR_SITE_COUNT)
                           ? ggml_cpu_bar_site_cur : GGML_CPU_BAR_SITE_OTHER;
            const int t    = ((unsigned) ggml_cpu_tls_bar_ith < (unsigned) GGML_MAX_N_THREADS)
                           ? ggml_cpu_tls_bar_ith : 0;
            ggml_cpu_bar_site_spin  [t][site] += ggml_cpu_tls_spin_iters;
            ggml_cpu_bar_site_waits [t][site] += 1;
            ggml_cpu_bar_site_cycles[t][site] += spin_t1 - spin_t0;
        }

        // one shared update per barrier, never one per iteration and never two per barrier
        __atomic_fetch_add(&ggml_cpu_n_barriers,   ggml_cpu_tls_waits,      __ATOMIC_RELAXED);
        __atomic_fetch_add(&ggml_cpu_n_spin_iters, ggml_cpu_tls_spin_iters, __ATOMIC_RELAXED);
        ggml_cpu_tls_waits = 0;
    } else if (ggml_cpu_barrier_backoff) {
        int backoff = 4;
        while (atomic_load_explicit(&tp->n_barrier_passed, memory_order_relaxed) == n_passed) {
            for (int i = 0; i < backoff; i++) {
                ggml_thread_cpu_relax();
            }
            if (backoff < ggml_cpu_barrier_backoff) {
                backoff += backoff;
            }
        }
    } else {
        while (atomic_load_explicit(&tp->n_barrier_passed, memory_order_relaxed) == n_passed) {
            ggml_thread_cpu_relax();
        }
    }

    // exit barrier (full seq-cst fence)
    // TSAN doesn't support standalone fence yet, we use a dummy read-modify-write instead
    #ifdef GGML_TSAN_ENABLED
    atomic_fetch_add_explicit(&tp->n_barrier_passed, 0, memory_order_seq_cst);
    #else
    atomic_thread_fence(memory_order_seq_cst);
    #endif
#endif
}

void ggml_threadpool_chunk_set(struct ggml_threadpool * tp, int value) {
    atomic_store_explicit(&tp->current_chunk, value, memory_order_relaxed);
}

int ggml_threadpool_chunk_add(struct ggml_threadpool * tp, int value) {
    return atomic_fetch_add_explicit(&tp->current_chunk, value, memory_order_relaxed);
}

#if defined(__gnu_linux__)
static cpu_set_t ggml_get_numa_affinity(void) {
    cpu_set_t cpuset;
    pthread_t thread;
    thread = pthread_self();
    CPU_ZERO(&cpuset);
    pthread_getaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
    return cpuset;
}
#else
static uint32_t ggml_get_numa_affinity(void) {
    return 0; // no NUMA support
}
#endif

void ggml_numa_init(enum ggml_numa_strategy numa_flag) {
    if (g_state.numa.n_nodes > 0) {
        fprintf(stderr, "ggml_numa_init: NUMA already initialized\n");

        return;
    }

#if defined(__gnu_linux__)
    struct stat st;
    char path[256];
    int rv;

    // set numa scheme
    g_state.numa.numa_strategy = numa_flag;

    GGML_PRINT_DEBUG("numa strategy %u\n",g_state.numa.numa_strategy);

    g_state.numa.cpuset = ggml_get_numa_affinity();

    // enumerate nodes
    while (g_state.numa.n_nodes < GGML_NUMA_MAX_NODES) {
        rv = snprintf(path, sizeof(path), "/sys/devices/system/node/node%u", g_state.numa.n_nodes);
        GGML_ASSERT(rv > 0 && (unsigned)rv < sizeof(path));
        if (stat(path, &st) != 0) { break; }
        ++g_state.numa.n_nodes;
    }

    // enumerate CPUs
    while (g_state.numa.total_cpus < GGML_NUMA_MAX_CPUS) {
        rv = snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%u", g_state.numa.total_cpus);
        GGML_ASSERT(rv > 0 && (unsigned)rv < sizeof(path));
        if (stat(path, &st) != 0) { break; }
        ++g_state.numa.total_cpus;
    }

    GGML_PRINT_DEBUG("found %u numa nodes, %u CPUs\n", g_state.numa.n_nodes, g_state.numa.total_cpus);

    // figure out which node we're on
    uint current_cpu;
    int getcpu_ret = 0;
#if __GLIBC__ > 2 || (__GLIBC__ == 2 && __GLIBC_MINOR__ > 33) || defined(__COSMOPOLITAN__)
    getcpu_ret = getcpu(&current_cpu, &g_state.numa.current_node);
#else
    // old glibc doesn't have a wrapper for this call. Fall back on direct syscall
#   if !defined(SYS_getcpu) && defined(SYS_get_cpu)
#       define SYS_getcpu SYS_get_cpu // some older glibc versions use this name
#   endif
    getcpu_ret = syscall(SYS_getcpu, &current_cpu, &g_state.numa.current_node);
#endif

    if (g_state.numa.n_nodes < 1 || g_state.numa.total_cpus < 1 || getcpu_ret != 0) {
        g_state.numa.n_nodes = 0;
        return;
    }

    GGML_PRINT_DEBUG("found our process on numa node %u, CPU %u\n", g_state.numa.current_node, current_cpu);

    for (uint32_t n = 0; n < g_state.numa.n_nodes; ++n) {
        struct ggml_numa_node * node = &g_state.numa.nodes[n];
        GGML_PRINT_DEBUG("CPUs on node %u:", n);
        node->n_cpus = 0;
        for (uint32_t c = 0; c < g_state.numa.total_cpus; ++c) {
            rv = snprintf(path, sizeof(path), "/sys/devices/system/node/node%u/cpu%u", n, c);
            GGML_ASSERT(rv > 0 && (unsigned)rv < sizeof(path));
            if (stat(path, &st) == 0) {
                node->cpus[node->n_cpus++] = c;
                GGML_PRINT_DEBUG(" %u", c);
            }
        }
        GGML_PRINT_DEBUG("\n");
    }

    if (ggml_is_numa()) {
        FILE *fptr = fopen("/proc/sys/kernel/numa_balancing", "r");
        if (fptr != NULL) {
            char buf[42];
            if (fgets(buf, sizeof(buf), fptr) && strncmp(buf, "0\n", sizeof(buf)) != 0) {
                GGML_LOG_WARN("/proc/sys/kernel/numa_balancing is enabled, this has been observed to impair performance\n");
            }
            fclose(fptr);
        }
    }
#else
    UNUSED(numa_flag);
    // TODO
#endif
}

bool ggml_is_numa(void) {
    return g_state.numa.n_nodes > 1;
}

#if defined(__ARM_ARCH)
#if defined(__aarch64__) && defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>
static void ggml_init_arm_arch_features(void) {
    ggml_arm_arch_features.sve_cnt = svcntb();
}
#else
static void ggml_init_arm_arch_features(void) {}
#endif
#endif // __ARM_ARCH

#if defined(__riscv) && defined(__riscv_v_intrinsic)
#include <riscv_vector.h>
static void ggml_init_riscv_arch_features(void) {
    ggml_riscv_arch_features.rvv_vlen = __riscv_vlenb();
}
#else
static void ggml_init_riscv_arch_features(void) {}
#endif

struct ggml_tensor * ggml_new_i32(struct ggml_context * ctx, int32_t value) {
    GGML_ASSERT(!ggml_get_no_alloc(ctx));

    struct ggml_tensor * result = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);

    ggml_set_i32(result, value);

    return result;
}

struct ggml_tensor * ggml_new_f32(struct ggml_context * ctx, float value) {
    GGML_ASSERT(!ggml_get_no_alloc(ctx));

    struct ggml_tensor * result = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);

    ggml_set_f32(result, value);

    return result;
}

struct ggml_tensor * ggml_set_i32 (struct ggml_tensor * tensor, int32_t value) {
    const int n     = ggml_nrows(tensor);
    const int nc    = tensor->ne[0];
    const size_t n1 = tensor->nb[1];

    char * const data = tensor->data;

    switch (tensor->type) {
        case GGML_TYPE_I8:
            {
                assert(tensor->nb[0] == sizeof(int8_t));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_i8(nc, (int8_t *)(data + i*n1), value);
                }
            } break;
        case GGML_TYPE_I16:
            {
                assert(tensor->nb[0] == sizeof(int16_t));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_i16(nc, (int16_t *)(data + i*n1), value);
                }
            } break;
        case GGML_TYPE_I32:
            {
                assert(tensor->nb[0] == sizeof(int32_t));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_i32(nc, (int32_t *)(data + i*n1), value);
                }
            } break;
        case GGML_TYPE_F16:
            {
                assert(tensor->nb[0] == sizeof(ggml_fp16_t));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_f16(nc, (ggml_fp16_t *)(data + i*n1), GGML_CPU_FP32_TO_FP16(value));
                }
            } break;
        case GGML_TYPE_BF16:
            {
                assert(tensor->nb[0] == sizeof(ggml_fp16_t));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_bf16(nc, (ggml_bf16_t *)(data + i*n1), GGML_FP32_TO_BF16(value));
                }
            } break;
        case GGML_TYPE_F32:
            {
                assert(tensor->nb[0] == sizeof(float));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_f32(nc, (float *)(data + i*n1), value);
                }
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }

    return tensor;
}

struct ggml_tensor * ggml_set_f32(struct ggml_tensor * tensor, float value) {
    const int n     = ggml_nrows(tensor);
    const int nc    = tensor->ne[0];
    const size_t n1 = tensor->nb[1];

    char * const data = tensor->data;

    switch (tensor->type) {
        case GGML_TYPE_I8:
            {
                assert(tensor->nb[0] == sizeof(int8_t));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_i8(nc, (int8_t *)(data + i*n1), value);
                }
            } break;
        case GGML_TYPE_I16:
            {
                assert(tensor->nb[0] == sizeof(int16_t));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_i16(nc, (int16_t *)(data + i*n1), value);
                }
            } break;
        case GGML_TYPE_I32:
            {
                assert(tensor->nb[0] == sizeof(int32_t));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_i32(nc, (int32_t *)(data + i*n1), value);
                }
            } break;
        case GGML_TYPE_F16:
            {
                assert(tensor->nb[0] == sizeof(ggml_fp16_t));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_f16(nc, (ggml_fp16_t *)(data + i*n1), GGML_CPU_FP32_TO_FP16(value));
                }
            } break;
        case GGML_TYPE_BF16:
            {
                assert(tensor->nb[0] == sizeof(ggml_bf16_t));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_bf16(nc, (ggml_bf16_t *)(data + i*n1), GGML_FP32_TO_BF16(value));
                }
            } break;
        case GGML_TYPE_F32:
            {
                assert(tensor->nb[0] == sizeof(float));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_f32(nc, (float *)(data + i*n1), value);
                }
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }

    return tensor;
}

int32_t ggml_get_i32_1d(const struct ggml_tensor * tensor, int i) {
    if (!ggml_is_contiguous(tensor)) {
        int64_t id[4] = { 0, 0, 0, 0 };
        ggml_unravel_index(tensor, i, &id[0], &id[1], &id[2], &id[3]);
        return ggml_get_i32_nd(tensor, id[0], id[1], id[2], id[3]);
    }
    switch (tensor->type) {
        case GGML_TYPE_I8:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(int8_t));
                return ((int8_t *)(tensor->data))[i];
            }
        case GGML_TYPE_I16:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(int16_t));
                return ((int16_t *)(tensor->data))[i];
            }
        case GGML_TYPE_I32:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(int32_t));
                return ((int32_t *)(tensor->data))[i];
            }
        case GGML_TYPE_F16:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(ggml_fp16_t));
                return GGML_CPU_FP16_TO_FP32(((ggml_fp16_t *)(tensor->data))[i]);
            }
        case GGML_TYPE_BF16:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(ggml_bf16_t));
                return GGML_BF16_TO_FP32(((ggml_bf16_t *)(tensor->data))[i]);
            }
        case GGML_TYPE_F32:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(float));
                return ((float *)(tensor->data))[i];
            }
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

void ggml_set_i32_1d(const struct ggml_tensor * tensor, int i, int32_t value) {
    if (!ggml_is_contiguous(tensor)) {
        int64_t id[4] = { 0, 0, 0, 0 };
        ggml_unravel_index(tensor, i, &id[0], &id[1], &id[2], &id[3]);
        ggml_set_i32_nd(tensor, id[0], id[1], id[2], id[3], value);
        return;
    }
    switch (tensor->type) {
        case GGML_TYPE_I8:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(int8_t));
                ((int8_t *)(tensor->data))[i] = value;
            } break;
        case GGML_TYPE_I16:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(int16_t));
                ((int16_t *)(tensor->data))[i] = value;
            } break;
        case GGML_TYPE_I32:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(int32_t));
                ((int32_t *)(tensor->data))[i] = value;
            } break;
        case GGML_TYPE_F16:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(ggml_fp16_t));
                ((ggml_fp16_t *)(tensor->data))[i] = GGML_CPU_FP32_TO_FP16(value);
            } break;
        case GGML_TYPE_BF16:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(ggml_bf16_t));
                ((ggml_bf16_t *)(tensor->data))[i] = GGML_FP32_TO_BF16(value);
            } break;
        case GGML_TYPE_F32:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(float));
                ((float *)(tensor->data))[i] = value;
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

int32_t ggml_get_i32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3) {
    void * data   = (char *) tensor->data + i0*tensor->nb[0] + i1*tensor->nb[1] + i2*tensor->nb[2] + i3*tensor->nb[3];
    switch (tensor->type) {
        case GGML_TYPE_I8:
            return ((int8_t *) data)[0];
        case GGML_TYPE_I16:
            return ((int16_t *) data)[0];
        case GGML_TYPE_I32:
            return ((int32_t *) data)[0];
        case GGML_TYPE_F16:
            return GGML_CPU_FP16_TO_FP32(((ggml_fp16_t *) data)[0]);
        case GGML_TYPE_BF16:
            return GGML_BF16_TO_FP32(((ggml_bf16_t *) data)[0]);
        case GGML_TYPE_F32:
            return ((float *) data)[0];
        default:
            GGML_ABORT("fatal error");
    }
}

void ggml_set_i32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3, int32_t value) {
    void * data   = (char *) tensor->data + i0*tensor->nb[0] + i1*tensor->nb[1] + i2*tensor->nb[2] + i3*tensor->nb[3];
    switch (tensor->type) {
        case GGML_TYPE_I8:
            {
                ((int8_t *)(data))[0] = value;
            } break;
        case GGML_TYPE_I16:
            {
                ((int16_t *)(data))[0] = value;
            } break;
        case GGML_TYPE_I32:
            {
                ((int32_t *)(data))[0] = value;
            } break;
        case GGML_TYPE_F16:
            {
                ((ggml_fp16_t *)(data))[0] = GGML_CPU_FP32_TO_FP16(value);
            } break;
        case GGML_TYPE_BF16:
            {
                ((ggml_bf16_t *)(data))[0] = GGML_FP32_TO_BF16(value);
            } break;
        case GGML_TYPE_F32:
            {
                ((float *)(data))[0] = value;
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

float ggml_get_f32_1d(const struct ggml_tensor * tensor, int i) {
    if (!ggml_is_contiguous(tensor)) {
        int64_t id[4] = { 0, 0, 0, 0 };
        ggml_unravel_index(tensor, i, &id[0], &id[1], &id[2], &id[3]);
        return ggml_get_f32_nd(tensor, id[0], id[1], id[2], id[3]);
    }
    switch (tensor->type) {
        case GGML_TYPE_I8:
            {
                return ((int8_t *)(tensor->data))[i];
            }
        case GGML_TYPE_I16:
            {
                return ((int16_t *)(tensor->data))[i];
            }
        case GGML_TYPE_I32:
            {
                return ((int32_t *)(tensor->data))[i];
            }
        case GGML_TYPE_F16:
            {
                return GGML_CPU_FP16_TO_FP32(((ggml_fp16_t *)(tensor->data))[i]);
            }
        case GGML_TYPE_BF16:
            {
                return GGML_BF16_TO_FP32(((ggml_bf16_t *)(tensor->data))[i]);
            }
        case GGML_TYPE_F32:
            {
                return ((float *)(tensor->data))[i];
            }
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

void ggml_set_f32_1d(const struct ggml_tensor * tensor, int i, float value) {
    if (!ggml_is_contiguous(tensor)) {
        int64_t id[4] = { 0, 0, 0, 0 };
        ggml_unravel_index(tensor, i, &id[0], &id[1], &id[2], &id[3]);
        ggml_set_f32_nd(tensor, id[0], id[1], id[2], id[3], value);
        return;
    }
    switch (tensor->type) {
        case GGML_TYPE_I8:
            {
                ((int8_t *)(tensor->data))[i] = value;
            } break;
        case GGML_TYPE_I16:
            {
                ((int16_t *)(tensor->data))[i] = value;
            } break;
        case GGML_TYPE_I32:
            {
                ((int32_t *)(tensor->data))[i] = value;
            } break;
        case GGML_TYPE_F16:
            {
                ((ggml_fp16_t *)(tensor->data))[i] = GGML_CPU_FP32_TO_FP16(value);
            } break;
        case GGML_TYPE_BF16:
            {
                ((ggml_bf16_t *)(tensor->data))[i] = GGML_FP32_TO_BF16(value);
            } break;
        case GGML_TYPE_F32:
            {
                ((float *)(tensor->data))[i] = value;
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

float ggml_get_f32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3) {
    void * data   = (char *) tensor->data + i0*tensor->nb[0] + i1*tensor->nb[1] + i2*tensor->nb[2] + i3*tensor->nb[3];
    switch (tensor->type) {
        case GGML_TYPE_I8:
            return ((int8_t *) data)[0];
        case GGML_TYPE_I16:
            return ((int16_t *) data)[0];
        case GGML_TYPE_I32:
            return ((int32_t *) data)[0];
        case GGML_TYPE_F16:
            return GGML_CPU_FP16_TO_FP32(((ggml_fp16_t *) data)[0]);
        case GGML_TYPE_BF16:
            return GGML_BF16_TO_FP32(((ggml_bf16_t *) data)[0]);
        case GGML_TYPE_F32:
            return ((float *) data)[0];
        default:
            GGML_ABORT("fatal error");
    }
}

void ggml_set_f32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3, float value) {
    void * data   = (char *) tensor->data + i0*tensor->nb[0] + i1*tensor->nb[1] + i2*tensor->nb[2] + i3*tensor->nb[3];
    switch (tensor->type) {
        case GGML_TYPE_I8:
            {
                ((int8_t *)(data))[0] = value;
            } break;
        case GGML_TYPE_I16:
            {
                ((int16_t *)(data))[0] = value;
            } break;
        case GGML_TYPE_I32:
            {
                ((int32_t *)(data))[0] = value;
            } break;
        case GGML_TYPE_F16:
            {
                ((ggml_fp16_t *)(data))[0] = GGML_CPU_FP32_TO_FP16(value);
            } break;
        case GGML_TYPE_BF16:
            {
                ((ggml_bf16_t *)(data))[0] = GGML_FP32_TO_BF16(value);
            } break;
        case GGML_TYPE_F32:
            {
                ((float *)(data))[0] = value;
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

////////////////////////////////////////////////////////////////////////////////

// ---- A1 fusion: the state the GEMM and the matcher share (the matcher itself is much further down) ---//
// A1 is {MUL_MAT, <free view>, ADD}. The fusion skips the free view and the ADD, but the matmul must still
// run, and the fused add-pass must be performed in the SAME barrier interval as that matmul -- otherwise
// the ADD's barrier is not saved and the fusion buys nothing.
//
// There are exactly two ways to be in that interval, and the measured graph uses both:
//
//   * ggml's own chunk loop (1 of 52 anchors): the pass runs right after each chunk, on the very rows that
//     chunk just wrote. The chunk->thread assignment is dynamic (a thread starts at chunk == ith and then
//     auto-claims), so reading any other thread's rows would need a barrier -- which is why the pass must
//     stay INSIDE the loop, with the same ir0/ir1 ranges.
//   * the llamafile sgemm (51 of 52 anchors): it owns the whole output and ends with a barrier ordered
//     after its writes, so once it returns the pass may use any partition and adds no barrier of its own.
//     See the handshake in llamafile/sgemm.cpp -- and note the caller CHECKS for that barrier rather than
//     assuming it, falling back to the stock chain if the instantiation that ran had none.
//
// A1-EPILOGUE-FUSION-STATUS.md has the measurements behind both counts.
struct ggml_cpu_t9_pending {
    const struct ggml_tensor * mm;      // the anchor MUL_MAT node
    const struct ggml_tensor * view;    // the free view between anchor and ADD (never dereferenced for data)
    const struct ggml_tensor * add;     // the ADD node, i.e. the write target
    const struct ggml_tensor * other;   // the ADD's non-view operand
    int64_t s_ne1, s_ne2, s_ne3;        // the view's dims 1..3: the reduction/broadcast shape
    int64_t nr_rows;                    // s_ne1*s_ne2*s_ne3, the row count the pass partitions over
    int     armed;
    int     pass_ran;                   // set once the fused pass has actually been performed
    int     diag[4];                    // trace only: src0 type, src1 type, vec_dot type, which early exit
};

// THREAD LOCAL ON PURPOSE: the matcher runs on every thread and each thread then calls the GEMM itself, so
// the hand-off never crosses threads -- set and consumed within one call, with no barrier in between. A
// shared static would be a race, and would leak across graphs if a refusal landed between set and call.
static _Thread_local struct ggml_cpu_t9_pending ggml_cpu_t9_pend;

// Defined in llamafile/sgemm.cpp. See the handshake comment there: tinyBLAS::gemm ends with a barrier that
// is ordered after all of its writes, and these two let the fused pass reuse it -- or refuse to run when
// the instantiation that served this matmul had no such barrier.
void ggml_cpu_tb_clear_writes_published(void);
int  ggml_cpu_tb_take_writes_published(void);

// Called by the GEMM for every MUL_MAT it computes. Returns non-zero only for the anchor of a pending
// match, which costs one thread-local load and a pointer compare on the hot path.
static int ggml_cpu_t9_anchor_pending(const struct ggml_tensor * dst) {
    return ggml_cpu_t9_pend.armed && ggml_cpu_t9_pend.mm == dst;
}

// The fused pass, over a range of the MUL_MAT's rows (which are the view's rows) and a range of the fast
// dimension. Two callers use two different partitions of the same work; see each call site.
static void ggml_cpu_t9_anchor_epilogue(const struct ggml_compute_params * params,
                                        const struct ggml_tensor * mm,
                                        int64_t r_begin, int64_t r_end,
                                        int64_t c_begin, int64_t c_end);

// Ran inside the GEMM's chunk loop (the thread reads only rows it just wrote).
static void ggml_cpu_t9_pass_in_chunk(const struct ggml_compute_params * params,
                                      const struct ggml_tensor * mm,
                                      int64_t ir0_start, int64_t ir0_end,
                                      int64_t ir1_start, int64_t ir1_end) {
    ggml_cpu_t9_anchor_epilogue(params, mm, ir1_start, ir1_end, ir0_start, ir0_end);
}

// Ran after a llamafile sgemm that ended with its own publish barrier. Every thread has passed that
// barrier, so the whole output is visible and ANY partition is safe here -- including this plain row split,
// which is cheaper to compute than the GEMM's dynamic chunking and adds no barrier.
static void ggml_cpu_t9_pass_after_sgemm(const struct ggml_compute_params * params,
                                         const struct ggml_tensor * mm) {
    // Redundant-barrier probe (KATAGO_CPU_T9_EXTRABAR=1, user's third candidate). The hypothesis it tests is
    // counterintuitive but cheap: removing the ADD node's barrier may have turned eight synchronised threads
    // into eight threads that all immediately read data their peers have just written, bouncing cache lines
    // between cores. If deliberately ADDING a barrier makes the fused arm FASTER, that is proven -- and it
    // would mean the barrier being saved was never the expensive part, the memory ownership was.
    {
        static int t9_extrabar = -1;
        if (t9_extrabar < 0) {
            t9_extrabar = (getenv("KATAGO_CPU_T9_EXTRABAR") != NULL) ? 1 : 0;
        }
        if (t9_extrabar) {
            ggml_barrier(params->threadpool);
        }
    }
    const int64_t nth = params->nth;
    const int64_t dr  = (ggml_cpu_t9_pend.nr_rows + nth - 1) / nth;
    const int64_t r0  = dr * params->ith;
    const int64_t r1  = MIN(r0 + dr, ggml_cpu_t9_pend.nr_rows);
    ggml_cpu_t9_anchor_epilogue(params, mm, r0, r1, 0, mm->ne[0]);
}

// ggml_compute_forward_mul_mat

static void ggml_compute_forward_mul_mat_one_chunk(
    const struct ggml_compute_params * params,
    struct ggml_tensor * dst,
    const enum ggml_type type,
    const int64_t num_rows_per_vec_dot,
    const int64_t ir0_start,
    const int64_t ir0_end,
    const int64_t ir1_start,
    const int64_t ir1_end) {

    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_BINARY_OP_LOCALS

    const bool src1_cont = ggml_is_contiguous(src1);

    ggml_vec_dot_t const vec_dot      = type_traits_cpu[type].vec_dot;
    enum ggml_type const vec_dot_type = type_traits_cpu[type].vec_dot_type;

    // broadcast factors
    const int64_t r2 = ne12 / ne02;
    const int64_t r3 = ne13 / ne03;

    //printf("ir0_start = %6lld, ir0_end = %6lld, ir1_start = %6lld, ir1_end = %6lld\n", ir0_start, ir0_end, ir1_start, ir1_end);

    // threads with no work simply yield (not sure if it helps)
    if (ir0_start >= ir0_end || ir1_start >= ir1_end) {
        return;
    }

    const void * wdata = (src1->type == vec_dot_type) ? src1->data : params->wdata;
    const size_t row_size = ggml_row_size(vec_dot_type, ne10);

    assert(ne12 % ne02 == 0);
    assert(ne13 % ne03 == 0);

    // block-tiling attempt
    const int64_t blck_0 = 16;
    const int64_t blck_1 = 16;

    const size_t src1_col_stride = src1_cont || src1->type != vec_dot_type ? row_size : nb11;

    // attempt to reduce false-sharing (does not seem to make a difference)
    // 16 * 2, accounting for mmla kernels
    float tmp[32];

    for (int64_t iir1 = ir1_start; iir1 < ir1_end; iir1 += blck_1) {
        for (int64_t iir0 = ir0_start; iir0 < ir0_end; iir0 += blck_0) {
            for (int64_t ir1 = iir1; ir1 < iir1 + blck_1 && ir1 < ir1_end; ir1 += num_rows_per_vec_dot) {
                const int64_t i13 = (ir1 / (ne12 * ne1));
                const int64_t i12 = (ir1 - i13 * ne12 * ne1) / ne1;
                const int64_t i11 = (ir1 - i13 * ne12 * ne1 - i12 * ne1);

                // broadcast src0 into src1
                const int64_t i03 = i13 / r3;
                const int64_t i02 = i12 / r2;

                const int64_t i1 = i11;
                const int64_t i2 = i12;
                const int64_t i3 = i13;

                const char * src0_row = (const char*)src0->data + (0 + i02 * nb02 + i03 * nb03);

                // desc: when src1 is not a contiguous memory block we have to calculate the offset using the strides
                //       if it is, then we have either copied the data to params->wdata and made it contiguous or we are using
                //       the original src1 data pointer, so we should index using the indices directly
                // TODO: this is a bit of a hack, we should probably have a better way to handle this
                const char * src1_col = (const char*)wdata +
                    (src1_cont || src1->type != vec_dot_type
                        ? (i11 + i12 * ne11 + i13 * ne12 * ne11) * row_size
                        : (i11 * nb11 + i12 * nb12 + i13 * nb13));
                float * dst_col = (float*)((char*)dst->data + (i1 * nb1 + i2 * nb2 + i3 * nb3));

                //for (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_end; ++ir0) {
                //    vec_dot(ne00, &dst_col[ir0], src0_row + ir0*nb01, src1_col);
                //}

                for (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_end; ir0 += num_rows_per_vec_dot) {
                    vec_dot(ne00, &tmp[ir0 - iir0], (num_rows_per_vec_dot > 1 ? 16 : 0), src0_row + ir0 * nb01, (num_rows_per_vec_dot > 1 ? nb01 : 0), src1_col, (num_rows_per_vec_dot > 1 ? src1_col_stride : 0), num_rows_per_vec_dot);
                }

                for (int cn = 0; cn < num_rows_per_vec_dot; ++cn) {
                    memcpy(&dst_col[iir0 + cn * nb1 / nb0], tmp + (cn * 16), (MIN(iir0 + blck_0, ir0_end) - iir0) * sizeof(float));
                }
            }
        }
    }
}

void ggml_compute_forward_mul_mat(
        const struct ggml_compute_params * params,
              struct ggml_tensor * dst) {

    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    // A1: is a fused add-pass pending for THIS matmul? If so the pass runs in the chunk loop below. The
    // GEMM paths here are NOT diverted to make room for it -- forcing a MUL_MAT off the llamafile sgemm
    // path changes its summation order and therefore its bits (p0-bitequal.ps1 relies on exactly that
    // difference for its negative control), which would show up as a network-level change even with a
    // perfectly correct pass. Instead the pass records whether it actually ran and the matcher falls back
    // to the stock chain when it did not. See t9-stats path_chunk/path_early.
    const int t9_pending = ggml_cpu_t9_anchor_pending(dst);

    const int32_t hint = ggml_get_op_params_i32(dst, 1);
    if (t9_pending) {
        // Clear before the matmul so a stale "published" note from an earlier sgemm on this thread can never
        // be mistaken for this call's. Trace fields are filled here too: which GEMM path an anchor takes
        // decides whether the fused pass can run at all, so it is recorded rather than inferred.
        ggml_cpu_tb_clear_writes_published();
        ggml_cpu_t9_pend.diag[0] = (int) src0->type;
        ggml_cpu_t9_pend.diag[1] = (int) src1->type;
        ggml_cpu_t9_pend.diag[2] = (int) type_traits_cpu[src0->type].vec_dot_type;
        ggml_cpu_t9_pend.diag[3] = 0;
    }
    if (hint == GGML_HINT_SRC0_IS_HADAMARD && !params->use_ref) {
        if (t9_pending) { ggml_cpu_t9_pend.diag[3] = 1; }
        ggml_compute_forward_fwht(params, dst);
        return;
    }

    GGML_TENSOR_BINARY_OP_LOCALS

    const int ith = params->ith;
    const int nth = params->nth;

    enum ggml_type           const vec_dot_type         = type_traits_cpu[src0->type].vec_dot_type;
    ggml_from_float_t        const from_float           = type_traits_cpu[vec_dot_type].from_float;
    int64_t                  const vec_dot_num_rows     = type_traits_cpu[src0->type].nrows;

    GGML_ASSERT(ne0 == ne01);
    GGML_ASSERT(ne1 == ne11);
    GGML_ASSERT(ne2 == ne12);
    GGML_ASSERT(ne3 == ne13);

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_size(src0->type));
    GGML_ASSERT(nb10 == ggml_type_size(src1->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == sizeof(float));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    // nb01 >= nb00 - src0 is not transposed
    //   compute by src0 rows

    // TODO: extract to "extra_op"
#if GGML_USE_LLAMAFILE
    // broadcast factors
    const int64_t r2 = ne12 / ne02;
    const int64_t r3 = ne13 / ne03;

    const bool src1_cont = ggml_is_contiguous(src1);

    // Lead B forces the materialisation, and that means site 1 must NOT take its shortcut: otherwise the
    // copy below never runs, the arm is bit-identical to the retype arm while claiming to test the
    // opposite hypothesis, and the A/B measures nothing (the exact failure that the first MUL->F16 landing
    // shipped: 0 of 256 tensors retyped, both arms hashed the same, and only the counter noticed).
    const bool force_mat = (src1->type == GGML_TYPE_F16 && vec_dot_type == GGML_TYPE_F16 &&
                            ggml_cpu_mm_repack_f16() != 0);

    if (src1_cont && !force_mat) {
        for (int64_t i13 = 0; i13 < ne13; i13++)
            for (int64_t i12 = 0; i12 < ne12; i12++)
                if (!llamafile_sgemm(params,
                                     ne01, ne11, ne00/ggml_blck_size(src0->type),
                                     (const char *)src0->data + i12/r2*nb02 + i13/r3*nb03,
                                     nb01/ggml_type_size(src0->type),
                                     (const char *)src1->data + i12*nb12 + i13*nb13,
                                     nb11/ggml_type_size(src1->type),
                                     (char *)dst->data + i12*nb2 + i13*nb3,
                                     nb1/ggml_type_size(dst->type),
                                     src0->type,
                                     src1->type,
                                     dst->type))
                    goto UseGgmlGemm1;
        // A1: the sgemm served this matmul and left a barrier after its writes; fold the ADD in here.
        if (t9_pending) {
            ggml_cpu_t9_pend.diag[3] = 2;
            if (ggml_cpu_tb_take_writes_published()) {
                ggml_cpu_t9_pass_after_sgemm(params, dst);
                ggml_cpu_t9_pend.pass_ran = 1;
            }
        }
        return;
    }
UseGgmlGemm1:;
#endif

    // did_copy: lead B's switch (ggml_cpu_mm_repack_f16()). src1 is f16 and already in vec_dot form, so the
    // block below copies rows into wdata instead of converting them -- and the second sgemm site further
    // down is opened up so that the copy is actually READ. Copying without that second change would be
    // pure added work: the GEMM would still read src1->data, or fall through to the generic chunk kernel.
    const bool did_copy = force_mat;
    if (src1->type != vec_dot_type || did_copy) {
        // Does the F32->vec_dot repack actually run on this model? That decides what the barrier ~40 lines below
        // waits for: the repack (a memory-copy phase) or merely entry/dispatch spread. It DOES run: measured
        // done=1056 skipped=8 per process on b10c384.
        //
        // And the wait is only the visible half. The repack ITSELF is a full read of src1 + write of src1',
        // and its cost sits in MUL_MAT's own work -- invisible to a waiting-type instrument. The counter below
        // measures it directly. Two other readings are on record and they DISAGREE, so this number is not a
        // confirmation of either:
        //   * benchgemm's f16f32 -> f16f16 A/B (which does time the whole mul_mat, repack included) came out at
        //     a median of -0.35% over twelve shapes with the sign split 5/5 -- i.e. about zero;
        //   * a traffic estimate from the byte count (300-400 MB per walk) put it at 5-9%, but that used DRAM
        //     bandwidth for bytes that are cache-hot: 202 MB per walk against ~1 ms of wall is an aggregate of
        //     about 200 GB/s, which no DRAM delivers (G-18).
        // A correct ablation (zero-fill instead of convert, finite operands, work-equal, paired) decides which
        // is right. It is NOT built yet: the first attempt skipped the write entirely, produced non-finite
        // activations, and made KataGo stop printing raw-nn after 1.3 KB instead of 58.6 KB -- see G-19.
        // `done` counts CONVERTING repacks only. In lead B's arm the block is also entered for a copy, and
        // counting those here would make done=1064 while done-copy_done=384 is the number gate 4 declares.
        if (ith == 0) {
            // The OFF arm's ACTUAL source: the scratch every GEMM reads instead of its src1. One fixed
            // buffer, so one page offset applies to all of them -- which is exactly what makes the
            // 4K-aliasing comparison between the two arms possible at all (KATAGO_CPU_MMALIGN_DBG=1).
            static int wd_off_printed = 0;
            if (!wd_off_printed && getenv("KATAGO_CPU_MMALIGN_DBG") != NULL) {
                wd_off_printed = 1;
                fprintf(stderr, "[mm-align] OFF-arm source: params->wdata%%4096=%llu  wdata%%64=%llu  "
                                "(one fixed scratch shared by every GEMM)\n",
                        (unsigned long long) ((uintptr_t) params->wdata % 4096),
                        (unsigned long long) ((uintptr_t) params->wdata % 64));
            }
        }
        if (ith == 0 && !did_copy) { __atomic_fetch_add(&ggml_cpu_mm_repack_done, 1, __ATOMIC_RELAXED); }
        char * wdata = params->wdata;
        const uint64_t repack_t0 = ggml_cpu_bar_tsc();
        uint64_t f16_elems_local = 0;   // lead B's copied elements, added to the shared counter once per node

        const size_t nbw0 = ggml_type_size(vec_dot_type);
        const size_t nbw1 = ggml_row_size(vec_dot_type, ne10);
        const size_t nbw2 = nbw1*ne11;
        const size_t nbw3 = nbw2*ne12;

        assert(params->wsize >= ne13*nbw3);
        // The general repack converts F32. Lead B's variant copies f16 rows as they are; anything else (a
        // quantised or bf16 activation) has no path here and must not pretend to have one.
        GGML_ASSERT(src1->type == GGML_TYPE_F32 || (did_copy && src1->type == GGML_TYPE_F16));

    #if 0
        for (int64_t i13 = 0; i13 < ne13; ++i13) {
            for (int64_t i12 = 0; i12 < ne12; ++i12) {
                for (int64_t i11 = ith; i11 < ne11; i11 += nth) {
                    from_float((float *)((char *) src1->data + i13*nb13 + i12*nb12 + i11*nb11),
                               (void *)               (wdata + i13*nbw3 + i12*nbw2 + i11*nbw1),
                                ne10);
                }
            }
        }
    #else
        for (int64_t i13 = 0; i13 < ne13; ++i13) {
            for (int64_t i12 = 0; i12 < ne12; ++i12) {
                for (int64_t i11 = 0; i11 < ne11; ++i11) {
                    size_t bs = ggml_blck_size(vec_dot_type);
                    int64_t ne10_block_start = (ith * ne10/bs) / nth;
                    int64_t ne10_block_end   = ((ith + 1) * ne10/bs) / nth;
                        if (!did_copy) {
                            // Only the CONVERTING repack counts as "repack elems": lead B's copy is
                            // different work, and mixing it in would make the one volume that gate 4
                            // declares unreadable.
                            ggml_cpu_tls_repack_elems += (uint64_t) (ne10_block_end - ne10_block_start) * bs;
                        }
                        void * wdst = (void *) (wdata + i13*nbw3 + i12*nbw2 + i11*nbw1 + ne10_block_start*nbw0);
                        if (did_copy) {
                            // Lead B: a pure row memcpy, no conversion, and the same destination layout as
                            // the general repack (nbw0 == sizeof(f16), bs == 1), so the second sgemm's ldb
                            // is unchanged. Same BITS in, so the GEMM's output bits cannot move.
                            const size_t nbytes =
                                (size_t) (ne10_block_end - ne10_block_start) * bs * ggml_type_size(GGML_TYPE_F16);
                            memcpy(wdst,
                                   (const char *) src1->data + i13*nb13 + i12*nb12 + i11*nb11 + ne10_block_start*bs*nb10,
                                   nbytes);
                            f16_elems_local += (uint64_t) (ne10_block_end - ne10_block_start) * bs;
                        } else if (ggml_cpu_mm_norepack()) {
                        // ABLATION (KATAGO_CPU_MM_NOREPACK=1): identical loop, identical write volume, no F32
                        // read and no conversion arithmetic. Zeros keep the operands FINITE, which is the whole
                        // reason this form is used instead of "skip it": skipping produced non-finite
                        // activations, made KataGo stop printing raw-nn after 1.3 KB instead of 58.6 KB, and a
                        // single unpaired wall reading then looked like a 2.2x win (G-19).
                        // The fill's own cost is INSIDE this measurement, so full-vs-ablation is a LOWER BOUND
                        // on the read+convert half; if even that shows ~0, the repack overlaps and can close.
                        memset(wdst, 0, (size_t) (ne10_block_end - ne10_block_start) * bs * ggml_type_size(vec_dot_type));
                    } else {
                        from_float((float *)((char *) src1->data + i13*nb13 + i12*nb12 + i11*nb11 + ne10_block_start*bs*nb10),
                                   wdst,
                                   (ne10_block_end - ne10_block_start) * bs);
                    }
                }
            }
        }
    #endif
        if (ggml_cpu_act_range()) {
            // Scalar and flag-gated: read-only, so it cannot move the output hash, and it is off in every
            // measured arm, so it cannot move a timing either.
            const size_t bs2 = ggml_blck_size(vec_dot_type);
            const int64_t r_b0 = (ith * ne10/bs2) / nth, r_b1 = ((ith + 1) * ne10/bs2) / nth;
            struct ggml_tensor * r_base = ggml_cpu_base_tensor(src1);
            const int r_prod = (r_base != NULL && r_base->src[0] != NULL) ? (int) r_base->src[0]->op : GGML_OP_COUNT;
            uint64_t r_den = 0;
            for (int64_t i13 = 0; i13 < ne13; i13++)
            for (int64_t i12 = 0; i12 < ne12; i12++)
            for (int64_t i11 = 0; i11 < ne11; i11++) {
                const float * p = (const float *)((const char *) src1->data + i13*nb13 + i12*nb12 + i11*nb11);
                for (int64_t k = r_b0*bs2; k < r_b1*bs2; k++) {
                    const float v = fabsf(p[k]);
                    if (v > ggml_cpu_tls_act_maxabs) { ggml_cpu_tls_act_maxabs = v; }
                    if (v > 65504.0f)                 { ggml_cpu_tls_act_over++; }
                    else if (v > 0.0f && v < 6.103515625e-5f) { ggml_cpu_tls_act_denorm++; r_den++; }
                    ggml_cpu_tls_act_n++;
                }
            }
            if (r_den) { __atomic_fetch_add(&ggml_cpu_act_denorm_by_op[r_prod], r_den, __ATOMIC_RELAXED); }
        }
        if (did_copy) {
            // Lead B's own counters. Kept out of repack_elems/repack_cycles on purpose: those price the
            // F32->f16 conversion, and this is a copy. The wall clock is the arbiter for the copy's cost.
            if (ith == 0) { __atomic_fetch_add(&ggml_cpu_mm_repack_f16_done, 1, __ATOMIC_RELAXED); }
            if (f16_elems_local) {
                __atomic_fetch_add(&ggml_cpu_mm_repack_f16_elems, f16_elems_local, __ATOMIC_RELAXED);
            }
        } else {
            ggml_cpu_tls_repack_cycles += ggml_cpu_bar_tsc() - repack_t0;
        }
    } else {
        if (ith == 0) { __atomic_fetch_add(&ggml_cpu_mm_repack_skip, 1, __ATOMIC_RELAXED); }
    }

    if (ith == 0) {
        // Every thread starts at ith, so the first unprocessed chunk is nth.  This save a bit of coordination right at the start.
        atomic_store_explicit(&params->threadpool->current_chunk, nth, memory_order_relaxed);
    }

    GGML_CPU_BAR_SITE(GGML_CPU_BAR_SITE_MULMAT_CHUNK);
    ggml_barrier(params->threadpool);

#if GGML_USE_LLAMAFILE
    if (src1->type != vec_dot_type || did_copy) {
        // src1 is either F32 (converted into wdata above) or f16 that lead B copied into wdata; in both
        // cases the buffer to read is the scratch. Without opening this site up, lead B's copy would be
        // pure added work: the GEMM would still read src1->data, or fall through to the generic kernel.
        const void* wdata = params->wdata;
        const size_t row_size = ggml_row_size(vec_dot_type, ne10);

        for (int64_t i13 = 0; i13 < ne13; i13++)
            for (int64_t i12 = 0; i12 < ne12; i12++)
                if (!llamafile_sgemm(params,
                                     ne01, ne11, ne00/ggml_blck_size(src0->type),
                                     (const char *)src0->data + i12/r2*nb02 + i13/r3*nb03,
                                     nb01/ggml_type_size(src0->type),
                                     (const char *)wdata + (i12*ne11 + i13*ne12*ne11)*row_size,
                                     row_size/ggml_type_size(vec_dot_type),
                                     (char *)dst->data + i12*nb2 + i13*nb3,
                                     nb1/ggml_type_size(dst->type),
                                     src0->type,
                                     vec_dot_type,
                                     dst->type))
                    goto UseGgmlGemm2;
        // A1: same as the first sgemm site -- and this is the one the measured anchors actually take
        // (src0 f16, vec_dot f16, so the activations were converted into wdata first).
        if (t9_pending) {
            ggml_cpu_t9_pend.diag[3] = 3;
            if (ggml_cpu_tb_take_writes_published()) {
                ggml_cpu_t9_pass_after_sgemm(params, dst);
                ggml_cpu_t9_pend.pass_ran = 1;
            }
        }
        return;
    }
UseGgmlGemm2:;
#endif

    // This is the size of the first dimension of the result, so we can iterate that way. (see the ASSERT above, these are the same numbers)
    const int64_t nr0 = ne0;

    // A1: reaching this line means the GEMM is going to run in the chunk loop, which is where the fused
    // add-pass lives. Every thread passes through here or through one of the early-outs above, and the
    // choice depends only on shapes and types, so the whole pool agrees -- the matcher can therefore
    // decide "skip the ADD" from its OWN thread's flag, with no barrier and no shared state.
    if (t9_pending) {
        ggml_cpu_t9_pend.pass_ran = 1;
    }

    // This is the size of the rest of the dimensions of the result
    const int64_t nr1 = ne1 * ne2 * ne3;

    // Now select a reasonable chunk size.
    int chunk_size = 16;

    // We need to step up the size if it's small
    if (nr0 == 1 || nr1 == 1) {
        chunk_size = 64;
    }

    // distribute the work across the inner or outer loop based on which one is larger
    // The number of chunks in the 0/1 dim.
    // CEIL(nr0/chunk_size)
    int64_t nchunk0 = (nr0 + chunk_size - 1) / chunk_size;
    int64_t nchunk1 = (nr1 + chunk_size - 1) / chunk_size;

    // If the chunking is poor for the number of threads on this setup, scrap the whole plan.  Re-chunk it by thread.
    //   Also, chunking by thread was measured to have perform better on NUMA systems.  See https://github.com/ggml-org/llama.cpp/pull/6915
    //   In theory, chunking should be just as useful on NUMA and non NUMA systems, but testing disagreed with that.
    if (nchunk0 * nchunk1 < nth * 4 || ggml_is_numa()) {
        // distribute the thread work across the inner or outer loop based on which one is larger
        nchunk0 = nr0 > nr1 ? nth : 1; // parallelize by src0 rows
        nchunk1 = nr0 > nr1 ? 1 : nth; // parallelize by src1 rows
    }

    // The number of elements in each chunk
    const int64_t dr0 = (nr0 + nchunk0 - 1) / nchunk0;
    const int64_t dr1 = (nr1 + nchunk1 - 1) / nchunk1;

    // The first chunk comes from our thread_id, the rest will get auto-assigned.
    int current_chunk = ith;

    while (current_chunk < nchunk0 * nchunk1) {
        const int64_t ith0 = current_chunk % nchunk0;
        const int64_t ith1 = current_chunk / nchunk0;

        const int64_t ir0_start = dr0 * ith0;
        const int64_t ir0_end = MIN(ir0_start + dr0, nr0);

        const int64_t ir1_start = dr1 * ith1;
        const int64_t ir1_end = MIN(ir1_start + dr1, nr1);

        // dot kernels can handle 1 row and col at a time, but mmla kernels can process 2 rows and cols
        int64_t num_rows_per_vec_dot = vec_dot_num_rows;

        // these checks are needed to avoid crossing dim1 boundaries
        // can be optimized, but the logic would become more complicated, so keeping it like this for simplicity
        if ((nr0 % 2 != 0) || (ne11 % 2 != 0) || ((ir0_end - ir0_start) % 2 != 0) || ((ir1_end - ir1_start) % 2 != 0)) {
            num_rows_per_vec_dot = 1;
        }
        ggml_compute_forward_mul_mat_one_chunk(params, dst, src0->type, num_rows_per_vec_dot, ir0_start, ir0_end, ir1_start, ir1_end);

        // A1: fold the pending ADD into the chunk this thread just produced. Deliberately the SAME
        // ir0/ir1 ranges as the call above, which is what makes every element read here an element this
        // thread wrote a moment ago -- the constraint that lets the pass live without a barrier, and the
        // reason it must not be hoisted out of this loop.
        if (t9_pending) {
            ggml_cpu_t9_pass_in_chunk(params, dst, ir0_start, ir0_end, ir1_start, ir1_end);
        }

        if (nth >= nchunk0 * nchunk1) {
            break;
        }

        current_chunk = atomic_fetch_add_explicit(&params->threadpool->current_chunk, 1, memory_order_relaxed);
    }
}

// ggml_compute_forward_mul_mat_id

#define MMID_MATRIX_ROW(row_id, i1) matrix_rows[(row_id)*ids->ne[0]*ids->ne[1] + (i1)]

struct mmid_row_mapping {
    int32_t i1;
    int32_t i2;
};

static void ggml_compute_forward_mul_mat_id_one_chunk(
    struct ggml_tensor * dst,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    const struct ggml_tensor * ids,
    const int64_t cur_a,
    const int64_t ir0_start,
    const int64_t ir0_end,
    const int64_t ir1_start,
    const int64_t ir1_end,
    const char * src0_cur,
    const struct mmid_row_mapping * matrix_rows,
    const size_t row_size,
    const bool src1_cont,
    const void * wdata) {

    GGML_TENSOR_BINARY_OP_LOCALS

    const enum ggml_type type = src0->type;

    ggml_vec_dot_t    const vec_dot      = type_traits_cpu[type].vec_dot;
    enum ggml_type    const vec_dot_type = type_traits_cpu[type].vec_dot_type;

    const int64_t blck_0 = 16;
    const int64_t blck_1 = 16;

    float tmp[16];

    for (int64_t iir1 = ir1_start; iir1 < ir1_end; iir1 += blck_1) {
        for (int64_t iir0 = ir0_start; iir0 < ir0_end; iir0 += blck_0) {
            for (int64_t ir1 = iir1; ir1 < iir1 + blck_1 && ir1 < ir1_end; ++ir1) {
                const int64_t _i12 = ir1; // logical row index for this expert

                struct mmid_row_mapping row_mapping = MMID_MATRIX_ROW(cur_a, _i12);
                const int id       = row_mapping.i1; // selected expert index

                const int64_t  i11 = id % ne11;
                const int64_t  i12 = row_mapping.i2; // row index in src1

                const int64_t  i1 = id;  // selected expert index
                const int64_t  i2 = i12; // row

                // desc: when src1 is not a contiguous memory block we have to calculate the offset using the strides
                //       if it is, then we have either copied the data to params->wdata and made it contiguous or we are using
                //       the original src1 data pointer, so we should index using the indices directly
                // TODO: this is a bit of a hack, we should probably have a better way to handle this
                const char * src1_col = (const char *) wdata +
                    (src1_cont || src1->type != vec_dot_type
                    ? (i11      + i12*ne11)*row_size
                    : (i11*nb11 + i12*nb12));

                float * dst_col = (float *) ((char *) dst->data + (i1*nb1 + i2*nb2));

                for (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_end; ++ir0) {
                    vec_dot(ne00, &tmp[ir0 - iir0], 0, src0_cur + ir0*nb01, 0, src1_col, 0, 1);
                }

                memcpy(&dst_col[iir0], tmp, (MIN(iir0 + blck_0, ir0_end) - iir0)*sizeof(float));
            }
        }
    }
}

static void * incr_ptr_aligned(void ** p, size_t size, size_t align) {

    void * ptr = *p;
    ptr = (void *) GGML_PAD((uintptr_t) ptr, align);
    *p = (void *) ((char *) ptr + size);
    return ptr;
}

static void ggml_compute_forward_mul_mat_id(
        const struct ggml_compute_params * params,
              struct ggml_tensor * dst) {

    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];
    const struct ggml_tensor * ids = dst->src[2];

    GGML_TENSOR_BINARY_OP_LOCALS

    const int ith = params->ith;
    const int nth = params->nth;

    const enum ggml_type type = src0->type;

    const bool src1_cont = ggml_is_contiguous(src1);

    enum ggml_type    const vec_dot_type    = type_traits_cpu[type].vec_dot_type;
    ggml_from_float_t const from_float      = type_traits_cpu[vec_dot_type].from_float;

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_size(type));
    GGML_ASSERT(nb10 == ggml_type_size(src1->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == sizeof(float));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    // row groups
    const int n_ids = ids->ne[0]; // n_expert_used
    const int n_as  = ne02;       // n_expert

    void * wdata_cur = params->wdata;

    if (src1->type != vec_dot_type) {
        incr_ptr_aligned(&wdata_cur, ggml_row_size(vec_dot_type, ggml_nelements(src1)), sizeof(int64_t));
    }

    int64_t * matrix_row_counts = // [n_as]
        incr_ptr_aligned(&wdata_cur, n_as*sizeof(int64_t), sizeof(int64_t));

    struct mmid_row_mapping * matrix_rows = // [n_as][ids->ne[0]*ids->ne[1]]
        incr_ptr_aligned(&wdata_cur, n_as*ids->ne[0]*ids->ne[1]*sizeof(struct mmid_row_mapping), sizeof(int64_t));

    char (*atomic_current_chunk)[CACHE_LINE_SIZE] = // [n_as]
        incr_ptr_aligned(&wdata_cur, CACHE_LINE_SIZE * n_as, CACHE_LINE_SIZE);

    GGML_ASSERT(params->wsize >= (size_t)((char *) wdata_cur - (char *) params->wdata));

    if (src1->type != vec_dot_type) {
        char * wdata = params->wdata;

        const size_t nbw0 = ggml_type_size(vec_dot_type);
        const size_t nbw1 = ggml_row_size(vec_dot_type, ne10);
        const size_t nbw2 = nbw1*ne11;
        const size_t nbw3 = nbw2*ne12;

        assert(params->wsize >= ne13*nbw3);
        GGML_ASSERT(src1->type == GGML_TYPE_F32);

#if 0
        for (int64_t i13 = 0; i13 < ne13; ++i13) {
            for (int64_t i12 = ith; i12 < ne12; i12 += nth) {
                for (int64_t i11 = 0; i11 < ne11; ++i11) {
                    from_float((float *)((char *) src1->data + i13*nb13 + i12*nb12 + i11*nb11),
                               (void *)               (wdata + i13*nbw3 + i12*nbw2 + i11*nbw1),
                               ne10);
                }
            }
        }
#else
        for (int64_t i13 = 0; i13 < ne13; ++i13) {
            for (int64_t i12 = 0; i12 < ne12; ++i12) {
                for (int64_t i11 = 0; i11 < ne11; ++i11) {
                    size_t bs = ggml_blck_size(vec_dot_type);
                    int64_t ne10_block_start = (ith * ne10/bs) / nth;
                    int64_t ne10_block_end   = ((ith + 1) * ne10/bs) / nth;
                    ggml_cpu_tls_repack_elems += (uint64_t) (ne10_block_end - ne10_block_start) * bs;
                    void * wdst = (void *) (wdata + i13*nbw3 + i12*nbw2 + i11*nbw1 + ne10_block_start*nbw0);
                    if (ggml_cpu_mm_norepack()) {
                        // ABLATION (KATAGO_CPU_MM_NOREPACK=1): identical loop, identical write volume, no F32
                        // read and no conversion arithmetic. Zeros keep the operands FINITE, which is the whole
                        // reason this form is used instead of "skip it": skipping produced non-finite
                        // activations, made KataGo stop printing raw-nn after 1.3 KB instead of 58.6 KB, and a
                        // single unpaired wall reading then looked like a 2.2x win (G-19).
                        // The fill's own cost is INSIDE this measurement, so full-vs-ablation is a LOWER BOUND
                        // on the read+convert half; if even that shows ~0, the repack overlaps and can close.
                        memset(wdst, 0, (size_t) (ne10_block_end - ne10_block_start) * bs * ggml_type_size(vec_dot_type));
                    } else {
                        from_float((float *)((char *) src1->data + i13*nb13 + i12*nb12 + i11*nb11 + ne10_block_start*bs*nb10),
                                   wdst,
                                   (ne10_block_end - ne10_block_start) * bs);
                    }
                }
            }
        }
#endif
    }

    if (ith == 0) {
        // initialize matrix_row_counts
        memset(matrix_row_counts, 0, n_as*sizeof(int64_t));

        // group rows by src0 matrix
        for (int64_t iid1 = 0; iid1 < ids->ne[1]; ++iid1) {
            for (int id = 0; id < n_ids; ++id) {
                const int32_t i02 = *(const int32_t *) ((const char *) ids->data + iid1*ids->nb[1] + id*ids->nb[0]);

                assert(i02 >= 0 && i02 < n_as);

                MMID_MATRIX_ROW(i02, matrix_row_counts[i02]) = (struct mmid_row_mapping) {id, iid1};
                matrix_row_counts[i02] += 1;
            }
        }
    }

    // reset current_chunk
    for (int cur_a = ith; cur_a < n_as; cur_a += nth) {
        atomic_int * current_chunk_ctr = (atomic_int *)(atomic_current_chunk + cur_a);
        *current_chunk_ctr = nth;
    }

    GGML_CPU_BAR_SITE(GGML_CPU_BAR_SITE_MULMATID_CHUNK);
    ggml_barrier(params->threadpool);

    for (int cur_a = 0; cur_a < n_as; ++cur_a) {
        const int64_t cne1 = matrix_row_counts[cur_a];

        if (cne1 == 0) {
            continue;
        }

        const char * src0_cur = (const char *) src0->data + cur_a * nb02;
        const void * wdata = (src1->type == vec_dot_type) ? src1->data : params->wdata;
        const size_t row_size = ggml_row_size(vec_dot_type, ne10);

        const int64_t nr0 = ne01;
        const int64_t nr1 = cne1;

        int chunk_size = 16;
        if (nr0 == 1 || nr1 == 1) {
            chunk_size = 64;
        }

        // disable for NUMA
        const bool disable_chunking = ggml_is_numa();

        int64_t nchunk0 = (nr0 + chunk_size - 1) / chunk_size;
        int64_t nchunk1 = (nr1 + chunk_size - 1) / chunk_size;

        if (nchunk0 * nchunk1 < nth * 4 || disable_chunking) {
            nchunk0 = nr0 > nr1 ? nth : 1;
            nchunk1 = nr0 > nr1 ? 1 : nth;
        }

        const int64_t dr0 = (nr0 + nchunk0 - 1) / nchunk0;
        const int64_t dr1 = (nr1 + nchunk1 - 1) / nchunk1;

        int current_chunk = ith;

        atomic_int * current_chunk_ctr = (atomic_int *)(atomic_current_chunk + cur_a);

        while (current_chunk < nchunk0 * nchunk1) {
            const int64_t ith0 = current_chunk % nchunk0;
            const int64_t ith1 = current_chunk / nchunk0;

            const int64_t ir0_start = dr0 * ith0;
            const int64_t ir0_end = MIN(ir0_start + dr0, nr0);

            const int64_t ir1_start = dr1 * ith1;
            const int64_t ir1_end = MIN(ir1_start + dr1, nr1);

            ggml_compute_forward_mul_mat_id_one_chunk(
                dst, src0, src1, ids, cur_a,
                ir0_start, ir0_end, ir1_start, ir1_end,
                src0_cur, matrix_rows, row_size, src1_cont, wdata
            );

            if (nth >= nchunk0 * nchunk1) {
                break;
            }

            current_chunk = atomic_fetch_add_explicit(current_chunk_ctr, 1, memory_order_relaxed);
        }
    }
}

/////////////////////////////////

static void ggml_compute_forward(struct ggml_compute_params * params, struct ggml_tensor * tensor) {
    GGML_ASSERT(params);

    if (tensor->op == GGML_OP_NONE || ggml_is_empty(tensor)) {
        return;
    }

    // extra_buffer op?
    if (ggml_cpu_extra_compute_forward(params, tensor)) {
        return;
    }

    switch (tensor->op) {
        case GGML_OP_DUP:
            {
                ggml_compute_forward_dup(params, tensor);
            } break;
        case GGML_OP_ADD:
            {
                ggml_compute_forward_add(params, tensor);
            } break;
        case GGML_OP_ADD_ID:
            {
                ggml_compute_forward_add_id(params, tensor);
            } break;
        case GGML_OP_ADD1:
            {
                ggml_compute_forward_add1(params, tensor);
            } break;
        case GGML_OP_ACC:
            {
                ggml_compute_forward_acc(params, tensor);
            } break;
        case GGML_OP_SUB:
            {
                ggml_compute_forward_sub(params, tensor);
            } break;
        case GGML_OP_MUL:
            {
                ggml_compute_forward_mul(params, tensor);
            } break;
        case GGML_OP_DIV:
            {
                ggml_compute_forward_div(params, tensor);
            } break;
        case GGML_OP_SQR:
            {
                ggml_compute_forward_sqr(params, tensor);
            } break;
        case GGML_OP_SQRT:
            {
                ggml_compute_forward_sqrt(params, tensor);
            } break;
        case GGML_OP_LOG:
            {
                ggml_compute_forward_log(params, tensor);
            } break;
        case GGML_OP_SIN:
            {
                ggml_compute_forward_sin(params, tensor);
            } break;
        case GGML_OP_COS:
            {
                ggml_compute_forward_cos(params, tensor);
            } break;
        case GGML_OP_SUM:
            {
                ggml_compute_forward_sum(params, tensor);
            } break;
        case GGML_OP_SUM_ROWS:
            {
                ggml_compute_forward_sum_rows(params, tensor);
            } break;
        case GGML_OP_CUMSUM:
            {
                ggml_compute_forward_cumsum(params, tensor);
            } break;
        case GGML_OP_MEAN:
            {
                ggml_compute_forward_mean(params, tensor);
            } break;
        case GGML_OP_ARGMAX:
            {
                ggml_compute_forward_argmax(params, tensor);
            } break;
        case GGML_OP_COUNT_EQUAL:
            {
                ggml_compute_forward_count_equal(params, tensor);
            } break;
        case GGML_OP_REPEAT:
            {
                ggml_compute_forward_repeat(params, tensor);
            } break;
        case GGML_OP_REPEAT_BACK:
            {
                ggml_compute_forward_repeat_back(params, tensor);
            } break;
        case GGML_OP_CONCAT:
            {
                ggml_compute_forward_concat(params, tensor);
            } break;
        case GGML_OP_SILU_BACK:
            {
                ggml_compute_forward_silu_back(params, tensor);
            } break;
        case GGML_OP_NORM:
            {
                ggml_compute_forward_norm(params, tensor);
            } break;
        case GGML_OP_RMS_NORM:
            {
                ggml_compute_forward_rms_norm(params, tensor);
            } break;
        case GGML_OP_RMS_NORM_BACK:
            {
                ggml_compute_forward_rms_norm_back(params, tensor);
            } break;
        case GGML_OP_GROUP_NORM:
            {
                ggml_compute_forward_group_norm(params, tensor);
            } break;
        case GGML_OP_L2_NORM:
            {
                ggml_compute_forward_l2_norm(params, tensor);
            } break;
        case GGML_OP_MUL_MAT:
            {
                ggml_compute_forward_mul_mat(params, tensor);
            } break;
        case GGML_OP_MUL_MAT_ID:
            {
                ggml_compute_forward_mul_mat_id(params, tensor);
            } break;
        case GGML_OP_OUT_PROD:
            {
                ggml_compute_forward_out_prod(params, tensor);
            } break;
        case GGML_OP_SCALE:
            {
                ggml_compute_forward_scale(params, tensor);
            } break;
        case GGML_OP_SET:
            {
                ggml_compute_forward_set(params, tensor);
            } break;
        case GGML_OP_CPY:
            {
                ggml_compute_forward_cpy(params, tensor);
            } break;
        case GGML_OP_CONT:
            {
                ggml_compute_forward_cont(params, tensor);
            } break;
        case GGML_OP_GET_ROWS:
            {
                ggml_compute_forward_get_rows(params, tensor);
            } break;
        case GGML_OP_GET_ROWS_BACK:
            {
                ggml_compute_forward_get_rows_back(params, tensor);
            } break;
        case GGML_OP_SET_ROWS:
            {
                ggml_compute_forward_set_rows(params, tensor);
            } break;
        case GGML_OP_DIAG:
            {
                ggml_compute_forward_diag(params, tensor);
            } break;
        case GGML_OP_DIAG_MASK_INF:
            {
                ggml_compute_forward_diag_mask_inf(params, tensor);
            } break;
        case GGML_OP_DIAG_MASK_ZERO:
            {
                ggml_compute_forward_diag_mask_zero(params, tensor);
            } break;
        case GGML_OP_SOFT_MAX:
            {
                ggml_compute_forward_soft_max(params, tensor);
            } break;
        case GGML_OP_SOFT_MAX_BACK:
            {
                ggml_compute_forward_soft_max_ext_back(params, tensor);
            } break;
        case GGML_OP_ROPE:
            {
                ggml_compute_forward_rope(params, tensor);
            } break;
        case GGML_OP_ROPE_BACK:
            {
                ggml_compute_forward_rope_back(params, tensor);
            } break;
        case GGML_OP_CLAMP:
            {
                ggml_compute_forward_clamp(params, tensor);
            } break;
        case GGML_OP_CONV_TRANSPOSE_1D:
            {
                ggml_compute_forward_conv_transpose_1d(params, tensor);
            } break;
        case GGML_OP_IM2COL:
            {
                ggml_compute_forward_im2col(params, tensor);
            } break;
        case GGML_OP_IM2COL_BACK:
            {
                ggml_compute_forward_im2col_back_f32(params, tensor);
            } break;
        case GGML_OP_IM2COL_3D:
            {
                ggml_compute_forward_im2col_3d(params, tensor);
            } break;
        case GGML_OP_COL2IM_1D:
            {
                ggml_compute_forward_col2im_1d(params, tensor);
            } break;
        case GGML_OP_CONV_2D:
            {
                ggml_compute_forward_conv_2d(params, tensor);
            } break;
        case GGML_OP_CONV_3D:
            {
                ggml_compute_forward_conv_3d(params, tensor);
            } break;
        case GGML_OP_CONV_2D_DW:
            {
                ggml_compute_forward_conv_2d_dw(params, tensor);
            } break;
        case GGML_OP_CONV_TRANSPOSE_2D:
            {
                ggml_compute_forward_conv_transpose_2d(params, tensor);
            } break;
        case GGML_OP_POOL_1D:
            {
                ggml_compute_forward_pool_1d(params, tensor);
            } break;
        case GGML_OP_POOL_2D:
            {
                ggml_compute_forward_pool_2d(params, tensor);
            } break;
        case GGML_OP_POOL_2D_BACK:
            {
                ggml_compute_forward_pool_2d_back(params, tensor);
            } break;
        case GGML_OP_UPSCALE:
            {
                ggml_compute_forward_upscale(params, tensor);
            } break;
        case GGML_OP_PAD:
            {
                ggml_compute_forward_pad(params, tensor);
            } break;
        case GGML_OP_PAD_REFLECT_1D:
            {
                ggml_compute_forward_pad_reflect_1d(params, tensor);
            } break;
        case GGML_OP_ROLL:
            {
                ggml_compute_forward_roll(params, tensor);
            } break;
        case GGML_OP_ARANGE:
            {
                ggml_compute_forward_arange(params, tensor);
            } break;
        case GGML_OP_TIMESTEP_EMBEDDING:
            {
                ggml_compute_forward_timestep_embedding(params, tensor);
            } break;
        case GGML_OP_ARGSORT:
            {
                ggml_compute_forward_argsort(params, tensor);
            } break;
        case GGML_OP_TOP_K:
            {
                ggml_compute_forward_top_k(params, tensor);
            } break;
        case GGML_OP_LEAKY_RELU:
            {
                ggml_compute_forward_leaky_relu(params, tensor);
            } break;
        case GGML_OP_TRI:
            {
                ggml_compute_forward_tri(params, tensor);
            } break;
        case GGML_OP_FILL:
            {
                ggml_compute_forward_fill(params, tensor);
            } break;
        case GGML_OP_FLASH_ATTN_EXT:
            {
                ggml_compute_forward_flash_attn_ext(params, tensor);
            } break;
        case GGML_OP_FLASH_ATTN_BACK:
            {
                int32_t t = ggml_get_op_params_i32(tensor, 0);
                GGML_ASSERT(t == 0 || t == 1);
                bool masked = t != 0;
                ggml_compute_forward_flash_attn_back(params, masked, tensor);
            } break;
        case GGML_OP_SSM_CONV:
            {
                ggml_compute_forward_ssm_conv(params, tensor);
            } break;
        case GGML_OP_SSM_SCAN:
            {
                ggml_compute_forward_ssm_scan(params, tensor);
            } break;
        case GGML_OP_WIN_PART:
            {
                ggml_compute_forward_win_part(params, tensor);
            } break;
        case GGML_OP_WIN_UNPART:
            {
                ggml_compute_forward_win_unpart(params, tensor);
            } break;
        case GGML_OP_UNARY:
            {
                ggml_compute_forward_unary(params, tensor);
            } break;
        case GGML_OP_GLU:
            {
                ggml_compute_forward_glu(params, tensor);
            } break;
        case GGML_OP_GET_REL_POS:
            {
                ggml_compute_forward_get_rel_pos(params, tensor);
            } break;
        case GGML_OP_ADD_REL_POS:
            {
                ggml_compute_forward_add_rel_pos(params, tensor);
            } break;
        case GGML_OP_RWKV_WKV6:
            {
                ggml_compute_forward_rwkv_wkv6(params, tensor);
            } break;
        case GGML_OP_GATED_LINEAR_ATTN:
            {
                ggml_compute_forward_gla(params, tensor);
            } break;
        case GGML_OP_RWKV_WKV7:
            {
                ggml_compute_forward_rwkv_wkv7(params, tensor);
            } break;
        case GGML_OP_SOLVE_TRI:
            {
                ggml_compute_forward_solve_tri(params, tensor);
            } break;
        case GGML_OP_GATED_DELTA_NET:
            {
                ggml_compute_forward_gated_delta_net(params, tensor);
            } break;
        case GGML_OP_LIGHTNING_INDEXER:
            {
                ggml_compute_forward_lightning_indexer(params, tensor);
            } break;
        case GGML_OP_DSV4_HC_COMB:
            {
                ggml_compute_forward_dsv4_hc_comb(params, tensor);
            } break;
        case GGML_OP_DSV4_HC_PRE:
            {
                ggml_compute_forward_dsv4_hc_pre(params, tensor);
            } break;
        case GGML_OP_DSV4_HC_POST:
            {
                ggml_compute_forward_dsv4_hc_post(params, tensor);
            } break;
        case GGML_OP_MAP_CUSTOM1:
            {
                ggml_compute_forward_map_custom1(params, tensor);
            }
            break;
        case GGML_OP_MAP_CUSTOM2:
            {
                ggml_compute_forward_map_custom2(params, tensor);
            }
            break;
        case GGML_OP_MAP_CUSTOM3:
            {
                ggml_compute_forward_map_custom3(params, tensor);
            }
            break;
        case GGML_OP_CUSTOM:
            {
                ggml_compute_forward_custom(params, tensor);
            }
            break;
        case GGML_OP_CROSS_ENTROPY_LOSS:
            {
                ggml_compute_forward_cross_entropy_loss(params, tensor);
            }
            break;
        case GGML_OP_CROSS_ENTROPY_LOSS_BACK:
            {
                ggml_compute_forward_cross_entropy_loss_back(params, tensor);
            }
            break;
        case GGML_OP_OPT_STEP_ADAMW:
            {
                ggml_compute_forward_opt_step_adamw(params, tensor);
            }
            break;
        case GGML_OP_OPT_STEP_SGD:
            {
                ggml_compute_forward_opt_step_sgd(params, tensor);
            }
            break;
        case GGML_OP_NONE:
            {
                // nop
            } break;
        case GGML_OP_RESHAPE:
            {
                // nop
            } break;
        case GGML_OP_PERMUTE:
            {
                // nop
            } break;
        case GGML_OP_VIEW:
            {
                // nop
            } break;
        case GGML_OP_TRANSPOSE:
            {
                // nop
            } break;
        case GGML_OP_COUNT:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// Android's libc implementation "bionic" does not support setting affinity
#if defined(__gnu_linux__)
static void set_numa_thread_affinity(int thread_n) {
    if (!ggml_is_numa()) {
        return;
    }

    int node_num;
    int rv;
    size_t setsize = CPU_ALLOC_SIZE(g_state.numa.total_cpus);

    switch(g_state.numa.numa_strategy) {
        case GGML_NUMA_STRATEGY_DISTRIBUTE:
            // run thread on node_num thread_n / (threads per node)
            node_num = thread_n % g_state.numa.n_nodes;
            break;
        case GGML_NUMA_STRATEGY_ISOLATE:
            // run thread on current_node
            node_num = g_state.numa.current_node;
            break;
        case GGML_NUMA_STRATEGY_NUMACTL:
            // use the cpuset that numactl gave us
            rv = pthread_setaffinity_np(pthread_self(), setsize, &g_state.numa.cpuset);
            if (rv) {
                fprintf(stderr, "warning: pthread_setaffinity_np() failed: %s\n",strerror(rv));
            }
            return;
        default:
            return;
    }

    struct ggml_numa_node * node = &g_state.numa.nodes[node_num];

    cpu_set_t * cpus = CPU_ALLOC(g_state.numa.total_cpus);
    CPU_ZERO_S(setsize, cpus);
    for (size_t i = 0; i < node->n_cpus; ++i) {
        CPU_SET_S(node->cpus[i], setsize, cpus);
    }

    rv = pthread_setaffinity_np(pthread_self(), setsize, cpus);
    if (rv) {
            fprintf(stderr, "warning: pthread_setaffinity_np() failed: %s\n", strerror(rv));
    }

    CPU_FREE(cpus);
}

static void clear_numa_thread_affinity(void) {
    if (!ggml_is_numa()) {
        return;
    }

    size_t setsize = CPU_ALLOC_SIZE(g_state.numa.total_cpus);

    cpu_set_t * cpus = CPU_ALLOC(g_state.numa.total_cpus);
    CPU_ZERO_S(setsize, cpus);
    for (unsigned i = 0; i < g_state.numa.total_cpus; ++i) {
        CPU_SET_S(i, setsize, cpus);
    }

    int rv = pthread_setaffinity_np(pthread_self(), setsize, cpus);
    if (rv) {
        fprintf(stderr, "warning: pthread_setaffinity_np() failed: %s\n", strerror(rv));
    }

    CPU_FREE(cpus);
}
#else
// TODO: Windows etc.
// (the linux implementation may also work on BSD, someone should test)
static void set_numa_thread_affinity(int thread_n) { UNUSED(thread_n);  }
static void clear_numa_thread_affinity(void) {}
#endif

static int ggml_get_n_tasks(struct ggml_tensor * node, int n_threads) {
    int n_tasks = 0;

    if (ggml_is_empty(node)) {
        // no need to multi-thread a no-op
        n_tasks = 1;
        return n_tasks;
    }

    switch (node->op) {
        case GGML_OP_CPY:
        case GGML_OP_DUP:
        case GGML_OP_CONT:
        case GGML_OP_ADD:
        case GGML_OP_ADD_ID:
        case GGML_OP_ADD1:
        case GGML_OP_ACC:
        case GGML_OP_CUMSUM:
        case GGML_OP_TRI:
        case GGML_OP_FILL:
            {
                n_tasks = n_threads;
            } break;
        case GGML_OP_SUB:
        case GGML_OP_SQR:
        case GGML_OP_SQRT:
        case GGML_OP_LOG:
        case GGML_OP_SIN:
        case GGML_OP_COS:
        case GGML_OP_SUM:
        case GGML_OP_SUM_ROWS:
        case GGML_OP_MEAN:
        case GGML_OP_ARGMAX:
            {
                n_tasks = 1;
            } break;
        case GGML_OP_COUNT_EQUAL:
        case GGML_OP_SOLVE_TRI:
        case GGML_OP_GATED_DELTA_NET:
        case GGML_OP_DSV4_HC_COMB:
        case GGML_OP_DSV4_HC_PRE:
        case GGML_OP_DSV4_HC_POST:
            {
                n_tasks = n_threads;
            } break;
        case GGML_OP_REPEAT:
        case GGML_OP_REPEAT_BACK:
        case GGML_OP_LEAKY_RELU:
            {
                n_tasks = 1;
            } break;
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(node)) {
                case GGML_UNARY_OP_ABS:
                case GGML_UNARY_OP_SGN:
                case GGML_UNARY_OP_NEG:
                case GGML_UNARY_OP_STEP:
                case GGML_UNARY_OP_TANH:
                case GGML_UNARY_OP_ELU:
                case GGML_UNARY_OP_RELU:
                case GGML_UNARY_OP_SIGMOID:
                case GGML_UNARY_OP_HARDSWISH:
                case GGML_UNARY_OP_HARDSIGMOID:
                case GGML_UNARY_OP_EXP:
                case GGML_UNARY_OP_SOFTPLUS:
                case GGML_UNARY_OP_EXPM1:
                case GGML_UNARY_OP_FLOOR:
                case GGML_UNARY_OP_CEIL:
                case GGML_UNARY_OP_ROUND:
                case GGML_UNARY_OP_TRUNC:
                    {
                        n_tasks = 1;
                    } break;

                case GGML_UNARY_OP_GELU:
                case GGML_UNARY_OP_GELU_ERF:
                case GGML_UNARY_OP_GELU_QUICK:
                case GGML_UNARY_OP_SILU:
                case GGML_UNARY_OP_XIELU:
                    {
                        n_tasks = n_threads;
                    } break;
                default:
                    GGML_ABORT("fatal error");
            }
            break;
        case GGML_OP_GLU:
            switch (ggml_get_glu_op(node)) {
                case GGML_GLU_OP_REGLU:
                case GGML_GLU_OP_GEGLU:
                case GGML_GLU_OP_SWIGLU:
                case GGML_GLU_OP_SWIGLU_OAI:
                case GGML_GLU_OP_GEGLU_ERF:
                case GGML_GLU_OP_GEGLU_QUICK:
                    {
                        n_tasks = n_threads;
                    } break;
                default:
                    GGML_ABORT("fatal error");
            }
            break;
        case GGML_OP_SILU_BACK:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
        case GGML_OP_NORM:
        case GGML_OP_RMS_NORM:
        case GGML_OP_RMS_NORM_BACK:
        case GGML_OP_L2_NORM:
        case GGML_OP_GROUP_NORM:
        case GGML_OP_CONCAT:
        case GGML_OP_MUL_MAT:
        case GGML_OP_MUL_MAT_ID:
        case GGML_OP_OUT_PROD:
            {
                n_tasks = n_threads;
            } break;
        case GGML_OP_GET_ROWS:
        case GGML_OP_SET_ROWS:
            {
                // FIXME: get_rows can use additional threads, but the cost of launching additional threads
                // decreases performance with GPU offloading
                //n_tasks = n_threads;
                n_tasks = 1;
            } break;
        case GGML_OP_SCALE:
        case GGML_OP_SET:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_GET_ROWS_BACK:
        case GGML_OP_DIAG:
            {
                n_tasks = 1;
            } break;
        case GGML_OP_DIAG_MASK_ZERO:
        case GGML_OP_DIAG_MASK_INF:
        case GGML_OP_SOFT_MAX_BACK:
        case GGML_OP_ROPE:
        case GGML_OP_ROPE_BACK:
        case GGML_OP_ADD_REL_POS:
            {
                n_tasks = n_threads;
            } break;
        case GGML_OP_CLAMP:
            {
                n_tasks = 1; //TODO
            } break;
        case GGML_OP_SOFT_MAX:
            {
                n_tasks = MIN(n_threads, ggml_nrows(node->src[0]));
            } break;
        case GGML_OP_IM2COL:
        case GGML_OP_IM2COL_BACK:
        case GGML_OP_IM2COL_3D:
        case GGML_OP_CONV_2D:
        case GGML_OP_CONV_3D:
        case GGML_OP_CONV_2D_DW:
        case GGML_OP_COL2IM_1D:
        case GGML_OP_CONV_TRANSPOSE_1D:
        case GGML_OP_CONV_TRANSPOSE_2D:
            {
                n_tasks = n_threads;
            } break;
        case GGML_OP_POOL_1D:
        case GGML_OP_POOL_2D:
        case GGML_OP_POOL_2D_BACK:
            {
                n_tasks = 1;
            } break;
        case GGML_OP_UPSCALE:
        case GGML_OP_PAD:
        case GGML_OP_PAD_REFLECT_1D:
        case GGML_OP_ROLL:
        case GGML_OP_ARANGE:
        case GGML_OP_TIMESTEP_EMBEDDING:
        case GGML_OP_ARGSORT:
        case GGML_OP_TOP_K:
        case GGML_OP_FLASH_ATTN_EXT:
        case GGML_OP_FLASH_ATTN_BACK:
        case GGML_OP_SSM_CONV:
        case GGML_OP_SSM_SCAN:
        case GGML_OP_LIGHTNING_INDEXER:
            {
                n_tasks = n_threads;
            } break;
        case GGML_OP_RWKV_WKV6:
        case GGML_OP_GATED_LINEAR_ATTN:
        case GGML_OP_RWKV_WKV7:
            {
                const int64_t n_heads = node->src[1]->ne[1];
                n_tasks = MIN(n_threads, n_heads);
            } break;
        case GGML_OP_WIN_PART:
        case GGML_OP_WIN_UNPART:
        case GGML_OP_GET_REL_POS:
            {
                n_tasks = 1;
            } break;
        case GGML_OP_MAP_CUSTOM1:
            {
                struct ggml_map_custom1_op_params p;
                memcpy(&p, node->op_params, sizeof(p));
                if (p.n_tasks == GGML_N_TASKS_MAX) {
                    n_tasks = n_threads;
                } else {
                    n_tasks = MIN(p.n_tasks, n_threads);
                }
            } break;
        case GGML_OP_MAP_CUSTOM2:
            {
                struct ggml_map_custom2_op_params p;
                memcpy(&p, node->op_params, sizeof(p));
                if (p.n_tasks == GGML_N_TASKS_MAX) {
                    n_tasks = n_threads;
                } else {
                    n_tasks = MIN(p.n_tasks, n_threads);
                }
            } break;
        case GGML_OP_MAP_CUSTOM3:
            {
                struct ggml_map_custom3_op_params p;
                memcpy(&p, node->op_params, sizeof(p));
                if (p.n_tasks == GGML_N_TASKS_MAX) {
                    n_tasks = n_threads;
                } else {
                    n_tasks = MIN(p.n_tasks, n_threads);
                }
            } break;
        case GGML_OP_CUSTOM:
            {
                struct ggml_custom_op_params p;
                memcpy(&p, node->op_params, sizeof(p));
                if (p.n_tasks == GGML_N_TASKS_MAX) {
                    n_tasks = n_threads;
                } else {
                    n_tasks = MIN(p.n_tasks, n_threads);
                }
            } break;
        case GGML_OP_CROSS_ENTROPY_LOSS:
        case GGML_OP_CROSS_ENTROPY_LOSS_BACK:
        case GGML_OP_OPT_STEP_ADAMW:
        case GGML_OP_OPT_STEP_SGD:
            {
                n_tasks = n_threads;
            } break;
        case GGML_OP_NONE:
            {
                n_tasks = 1;
            } break;
        case GGML_OP_COUNT:
            {
                GGML_ABORT("fatal error");
            }
        default:
            {
                fprintf(stderr, "%s: op not implemented: ", __func__);
                if (node->op < GGML_OP_COUNT) {
                    fprintf(stderr, "%s\n", ggml_op_name(node->op));
                } else {
                    fprintf(stderr, "%d\n", node->op);
                }
                GGML_ABORT("fatal error");
            }
    }

    assert(n_tasks > 0);

    return n_tasks;
}

static thread_ret_t ggml_graph_compute_secondary_thread(void* data);

#if defined(_WIN32)
#include "windows.h"

// TODO: support > 64 CPUs
static bool ggml_thread_apply_affinity(bool * mask) {
    HANDLE    h = GetCurrentThread();
    uint64_t  bitmask = 0ULL;

    assert(GGML_MAX_N_THREADS >= 64);

    for (int32_t i = 0; i < 8; i++) {
        int32_t idx = i * 8;
        uint8_t val = 0;
        val |= mask[idx + 0] << 0;
        val |= mask[idx + 1] << 1;
        val |= mask[idx + 2] << 2;
        val |= mask[idx + 3] << 3;
        val |= mask[idx + 4] << 4;
        val |= mask[idx + 5] << 5;
        val |= mask[idx + 6] << 6;
        val |= mask[idx + 7] << 7;
        bitmask |= (uint64_t)val << idx;
    }

    for (int32_t i = 64; i < GGML_MAX_N_THREADS; i++) {
        if (mask[i]) {
            fprintf(stderr, "warn: setting thread-affinity for > 64 CPUs isn't supported on windows!\n");
            break;
        }
    }

    DWORD_PTR m = (DWORD_PTR)bitmask;

    m = SetThreadAffinityMask(h, m);

    return m != 0;
}

static bool ggml_thread_apply_priority(int32_t prio) {
    // Note that on Windows the Process Priority Class must be updated in order to set Thread priority.
    // This is up to the applications.
    DWORD p = THREAD_PRIORITY_NORMAL;
    switch (prio) {
        case GGML_SCHED_PRIO_LOW:      p = THREAD_PRIORITY_BELOW_NORMAL;  break;
        case GGML_SCHED_PRIO_NORMAL:   p = THREAD_PRIORITY_NORMAL;        break;
        case GGML_SCHED_PRIO_MEDIUM:   p = THREAD_PRIORITY_ABOVE_NORMAL;  break;
        case GGML_SCHED_PRIO_HIGH:     p = THREAD_PRIORITY_HIGHEST;       break;
        case GGML_SCHED_PRIO_REALTIME: p = THREAD_PRIORITY_TIME_CRITICAL; break;
    }

    if (prio != GGML_SCHED_PRIO_LOW) {
        // Tell Windows that this thread should not be throttled (needs its own CPU core).
        // Newer Windows 11 versions aggressively park (offline) CPU cores and often place
        // all our threads onto the first 4 cores which results in terrible performance with
        // n_threads > 4
        #if _WIN32_WINNT >= 0x0602
        THREAD_POWER_THROTTLING_STATE t;
        ZeroMemory(&t, sizeof(t));
        t.Version     = THREAD_POWER_THROTTLING_CURRENT_VERSION;
        t.ControlMask = THREAD_POWER_THROTTLING_EXECUTION_SPEED;
        t.StateMask   = 0;

        if (!SetThreadInformation(GetCurrentThread(), ThreadPowerThrottling, &t, sizeof(t))) {
            GGML_LOG_DEBUG("failed to disable thread power throttling %d : (%d)\n", prio, (int) GetLastError());
            return false;
        }
        #endif
    }

    if (prio == GGML_SCHED_PRIO_NORMAL) {
        // Keep inherited policy/priority
        return true;
    }

    if (!SetThreadPriority(GetCurrentThread(), p)) {
        fprintf(stderr, "warn: failed to set thread priority %d : (%d)\n", prio, (int) GetLastError());
        return false;
    }

    return true;
}

#elif defined(__APPLE__)
#include <sys/types.h>
#include <sys/resource.h>

static bool ggml_thread_apply_affinity(const bool * mask) {
    // Not supported on Apple platforms
    UNUSED(mask);
    return true;
}

static bool ggml_thread_apply_priority(int32_t prio) {
    struct sched_param p;
    int32_t policy = SCHED_OTHER;
    switch (prio) {
        // TODO: there seems to be no way to set lower prio on Apple platforms
        case GGML_SCHED_PRIO_LOW:      policy = SCHED_OTHER; p.sched_priority = 0;  break;
        case GGML_SCHED_PRIO_NORMAL:   policy = SCHED_OTHER; p.sched_priority = 0;  break;
        case GGML_SCHED_PRIO_MEDIUM:   policy = SCHED_FIFO;  p.sched_priority = 40; break;
        case GGML_SCHED_PRIO_HIGH:     policy = SCHED_FIFO;  p.sched_priority = 80; break;
        case GGML_SCHED_PRIO_REALTIME: policy = SCHED_FIFO;  p.sched_priority = 90; break;
    }

    if (prio == GGML_SCHED_PRIO_NORMAL) {
        // Keep inherited policy/priority
        return true;
    }

    int32_t err = pthread_setschedparam(pthread_self(), policy, &p);
    if (err != 0) {
        fprintf(stderr, "warn: failed to set thread priority %d : %s (%d)\n", prio, strerror(err), err);
        return false;
    }

    return true;
}

#elif defined(__linux__)
// TODO: this may not work on BSD, to be verified

static bool ggml_thread_apply_affinity(const bool * mask) {
    cpu_set_t cpuset;
    int err;

    CPU_ZERO(&cpuset);

    for (uint32_t i = 0; i < GGML_MAX_N_THREADS; i++) {
        if (mask[i]) {
            GGML_PRINT_DEBUG("Thread %lx: adding %d to cpuset\n", pthread_self(), i);
            CPU_SET(i, &cpuset);
        }
    }

#ifdef __ANDROID__
    err = sched_setaffinity(0, sizeof(cpuset), &cpuset);
    if (err < 0) {
        err = errno;
    }
#else
    err = pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);
#endif
    if (err != 0) {
        fprintf(stderr, "warn: failed to set affinity mask 0x%llx : %s (%d)\n", (unsigned long long)mask, strerror(err), err);
        return false;
    }

    return true;
}

static bool ggml_thread_apply_priority(int32_t prio) {
    struct sched_param p;
    int32_t policy = SCHED_OTHER;
    switch (prio) {
        case GGML_SCHED_PRIO_LOW:      policy = SCHED_BATCH; p.sched_priority = 0;  break;
        case GGML_SCHED_PRIO_NORMAL:   policy = SCHED_OTHER; p.sched_priority = 0;  break;
        case GGML_SCHED_PRIO_MEDIUM:   policy = SCHED_FIFO;  p.sched_priority = 40; break;
        case GGML_SCHED_PRIO_HIGH:     policy = SCHED_FIFO;  p.sched_priority = 80; break;
        case GGML_SCHED_PRIO_REALTIME: policy = SCHED_FIFO;  p.sched_priority = 90; break;
    }

    if (prio == GGML_SCHED_PRIO_NORMAL) {
        // Keep inherited policy/priority
        return true;
    }

    int32_t err = pthread_setschedparam(pthread_self(), policy, &p);
    if (err != 0) {
        fprintf(stderr, "warn: failed to set thread priority %d : %s (%d)\n", prio, strerror(err), err);
        return false;
    }

    return true;
}

#else // unsupported platforms

static bool ggml_thread_apply_affinity(const bool * mask) {
    UNUSED(mask);
    return true;
}

static bool ggml_thread_apply_priority(int32_t prio) {
    UNUSED(prio);
    return true;
}

#endif

static bool ggml_thread_cpumask_is_valid(const bool * mask) {
    for (int i = 0; i < GGML_MAX_N_THREADS; i++) {
        if (mask[i]) { return true; }
    }
    return false;
}

static void ggml_thread_cpumask_next(const bool * global_mask, bool * local_mask, bool strict, int32_t* iter) {
    if (!strict) {
        memcpy(local_mask, global_mask, GGML_MAX_N_THREADS);
        return;
    } else {
        memset(local_mask, 0, GGML_MAX_N_THREADS);
        int32_t base_idx = *iter;
        for (int32_t i = 0; i < GGML_MAX_N_THREADS; i++) {
            int32_t idx = base_idx + i;
            if (idx >= GGML_MAX_N_THREADS) {
                // Just a cheaper modulo
                idx -= GGML_MAX_N_THREADS;
            }
            if (global_mask[idx]) {
                local_mask[idx] = 1;
                *iter = idx + 1;
                return;
            }
        }
    }
}

void ggml_threadpool_free(struct ggml_threadpool* threadpool) {
    if (!threadpool) return;

    const int n_threads = threadpool->n_threads;

#ifndef GGML_USE_OPENMP
    struct ggml_compute_state* workers = threadpool->workers;

    ggml_mutex_lock(&threadpool->mutex);

    threadpool->stop = true;
    threadpool->pause = false;

    ggml_cond_broadcast(&threadpool->cond);
    ggml_mutex_unlock(&threadpool->mutex);

    for (int j = 1; j < n_threads; j++) {
        int32_t rc = ggml_thread_join(workers[j].thrd, NULL);
        GGML_ASSERT(rc == GGML_EXIT_SUCCESS || rc == GGML_EXIT_ABORTED);
        UNUSED(rc);
    }

    ggml_mutex_destroy(&threadpool->mutex);
    ggml_cond_destroy(&threadpool->cond);
#endif // GGML_USE_OPENMP

    const size_t workers_size = sizeof(struct ggml_compute_state) * n_threads;
    ggml_aligned_free(threadpool->workers, workers_size);
    ggml_aligned_free(threadpool, sizeof(struct ggml_threadpool));
}

#ifndef GGML_USE_OPENMP
// pause/resume must be called under mutex
static void ggml_threadpool_pause_locked(struct ggml_threadpool * threadpool) {
    GGML_PRINT_DEBUG("Pausing threadpool\n");
    threadpool->pause = true;
    ggml_cond_broadcast(&threadpool->cond);
}

static void ggml_threadpool_resume_locked(struct ggml_threadpool * threadpool) {
    GGML_PRINT_DEBUG("Resuming threadpool\n");
    threadpool->pause = false;
    ggml_cond_broadcast(&threadpool->cond);
}
#endif

void ggml_threadpool_pause(struct ggml_threadpool * threadpool) {
#ifndef GGML_USE_OPENMP
    ggml_mutex_lock(&threadpool->mutex);
    if (!threadpool->pause) {
       ggml_threadpool_pause_locked(threadpool);
    }
    ggml_mutex_unlock(&threadpool->mutex);
#else
    UNUSED(threadpool);
#endif
}

void ggml_threadpool_resume(struct ggml_threadpool * threadpool) {
#ifndef GGML_USE_OPENMP
    ggml_mutex_lock(&threadpool->mutex);
    if (threadpool->pause) {
       ggml_threadpool_resume_locked(threadpool);
    }
    ggml_mutex_unlock(&threadpool->mutex);
#else
    UNUSED(threadpool);
#endif
}

struct ggml_cplan ggml_graph_plan(
          const struct ggml_cgraph * cgraph,
                               int   n_threads,
            struct ggml_threadpool * threadpool) {

    if (threadpool == NULL) {
        //GGML_PRINT_DEBUG("Threadpool is not specified. Will create a disposable threadpool : n_threads %d\n", n_threads);
    }
    if (n_threads <= 0) {
        n_threads = threadpool ? threadpool->n_threads : GGML_DEFAULT_N_THREADS;
    }

#if defined(__EMSCRIPTEN__) && !defined(__EMSCRIPTEN_PTHREADS__)
    // Emscripten without pthreads support can only use a single thread
    n_threads = 1;
#endif

#if defined(__wasi__)
    // WASI doesn't support parallelism yet
    n_threads = 1;
#endif

    size_t work_size = 0;

    struct ggml_cplan cplan;
    memset(&cplan, 0, sizeof(struct ggml_cplan));

    int max_tasks = 1;

    // thread scheduling for the different operations + work buffer size estimation
    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * node = cgraph->nodes[i];

        const int n_tasks = ggml_get_n_tasks(node, n_threads);

        max_tasks = MAX(max_tasks, n_tasks);

        size_t cur = 0;

        if (!ggml_cpu_extra_work_size(n_threads, node, &cur)) {
            switch (node->op) {
                case GGML_OP_CPY:
                case GGML_OP_DUP:
                    {
                        if (ggml_is_quantized(node->type) ||
                            // F16 -> BF16 and BF16 -> F16 copies go through intermediate F32
                            (node->src[0]->type == GGML_TYPE_F16  && node->src[1] && node->src[1]->type == GGML_TYPE_BF16) ||
                            (node->src[0]->type == GGML_TYPE_BF16 && node->src[1] && node->src[1]->type == GGML_TYPE_F16) ||
                            // conversion between F32 and I32
                            (node->src[0]->type == GGML_TYPE_F32 && node->src[1] && node->src[1]->type == GGML_TYPE_I32) ||
                            (node->src[0]->type == GGML_TYPE_I32 && node->src[1] && node->src[1]->type == GGML_TYPE_F32)) {
                            cur = ggml_type_size(GGML_TYPE_F32) * node->ne[0] * n_tasks;
                        }
                    } break;
                case GGML_OP_ADD:
                case GGML_OP_ADD_ID:
                case GGML_OP_ADD1:
                    {
                        if (ggml_is_quantized(node->src[0]->type)) {
                            cur = ggml_type_size(GGML_TYPE_F32) * node->src[0]->ne[0] * n_tasks;
                        }
                    } break;
                case GGML_OP_ACC:
                    {
                        if (ggml_is_quantized(node->src[0]->type)) {
                            cur = ggml_type_size(GGML_TYPE_F32) * node->src[1]->ne[0] * n_tasks;
                        }
                    } break;
                case GGML_OP_COUNT_EQUAL:
                    {
                        cur = ggml_type_size(node->type)*n_tasks;
                    } break;
                case GGML_OP_MUL_MAT:
                    {
                        const enum ggml_type vec_dot_type = type_traits_cpu[node->src[0]->type].vec_dot_type;

                        if (node->src[1]->type != vec_dot_type) {
                            cur = ggml_row_size(vec_dot_type, ggml_nelements(node->src[1]));
                        }
                    } break;
                case GGML_OP_MUL_MAT_ID:
                    {
                        cur = 0;
                        const struct ggml_tensor * src0 = node->src[0];
                        const struct ggml_tensor * src1 = node->src[1];
                        const struct ggml_tensor * ids = node->src[2];
                        const enum ggml_type vec_dot_type = type_traits_cpu[src0->type].vec_dot_type;
                        const int n_as = src0->ne[2];
                        // src1
                        if (src1->type != vec_dot_type) {
                            cur += ggml_row_size(vec_dot_type, ggml_nelements(src1)) + sizeof(int64_t);
                        }
                        // matrix_row_counts
                        cur += n_as * sizeof(int64_t) + sizeof(int64_t);
                        // matrix_rows
                        cur += n_as*ids->ne[0]*ids->ne[1]*sizeof(struct mmid_row_mapping) + sizeof(int64_t);
                        // atomic_current_chunk
                        cur += CACHE_LINE_SIZE*n_as + CACHE_LINE_SIZE;
                    } break;
                case GGML_OP_OUT_PROD:
                    {
                        if (ggml_is_quantized(node->src[0]->type) ||
                            node->src[0]->type == GGML_TYPE_F16) {
                            cur = ggml_type_size(GGML_TYPE_F32) * node->src[0]->ne[0] * n_tasks;
                        }
                    } break;
                case GGML_OP_SET_ROWS:
                    {
                        if (node->src[0]->type == GGML_TYPE_F16 && node->type != GGML_TYPE_F16) {
                            cur = ggml_type_size(GGML_TYPE_F32) * node->src[0]->ne[0] * n_tasks;
                        }
                    } break;
                case GGML_OP_SOFT_MAX:
                case GGML_OP_ROPE:
                case GGML_OP_ROPE_BACK:
                    {
                        cur = ggml_type_size(GGML_TYPE_F32) * node->ne[0] * n_tasks;
                    } break;
                case GGML_OP_CONV_TRANSPOSE_1D:
                    {
                        GGML_ASSERT(node->src[0]->ne[3] == 1);
                        GGML_ASSERT(node->src[1]->ne[2] == 1);
                        GGML_ASSERT(node->src[1]->ne[3] == 1);

                        const int64_t ne00 = node->src[0]->ne[0];  // K
                        const int64_t ne01 = node->src[0]->ne[1];  // Cout
                        const int64_t ne02 = node->src[0]->ne[2];  // Cin
                        const int64_t ne10 = node->src[1]->ne[0];  // L
                        const int64_t ne11 = node->src[1]->ne[1];  // Cin

                        if ((node->src[0]->type == GGML_TYPE_F16 ||
                             node->src[0]->type == GGML_TYPE_BF16) &&
                            node->src[1]->type == GGML_TYPE_F32) {
                            cur += sizeof(ggml_fp16_t)*ne00*ne01*ne02;
                            cur += sizeof(ggml_fp16_t)*ne10*ne11;
                        } else if (node->src[0]->type == GGML_TYPE_F32 &&
                                   node->src[1]->type == GGML_TYPE_F32) {
                            cur += sizeof(float)*ne00*ne01*ne02;
                            cur += sizeof(float)*ne10*ne11;
                        } else {
                            GGML_ABORT("fatal error");
                        }
                    } break;
                case GGML_OP_CONV_2D:
                case GGML_OP_CONV_3D:
                    {
                        cur = GGML_IM2COL_WORK_SIZE;
                    } break;
                case GGML_OP_CONV_TRANSPOSE_2D:
                    {
                        const int64_t ne00 = node->src[0]->ne[0]; // W
                        const int64_t ne01 = node->src[0]->ne[1]; // H
                        const int64_t ne02 = node->src[0]->ne[2]; // Channels Out
                        const int64_t ne03 = node->src[0]->ne[3]; // Channels In

                        const int64_t ne10 = node->src[1]->ne[0]; // W
                        const int64_t ne11 = node->src[1]->ne[1]; // H
                        const int64_t ne12 = node->src[1]->ne[2]; // Channels In

                        GGML_ASSERT(node->src[0]->type == GGML_TYPE_F16 || node->src[0]->type == GGML_TYPE_F32);
                        GGML_ASSERT(node->src[1]->type == GGML_TYPE_F32);

                        cur += ggml_type_size(node->src[0]->type) * ne00 * ne01 * ne02 * ne03;
                        cur += ggml_type_size(node->src[0]->type) * ne10 * ne11 * ne12;

                    } break;
                case GGML_OP_TOP_K:
                    {
                        cur += sizeof(int32_t)*node->src[0]->ne[0]*n_tasks;
                    } break;
                case GGML_OP_FLASH_ATTN_EXT:
                    {
                        const int64_t neq2 = node->src[0]->ne[2]; // number of query heads
                        const int64_t DK = node->src[1]->ne[0];
                        const int64_t DV = node->src[2]->ne[0];

                        // Tiled flash attention scratch (tile sizes defined in common.h)
                        // Per-thread: Q_q + KQ + mask + VKQ32 + V32 + K_f32 + padding
                        size_t prefill  = sizeof(float)*(GGML_FA_TILE_Q*DK + 2*GGML_FA_TILE_Q*GGML_FA_TILE_KV + GGML_FA_TILE_Q*DV + GGML_FA_TILE_KV*DV + GGML_FA_TILE_KV*DK)*n_tasks;

                        // Decode path: n_kv_chunks = n_tasks (one chunk per thread)
                        // Per-thread: VKQ accmulator (DV), partial M, partial S + intra-thread scratch for V, Q and VKQ
                        size_t n_chunks = n_tasks;
                        size_t decode   = sizeof(float)*(neq2*n_chunks*(2+DV) + n_tasks*(DK + 2*DV));

                        cur += MAX(prefill, decode);
                    } break;
                case GGML_OP_FLASH_ATTN_BACK:
                    {
                        const int64_t    D = node->src[0]->ne[0];
                        const int64_t ne11 = ggml_up(node->src[1]->ne[1], GGML_SOFT_MAX_UNROLL);
                        const int64_t mxDn = MAX(D, ne11) * 2; // *2 because of S and SM in ggml_compute_forward_flash_attn_back
                        if (node->src[1]->type == GGML_TYPE_F32) {
                            cur  = sizeof(float)*mxDn*n_tasks; // TODO: this can become (n_tasks-1)
                            cur += sizeof(float)*mxDn*n_tasks; // this is overestimated by x2
                        } else if (node->src[1]->type == GGML_TYPE_F16) {
                            cur  = sizeof(float)*mxDn*n_tasks; // TODO: this can become (n_tasks-1)
                            cur += sizeof(float)*mxDn*n_tasks; // this is overestimated by x2
                        } else if (node->src[1]->type == GGML_TYPE_BF16) {
                            cur  = sizeof(float)*mxDn*n_tasks; // TODO: this can become (n_tasks-1)
                            cur += sizeof(float)*mxDn*n_tasks; // this is overestimated by x2
                        }
                    } break;

                case GGML_OP_CROSS_ENTROPY_LOSS:
                    {
                        cur = ggml_type_size(node->type)*(n_tasks + node->src[0]->ne[0]*n_tasks);
                    } break;
                case GGML_OP_GATED_DELTA_NET:
                    {
                        const int64_t S_v = node->src[2]->ne[0];
                        const int64_t K   = ggml_get_op_params_i32(node, 0);
                        const int64_t per_thread = S_v + (K > 1 ? S_v * S_v : 0);
                        cur = per_thread * sizeof(float) * n_tasks;
                    } break;
                case GGML_OP_COUNT:
                    {
                        GGML_ABORT("fatal error");
                    }
                case GGML_OP_LIGHTNING_INDEXER:
                    {
                        // temp buffer for dequantizing lightning indexer keys
                        const int64_t ne10 = node->src[1]->ne[0];
                        cur += sizeof(float)*ne10*n_tasks;
                    } break;
                default:
                    break;
            }
        }

        work_size = MAX(work_size, cur);
    }

    if (work_size > 0) {
        work_size += CACHE_LINE_SIZE*(n_threads);
    }

    cplan.threadpool = threadpool;
    cplan.n_threads  = MIN(max_tasks, n_threads);
    cplan.work_size  = work_size;
    cplan.work_data  = NULL;

    return cplan;
}


// ---- fused RoPE diamond {MUL, GET_ROWS, MUL, ADD} ------------------------------------------------
//
// STATUS: EXPERIMENTAL, DEFAULT OFF (KATAGO_CPU_ROPE_FUSE=1 to enable). Read ROPE-FUSION-OUTCOME.md in
// the workspace before touching this. Summary of what was measured before it was parked:
//   * the predicate and the kernel are CORRECT -- with KATAGO_CPU_ROPE_NOSKIP=1 (fused kernel runs but
//     the three nodes are still executed afterwards) the network output is BIT-IDENTICAL to the baseline,
//     and an in-kernel audit that recomputes every element with the stock addressing finds 0 mismatches
//     across all 40 sites;
//   * the BARRIER LEDGER IMPROVES EXACTLY AS DESIGNED -- 10240 fewer barriers per 8 graph walks
//     (= 40 sites x 3 nodes x 8), 6392 -> 5112 barrier points per walk;
//   * but with the skip ENABLED the network output is wrong, deterministically, even for a single fused
//     site, and it stays wrong with KATAGO_CPU_NOALLOC=1 (every tensor separately allocated, so it is not
//     gallocr aliasing) and at one thread (so it is not a race).
// So the remaining problem is neither the arithmetic nor a data race: it is that having the executor skip
// nodes is not something the graph/plan layer accounts for. The fix belongs at graph-emission time, not in
// this loop -- see the document.
//
// The graph contains, 40 times per forward, the rotation pair that RoPE attention writes out as four
// separate nodes:
//     n0 MUL      <- src0 = RESHAPE(#44), src1 = table A
//     n1 GET_ROWS <- src0 = RESHAPE(#44), src1 = i32 indices
//     n2 MUL      <- src0 = n1,          src1 = table B
//     n3 ADD      <- src0 = n0,          src1 = n2
// That is a DIAMOND, not a chain: ggml_can_fuse rejects it because it requires every node to depend
// directly on its predecessor (ggml-impl.h:687), whereas ADD depends on n0 rather than on n1. The
// predicate therefore has to be hand-written, and it has to step over the free layout op at the head.
//
// Why it would be worth doing at all: it removes three non-empty nodes per site, hence three barriers per
// site per graph walk. Measured, the barrier cost is proportional to NODE COUNT rather than to data volume
// (BARRIER-SPIN-QUANTIFIED.md), so the win would be the ledger, not the saved loads.

// Fusable only if used exactly once, and not a view or a requested output. Mirrors ggml_node_has_n_uses
// (ggml-impl.h:641), which is not used directly because the count itself is wanted, so it can be cached.
static int ggml_cpu_rope_uses_once(const struct ggml_cgraph * cgraph, const struct ggml_tensor * t) {
    if (t->view_src != NULL) {
        return 0;
    }
    if ((t->flags & GGML_TENSOR_FLAG_OUTPUT) != 0) {
        return 0;
    }

    int32_t n_uses = 0;
    if (cgraph->use_counts != NULL) {
        const size_t hash_pos = ggml_hash_find(&cgraph->visited_hash_set, t);
        if (ggml_bitset_get(cgraph->visited_hash_set.used, hash_pos)) {
            n_uses = cgraph->use_counts[hash_pos];
        }
    }
    return n_uses == 1;
}

// Operand checks shared by both chain forms. `add` is the chain tail, `a` the first MUL, `b` the second.
// Both MULs must read the same head through their src0, and multiply by a table of the same full shape.
static int ggml_cpu_rope_operands_ok(const struct ggml_tensor * add, const struct ggml_tensor * a,
                                     const struct ggml_tensor * b) {
    if (add->type != GGML_TYPE_F32 || a->type != GGML_TYPE_F32 || b->type != GGML_TYPE_F32) {
        return 0;
    }
    if (a->src[0]->type != GGML_TYPE_F32 || a->src[1]->type != GGML_TYPE_F32 || b->src[1]->type != GGML_TYPE_F32) {
        return 0;
    }
    if (!ggml_are_same_shape(add, a) || !ggml_are_same_shape(add, b)) {
        return 0;
    }
    if (!ggml_are_same_shape(a->src[0], add) || !ggml_are_same_shape(a->src[1], add) ||
        !ggml_are_same_shape(b->src[0], add) || !ggml_are_same_shape(b->src[1], add)) {
        return 0;
    }
    // Contiguity is what lets the fused kernel stand in for apply_binary_op's broadcast path and lets the
    // table rows be addressed with the head's own strides.
    if (!ggml_is_contiguous(add) || !ggml_is_contiguous(a) || !ggml_is_contiguous(b) ||
        !ggml_is_contiguous(a->src[0]) || !ggml_is_contiguous(a->src[1]) || !ggml_is_contiguous(b->src[1])) {
        return 0;
    }
    return 1;
}

// The gather half of the four-node form.
static int ggml_cpu_rope_gather_ok(const struct ggml_tensor * add, const struct ggml_tensor * a,
                                   const struct ggml_tensor * gr, const struct ggml_tensor * b) {
    if (gr->op != GGML_OP_GET_ROWS) {
        return 0;
    }
    if (gr->type != GGML_TYPE_F32 || gr->src[0]->type != GGML_TYPE_F32 || gr->src[1]->type != GGML_TYPE_I32) {
        return 0;
    }
    if (a->src[0] != gr->src[0]) {   // diamond: both MULs read the same operand
        return 0;
    }
    if (b->src[0] != gr) {           // and the second table multiplies the gathered rows
        return 0;
    }
    if (!ggml_are_same_shape(gr, add) || !ggml_are_same_shape(gr->src[0], add)) {
        return 0;
    }
    // ggml_compute_forward_get_rows_f32 requires ggml_nrows(dst) == ggml_nelements(src1) and walks the
    // indices with nb10/nb11/nb12, so the index tensor must be contiguous in rows for the fused index
    // arithmetic to match. Element-contiguity alone is not sufficient here.
    if (!ggml_is_contiguous(gr) || !ggml_is_contiguous_rows(gr->src[1])) {
        return 0;
    }
    return 1;
}

// One-pass cache of "used exactly once" for every node of the graph being walked. Without it the
// predicate would do a hash lookup per node per walk, i.e. millions per forward pass.
//
// The type is deliberately not named ggml_cpu_rope_use_counts: in C a typedef name and a function name
// share the ordinary identifier namespace, so it would collide with the accessor defined below.
typedef struct {
    int *  used_once;
    size_t n_used;
    int    total;
    int    last_hit;
} ggml_cpu_rope_uses;

static ggml_cpu_rope_uses * ggml_cpu_rope_uses_cache              = NULL;
static const struct ggml_cgraph * ggml_cpu_rope_uses_cache_graph  = NULL;
static int ggml_cpu_rope_uses_lock = 0;

// Every compute thread reaches this, so the once-per-graph scan has to be serialised. An earlier version
// was not, and it segfaulted with 8 threads: one thread published the "graph cached" marker after
// allocating the array but before filling it, so the other seven read uninitialised -- or already freed
// -- memory. The release/acquire pair below is what makes the array contents visible to the others.
static ggml_cpu_rope_uses * ggml_cpu_rope_use_counts(const struct ggml_cgraph * cgraph) {
    if (__atomic_load_n(&ggml_cpu_rope_uses_cache_graph, __ATOMIC_ACQUIRE) == cgraph) {
        return ggml_cpu_rope_uses_cache;
    }

    while (__atomic_exchange_n(&ggml_cpu_rope_uses_lock, 1, __ATOMIC_ACQUIRE) != 0) {
        // another thread is running the scan; it is short
    }

    if (ggml_cpu_rope_uses_cache_graph != cgraph) {
        const size_t n_nodes = (size_t) cgraph->n_nodes;

        free(ggml_cpu_rope_uses_cache);
        ggml_cpu_rope_uses_cache = (ggml_cpu_rope_uses *) calloc(1, sizeof(ggml_cpu_rope_uses));
        if (ggml_cpu_rope_uses_cache != NULL) {
            ggml_cpu_rope_uses_cache->used_once = (int *) calloc(n_nodes > 0 ? n_nodes : 1, sizeof(int));
            if (ggml_cpu_rope_uses_cache->used_once == NULL) {
                free(ggml_cpu_rope_uses_cache);
                ggml_cpu_rope_uses_cache = NULL;
            } else {
                ggml_cpu_rope_uses_cache->n_used = n_nodes;
                for (size_t i = 0; i < n_nodes; i++) {
                    ggml_cpu_rope_uses_cache->used_once[i] =
                        ggml_cpu_rope_uses_once(cgraph, cgraph->nodes[i]);
                }
            }
        }

        __atomic_store_n(&ggml_cpu_rope_uses_cache_graph, cgraph, __ATOMIC_RELEASE);
    }

    __atomic_store_n(&ggml_cpu_rope_uses_lock, 0, __ATOMIC_RELEASE);
    return ggml_cpu_rope_uses_cache;
}

static int ggml_cpu_rope_uses_ok(const ggml_cpu_rope_uses * cache, const struct ggml_cgraph * cgraph, int node_n) {
    if (node_n < 0 || node_n >= cgraph->n_nodes) {
        return 0;
    }
    if (cache == NULL) {
        return ggml_cpu_rope_uses_once(cgraph, cgraph->nodes[node_n]);
    }
    return cache->used_once[node_n];
}

// Diagnostics, both off by default and never reached in a production run:
//   KATAGO_CPU_BARRIER_STATS=1  one summary line per graph, so a cross-machine csv carries its own evidence
//   KATAGO_CPU_ROPE_TRACE=1     one line per match; =2 also explains each rejected candidate. That is what
//                               tells "the predicate never fired" apart from "the fusion did not help".
static int ggml_cpu_rope_trace = -1;
static int ggml_cpu_rope_only  = -2;

// KATAGO_CPU_ROPE_ONLY=<node index> fuses a single matching site and leaves the rest alone. This is the
// bisection tool: fusing exactly one site answers "is the fused arithmetic right" without the answer being
// muddied by 40 sites or by downstream rearrangement.
static int ggml_cpu_rope_only_node(void) {
    if (ggml_cpu_rope_only < -1) {
        const char * e = getenv("KATAGO_CPU_ROPE_ONLY");
        ggml_cpu_rope_only = (e != NULL && e[0] != '\0') ? atoi(e) : -1;
    }
    return ggml_cpu_rope_only;
}

static int ggml_cpu_rope_trace_on(void) {
    if (ggml_cpu_rope_trace < 0) {
        const char * e = getenv("KATAGO_CPU_ROPE_TRACE");
        ggml_cpu_rope_trace = (e != NULL) ? atoi(e) : 0;
    }
    return ggml_cpu_rope_trace;
}

static void ggml_cpu_rope_note_hit(ggml_cpu_rope_uses * cache, const char * form, int node_n) {
    if (cache == NULL) {
        return;
    }
    if (ggml_cpu_rope_trace_on() >= 1) {
        fprintf(stderr, "[rope-fuse] hit %s node=%d n_nodes=%zu\n", form, node_n, cache->n_used);
    }
    if (node_n > cache->last_hit) {
        cache->last_hit = node_n;
        cache->total++;
    }
}

// One-shot structural census of the {MUL, GET_ROWS} candidates, printed straight to stderr.
// KATAGO_CPU_ROPE_DUMP=1. Purpose: the executable's own graph dump goes through KataGo's Logger and lands
// in gtp_logs/, which makes it awkward to line up with what the predicate actually sees. This prints the
// exact tensor identities the predicate compares, plus the free-view chain above the head, so a rejected
// candidate can be diagnosed from the backend itself.
static void ggml_cpu_rope_dump_census(const struct ggml_cgraph * cgraph, int thread_n) {
    static int done = 0;
    const char * env = getenv("KATAGO_CPU_ROPE_DUMP");
    if (env == NULL || env[0] != '1' || thread_n != 0 || done) {
        return;
    }
    done = 1;

    fprintf(stderr, "[rope-dump] n_nodes=%d\n", cgraph->n_nodes);
    // Full node census, KATAGO_CPU_ROPE_DUMP=full: one line per node in the same format the executable's
    // own dump uses, so the union table can be computed for the PRODUCTION shape (the Logging-hook dump
    // lands in gtp_logs/ and is awkward to line up, and the old _rope.txt is a different model).
    {
        const char * full = getenv("KATAGO_CPU_ROPE_DUMP");
        if (full != NULL && full[0] == 'f') {
            for (int i = 0; i < cgraph->n_nodes; i++) {
                const struct ggml_tensor * t = cgraph->nodes[i];
                const char * s0 = t->src[0] ? ggml_op_name(t->src[0]->op) : "NULL";
                const char * s1 = t->src[1] ? ggml_op_name(t->src[1]->op) : "NULL";
                fprintf(stderr, ": CPU graph node %d %s type=%s ne=[%lld,%lld,%lld,%lld] nb=[%lld,%lld,%lld,%lld] contig=%d src0=%s src1=%s\n",
                        i, ggml_op_name(t->op), ggml_type_name(t->type),
                        (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3],
                        (long long) t->nb[0], (long long) t->nb[1], (long long) t->nb[2], (long long) t->nb[3],
                        (int) ggml_is_contiguous(t), s0, s1);
            }
            return;
        }
    }
    for (int i = 0; i + 1 < cgraph->n_nodes; i++) {
        const struct ggml_tensor * n1 = cgraph->nodes[i];
        const struct ggml_tensor * g  = cgraph->nodes[i + 1];
        if (n1->op != GGML_OP_MUL || g->op != GGML_OP_GET_ROWS) {
            continue;
        }
        fprintf(stderr, "[rope-dump] MUL@%d src0=#%d(%s) src1=#%d | GR@%d src0=#%d(%s) src1=#%d(%s)\n",
                i, (int) n1->src[0]->op, ggml_op_name(n1->src[0]->op),
                (int) n1->src[1]->op, i + 1,
                (int) g->src[0]->op, ggml_op_name(g->src[0]->op),
                (int) g->src[1]->op, ggml_type_name(g->src[1]->type));
        // Walk the free-view chain above the MUL's src0, printing how far it goes.
        const struct ggml_tensor * t = n1->src[0];
        for (int d = 0; d < 6 && t != NULL; d++) {
            fprintf(stderr, "[rope-dump]    ^%d %s type=%s ne=[%lld,%lld,%lld,%lld] nb=[%lld,%lld,%lld,%lld] contig=%d src0=%s\n",
                    d, ggml_op_name(t->op), ggml_type_name(t->type),
                    (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3],
                    (long long) t->nb[0], (long long) t->nb[1], (long long) t->nb[2], (long long) t->nb[3],
                    (int) ggml_is_contiguous(t), t->src[0] ? ggml_op_name(t->src[0]->op) : "NULL");
            t = t->src[0];
        }
        // The identity the predicate compares: does the MUL's src0 reach a RESHAPE/VIEW whose own src0 is
        // the same tensor the gather reads?
        fprintf(stderr, "[rope-dump]    diamond: mul=%p mul..src0=%p gr..src0=%p head_candidate=%p\n",
                (const void *) n1, (const void *) n1->src[0], (const void *) g->src[0],
                (const void *) ((n1->src[0]->op == GGML_OP_RESHAPE || n1->src[0]->op == GGML_OP_VIEW ||
                                 n1->src[0]->op == GGML_OP_PERMUTE || n1->src[0]->op == GGML_OP_TRANSPOSE)
                                && n1->src[0]->src[0] != NULL ? n1->src[0]->src[0] : n1->src[0]));
        fprintf(stderr, "[rope-dump]    equal_direct=%d equal_head_vs_grsrc0=%d\n",
                n1->src[0] == g->src[0],
                (((n1->src[0]->op == GGML_OP_RESHAPE || n1->src[0]->op == GGML_OP_VIEW ||
                   n1->src[0]->op == GGML_OP_PERMUTE || n1->src[0]->op == GGML_OP_TRANSPOSE)
                  && n1->src[0]->src[0] != NULL ? n1->src[0]->src[0] : n1->src[0]) == g->src[0]));
    }
}

// Defined in ops.cpp (which owns the fused kernel these hooks cross-check).
extern void (*ggml_cpu_rope_dbg_store)(uint64_t h);
extern int  (*ggml_cpu_rope_dbg_check)(uint64_t h);

// Debug-only cross-check state, KATAGO_CPU_ROPE_HASH=1. Two runs are used on purpose://   run 1: KATAGO_CPU_ROPE_HASH=1 KATAGO_CPU_ROPE_ONCE=1  -> exactly one rope site is fused, the fused
//          kernel hashes the block it wrote, and the hook stores it (the stock chain never runs for it,
//          so nothing can overwrite it);
//   run 2: KATAGO_CPU_ROPE_HASH=1 KATAGO_CPU_ROPE_OFF=1   -> no fusion; every ADD hashes its output and
//          the hook compares against the stored value.
// Same-process comparison, so this is immune to ggml_gallocr assigning buffers differently across runs.
static uint64_t ggml_cpu_rope_hash_stored = 0;
static int      ggml_cpu_rope_hash_have   = 0;

static void ggml_cpu_rope_dbg_store_fn(uint64_t h) {
    ggml_cpu_rope_hash_stored = h;
    ggml_cpu_rope_hash_have   = 1;
    fprintf(stderr, "[rope-hash] FUSED  h=%016llx\n", (unsigned long long) h);
}

static int ggml_cpu_rope_dbg_check_fn(uint64_t h) {
    if (!ggml_cpu_rope_hash_have) {
        return 1;
    }
    const int same = (h == ggml_cpu_rope_hash_stored);
    static int n_rep = 0;
    if (n_rep < 3) {
        n_rep++;
        fprintf(stderr, "[rope-hash] STOCK  h=%016llx  %s (stored=%016llx)\n",
                (unsigned long long) h, same ? "MATCH" : "DIFFER",
                (unsigned long long) ggml_cpu_rope_hash_stored);
    }
    return same;
}

// Do two tensors occupy overlapping bytes of the same allocation? This is the check the Vulkan backend
// runs after deciding to fuse (GV:20349-20394, ggml_vk_tensors_overlap): a fused kernel writes the chain
// tail while reading the chain's sources, so if gallocr aliased the output onto a source then the kernel
// reads its own overwritten data. The CPU loop had no such check; KATAGO_CPU_ROPE_OVERLAP=1 reports it.
static int ggml_cpu_tensors_overlap(const struct ggml_tensor * a, const struct ggml_tensor * b) {
    if (a == b) {
        return 1;
    }
    if (a->data == NULL || b->data == NULL || a->buffer == NULL || a->buffer != b->buffer) {
        return 0;
    }

    size_t a_size = 0;
    size_t b_size = 0;
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        a_size += (size_t) (a->ne[i] - 1)*a->nb[i];
        b_size += (size_t) (b->ne[i] - 1)*b->nb[i];
    }
    a_size += ggml_type_size(a->type);
    b_size += ggml_type_size(b->type);

    const char * a0 = (const char *) a->data;
    const char * b0 = (const char *) b->data;
    return a0 < b0 + b_size && b0 < a0 + a_size;
}

// Emitted after the walk, from thread 0 only: all threads walk the same node sequence, so the thread test
// plus the last_hit latch yields exactly one line per graph.
static void ggml_cpu_rope_report_graph(ggml_cpu_rope_uses * cache, int n_nodes, int thread_n) {
    if (cache == NULL || thread_n != 0 || !ggml_cpu_barrier_stats_on) {
        return;
    }
    if (cache->total == 0 && ggml_cpu_rope_trace_on() == 0) {
        return;
    }
    fprintf(stderr, "fusion {MUL,GET_ROWS,MUL,ADD} ropeNodes=%d n_nodes=%d adjacent=1 singleUse=1\n",
            cache->total, n_nodes);
}

// Match the RoPE diamond and run the fused kernel.
// Returns the number of nodes to skip (3 for the four-node form, 2 for the three-node form), or 0.
//
// NOTE: this must NOT be called ggml_cpu_try_fuse_rope -- the wrapper below already has that name, and a
// same-named static silently makes the wrapper's own call resolve to itself.
static int ggml_cpu_rope_match(
        const struct ggml_cgraph * cgraph,
        const int node_n,
        const struct ggml_compute_params * params) {

    const int n_rest = cgraph->n_nodes - node_n;
    if (n_rest < 3) {
        return 0;
    }

    struct ggml_tensor * n0 = cgraph->nodes[node_n];

    if (n_rest < 4 || cgraph->nodes[node_n + 1]->op != GGML_OP_GET_ROWS) {
        return 0;
    }

    // The chain head is defined by what the two MULs and the gather actually READ, not by how many layout
    // ops happen to sit above them. Getting this wrong is what made the first version of this predicate
    // reject all 40 sites: it walked the head continuously up the free-view chain and landed on the CONT
    // underneath (measured chain: MUL/GET_ROWS -> RESHAPE -> CONT -> PERMUTE -> VIEW -> VIEW -> MUL_MAT),
    // while the shared operand of the three consumers is the RESHAPE one level up.
    //
    // So: the head is the tensor all three share. Stepping one further hop is allowed only when that hop is
    // itself a pure alias (RESHAPE) and is still shared by the gather -- CONT/PERMUTE are real work and
    // must never be stepped over.
    struct ggml_tensor * head = n0->src[0];
    if (head != NULL && head->op == GGML_OP_RESHAPE && head->src[0] != NULL &&
        head->src[0] == cgraph->nodes[node_n + 1]->src[0]) {
        head = head->src[0];
    }

    if (ggml_cpu_rope_trace_on() >= 2) {
        const struct ggml_tensor * g = cgraph->nodes[node_n + 1];
        fprintf(stderr, "[rope-blk] node=%d n_rest=%d mul.s0=%p gr.s0=%p head=%p eq=%d\n",
                node_n, n_rest, (const void *) n0->src[0], (const void *) g->src[0],
                (const void *) head, n0->src[0] == head);
        fprintf(stderr, "[rope-blk]   a.s0.ne=[%lld,%lld] op=%s contig=%d | g.src1.type=%s contigrows=%d\n",
                (long long) n0->src[0]->ne[0], (long long) n0->src[0]->ne[1],
                ggml_op_name(n0->src[0]->op), (int) ggml_is_contiguous(n0->src[0]),
                ggml_type_name(g->src[1]->type), (int) ggml_is_contiguous_rows(g->src[1]));
        const struct ggml_tensor * dma = cgraph->nodes[node_n];
        const struct ggml_tensor * dmb = cgraph->nodes[node_n + 2];
        const struct ggml_tensor * dad = cgraph->nodes[node_n + 3];
        fprintf(stderr, "[rope-blk]   TYPES x=%s tabA=%s tabB=%s add=%s | idx=%s | x.nb0=%lld tabA.nb0=%lld tabB.nb0=%lld add.nb0=%lld\n",
                ggml_type_name(dma->src[0]->type), ggml_type_name(dma->src[1]->type),
                ggml_type_name(dmb->src[1]->type), ggml_type_name(dad->type),
                ggml_type_name(g->src[1]->type),
                (long long) dma->src[0]->nb[0], (long long) dma->src[1]->nb[0],
                (long long) dmb->src[1]->nb[0], (long long) dad->nb[0]);
    }

    // Debug-only cross-check hooks. KATAGO_CPU_ROPE_HASH=1 activates the comparison; gated further by
    // KATAGO_CPU_ROPE_ONCE (fuse a single site and record) and KATAGO_CPU_ROPE_OFF (fuse nothing, so the
    // stock ADD hashes its own output and the hook diffs it).
    if (getenv("KATAGO_CPU_ROPE_HASH") != NULL) {
        ggml_cpu_rope_dbg_store = &ggml_cpu_rope_dbg_store_fn;
        ggml_cpu_rope_dbg_check = &ggml_cpu_rope_dbg_check_fn;
    }
    {
        static int once = -1;
        if (once < 0) {
            once = (getenv("KATAGO_CPU_ROPE_ONCE") != NULL) ? 1 : 0;
        }
        if (once) {
            static int done = 0;
            if (done) {
                return 0;
            }
            done = 1;
        }
    }
    if (getenv("KATAGO_CPU_ROPE_OFF") != NULL) {
        return 0;
    }

    ggml_cpu_rope_uses * cache = ggml_cpu_rope_use_counts(cgraph);

    // KATAGO_CPU_ROPE_ADDR=1: the address dump lives in the four-node branch below, where a/gr/b/ad are in
    // scope. It must run before the guards so it also reports the case the guard refuses.

    // --- three-node form: {MUL, MUL, ADD}. This is what N59's paired-read shape leaves behind once it has
    // removed the gather, and having it here is what keeps that change from becoming a silent regression
    // in which the fusion simply stops firing (gate 1's barrier ledger would catch it, but late). ---
    if (cgraph->nodes[node_n + 1]->op == GGML_OP_MUL && cgraph->nodes[node_n + 2]->op == GGML_OP_ADD) {
        struct ggml_tensor * a  = n0;
        struct ggml_tensor * b  = cgraph->nodes[node_n + 1];
        struct ggml_tensor * ad = cgraph->nodes[node_n + 2];

        if (a->src[0] == head && b->src[0] == head &&
            ad->src[0] == a && ad->src[1] == b &&
            ggml_cpu_rope_operands_ok(ad, a, b) &&
            ggml_cpu_rope_uses_ok(cache, cgraph, node_n) &&
            ggml_cpu_rope_uses_ok(cache, cgraph, node_n + 1)) {

            ggml_cpu_rope_note_hit(cache, "{MUL,MUL,ADD}", node_n);
            ggml_compute_forward_rope_fused(params, ad, a, b, NULL);
            return 2;
        }
    }

    // --- four-node form: {MUL, GET_ROWS, MUL, ADD} ---
    if (n_rest < 4 || cgraph->nodes[node_n + 1]->op != GGML_OP_GET_ROWS ||
        cgraph->nodes[node_n + 2]->op != GGML_OP_MUL ||
        cgraph->nodes[node_n + 3]->op != GGML_OP_ADD) {
        return 0;
    }

    struct ggml_tensor * a  = n0;
    struct ggml_tensor * gr = cgraph->nodes[node_n + 1];
    struct ggml_tensor * b  = cgraph->nodes[node_n + 2];
    struct ggml_tensor * ad = cgraph->nodes[node_n + 3];

    if (a->src[0] != head) {
        return 0;
    }
    if (!ggml_cpu_rope_operands_ok(ad, a, b)) {
        return 0;
    }
    if (!ggml_cpu_rope_gather_ok(ad, a, gr, b)) {
        return 0;
    }
    if (!ggml_cpu_rope_uses_ok(cache, cgraph, node_n)     ||
        !ggml_cpu_rope_uses_ok(cache, cgraph, node_n + 1) ||
        !ggml_cpu_rope_uses_ok(cache, cgraph, node_n + 2)) {
        return 0;
    }

    // 🔴 SAFETY GUARD. This is the CPU analogue of GV:20349-20394 + the GV:20260-20267 flag, and it is the
    // check whose ABSENCE caused the first version of this fusion to produce wrong numbers.
    //
    // Fusing means only the chain tail is written, so the three intermediate buffers become dead and
    // ggml_gallocr may hand their regions to the tail. When it does, the destination ends up overlapping
    // the region where the intermediate results (and therefore what the kernel reads) live. Measured on the
    // production shape: dst(ADD) aliases dst(MUL#0), and dst(GET_ROWS) aliases dst(MUL#1).
    //
    // Crucially the gather is a PERMUTED read (out[d] reads q[d^1]), so this is not the harmless
    // "in-place elementwise" case: the kernel reads row i after having written row i-1 of the same region.
    // Vulkan sets op_srcs_fused_elementwise to false for exactly this reason, so its overlap check refuses
    // the alias group; the CPU loop has no such stage, so the refusal has to happen here.
    //
    // KATAGO_CPU_ROPE_ADDR=1: dump every address the fused kernel would touch, for the first site. This
    // sits BEFORE the guard on purpose -- the guard refuses on this graph, so a dump placed after it can
    // never be read. The READ set is what the kernel actually loads (x, tabA, tabB, idx); the ELIDED set
    // is the three intermediate results it does NOT load, which is how a benign alias is told from a
    // harmful one.
    {
        const char * ad_env = getenv("KATAGO_CPU_ROPE_ADDR");
        static int n_addr = 0;
        if (ad_env != NULL && ad_env[0] == '1' && params->ith == 0 && n_addr < 1) {
            n_addr++;
            fprintf(stderr, "[rope-addr] node=%d nbytes(ad)=%lld nbytes(x)=%lld\n",
                    node_n, (long long) ggml_nbytes(ad), (long long) ggml_nbytes(a->src[0]));
            // Strides decide the layout question: if x has ne0 = D (the head dim) with nb0 == 4, then
            // GGML_OP_ROPE (which asserts n_dims <= ne0 and only rotates an ne0 prefix) can be used on a
            // RELABELLED VIEW with no data movement at all; if ne0 is S*N instead, it cannot.
            fprintf(stderr, "[rope-addr]   x.ne=[%lld,%lld,%lld,%lld] nb=[%lld,%lld,%lld,%lld]\n",
                    (long long) a->src[0]->ne[0], (long long) a->src[0]->ne[1],
                    (long long) a->src[0]->ne[2], (long long) a->src[0]->ne[3],
                    (long long) a->src[0]->nb[0], (long long) a->src[0]->nb[1],
                    (long long) a->src[0]->nb[2], (long long) a->src[0]->nb[3]);
            fprintf(stderr, "[rope-addr]   idx.ne=[%lld,%lld] tabA.ne=[%lld,%lld]\n",
                    (long long) gr->src[1]->ne[0], (long long) gr->src[1]->ne[1],
                    (long long) a->src[1]->ne[0], (long long) a->src[1]->ne[1]);
            fprintf(stderr, "[rope-addr]   WRITE  dst(ADD) =%p\n", (const void *) ad->data);
            fprintf(stderr, "[rope-addr]   READ   x        =%p\n", (const void *) a->src[0]->data);
            fprintf(stderr, "[rope-addr]   READ   tabA     =%p\n", (const void *) a->src[1]->data);
            fprintf(stderr, "[rope-addr]   READ   tabB     =%p\n", (const void *) b->src[1]->data);
            fprintf(stderr, "[rope-addr]   READ   idx      =%p\n", (const void *) gr->src[1]->data);
            fprintf(stderr, "[rope-addr]   ELIDED dst(MUL0)=%p\n", (const void *) a->data);
            fprintf(stderr, "[rope-addr]   ELIDED dst(GR)  =%p\n", (const void *) gr->data);
            fprintf(stderr, "[rope-addr]   ELIDED dst(MUL1)=%p\n", (const void *) b->data);
            fprintf(stderr, "[rope-addr]   OVERLAP x=%d tabA=%d tabB=%d idx=%d | MUL0=%d GR=%d MUL1=%d\n",
                    ggml_cpu_tensors_overlap(ad, a->src[0]), ggml_cpu_tensors_overlap(ad, a->src[1]),
                    ggml_cpu_tensors_overlap(ad, b->src[1]), ggml_cpu_tensors_overlap(ad, gr->src[1]),
                    ggml_cpu_tensors_overlap(ad, a), ggml_cpu_tensors_overlap(ad, gr),
                    ggml_cpu_tensors_overlap(ad, b));
        }
    }

    // LAYOUT GATE for GGML_OP_ROPE (KATAGO_CPU_ROPE_LAYOUT=1), one line per site.
    //
    // ops.cpp:6278 asserts n_dims <= ne0 and rotate_pairs only rotates an ne0 prefix, so the built-in rope
    // requires the rotated axis (the head dim D) to be (a) ne0 AND (b) stride es. That is a stronger
    // requirement than "a relabelled view exists": a relabelling can move D to ne0 but cannot change its
    // stride, so it still fails (b). This probe reports both per site so the (c) option is decided by data
    // on every site rather than by one observed sample.
    {
        const char * lay = getenv("KATAGO_CPU_ROPE_LAYOUT");
        if (lay != NULL && lay[0] == '1' && params->ith == 0) {
            const struct ggml_tensor * x = a->src[0];
            const int64_t es = (int64_t) ggml_type_size(x->type);
            fprintf(stderr, "[rope-layout] node=%d x.ne0=%lld x.ne1=%lld x.nb0=%lld x.nb1=%lld es=%lld "
                            "D_in_ne0=%d nb0_eq_es=%d D_fastest=%d\n",
                    node_n, (long long) x->ne[0], (long long) x->ne[1],
                    (long long) x->nb[0], (long long) x->nb[1], (long long) es,
                    x->ne[0] == 32, x->nb[0] == es, x->nb[0] == es && x->ne[0] == 32);
        }
    }

    // Check every buffer the kernel actually READS. Checking only the chain sources is not enough -- the
    // first version of this guard did that, found nothing, and let the fusion through.
    //
    // KATAGO_CPU_ROPE_NOA=1 disables ONLY the `ad vs a` term. That term is the one that fires on this graph
    // (a=1, everything else 0) and it is the one whose harm is arguable: the kernel recomputes t1 itself and
    // never loads dst(MUL#0), so aliasing it looks benign. This switch is the discriminator between
    // "the guard is protecting something real" and "the guard is rejecting a benign alias and the failure
    // has another cause" -- run it, read the ledger and the output hash, and do not argue.
    static int no_a = -1;
    if (no_a < 0) {
        no_a = (getenv("KATAGO_CPU_ROPE_NOA") != NULL) ? 1 : 0;
    }
    const int a_alias = no_a ? 0 : ggml_cpu_tensors_overlap(ad, a);

    if (a_alias                                   ||
        ggml_cpu_tensors_overlap(ad, gr)            ||
        ggml_cpu_tensors_overlap(ad, a->src[0])     ||
        ggml_cpu_tensors_overlap(ad, a->src[1])     ||
        ggml_cpu_tensors_overlap(ad, b->src[1])     ||
        ggml_cpu_tensors_overlap(ad, gr->src[1])) {
        if (ggml_cpu_rope_trace_on() >= 1 && params->ith == 0) {
            fprintf(stderr, "[rope-refuse] node=%d dst aliases a buffer the fused kernel reads "
                            "(a=%d gr=%d x=%d tabA=%d tabB=%d idx=%d)\n",
                    node_n,
                    ggml_cpu_tensors_overlap(ad, a), ggml_cpu_tensors_overlap(ad, gr),
                    ggml_cpu_tensors_overlap(ad, a->src[0]), ggml_cpu_tensors_overlap(ad, a->src[1]),
                    ggml_cpu_tensors_overlap(ad, b->src[1]), ggml_cpu_tensors_overlap(ad, gr->src[1]));
        }
        return 0;
    }

    // KATAGO_CPU_ROPE_ADDR=1: dump every address the fused kernel touches, for ONE site, so the aliasing
    // question is answered from data instead of argued. The READ set is what the kernel actually loads
    // (x, tabA, tabB, idx); the ELIDED set is the three intermediate results it does NOT load, which is
    // how a benign alias is told apart from a harmful one.
    {
        const char * ad_env = getenv("KATAGO_CPU_ROPE_ADDR");
        static int n_addr = 0;
        if (ad_env != NULL && ad_env[0] == '1' && params->ith == 0 && n_addr < 1) {
            n_addr++;
            fprintf(stderr, "[rope-addr] node=%d  nbytes(ad)=%lld\n",
                    node_n, (long long) ggml_nbytes(ad));
            fprintf(stderr, "[rope-addr]   WRITE dst(ADD)  =%p\n", (const void *) ad->data);
            fprintf(stderr, "[rope-addr]   READ  x         =%p\n", (const void *) a->src[0]->data);
            fprintf(stderr, "[rope-addr]   READ  tabA      =%p\n", (const void *) a->src[1]->data);
            fprintf(stderr, "[rope-addr]   READ  tabB      =%p\n", (const void *) b->src[1]->data);
            fprintf(stderr, "[rope-addr]   READ  idx       =%p\n", (const void *) gr->src[1]->data);
            fprintf(stderr, "[rope-addr]   ELIDED dst(MUL0)=%p\n", (const void *) a->data);
            fprintf(stderr, "[rope-addr]   ELIDED dst(GR)  =%p\n", (const void *) gr->data);
            fprintf(stderr, "[rope-addr]   ELIDED dst(MUL1)=%p\n", (const void *) b->data);
            fprintf(stderr, "[rope-addr]   OVERLAP x=%d tabA=%d tabB=%d idx=%d | MUL0=%d GR=%d MUL1=%d\n",
                    ggml_cpu_tensors_overlap(ad, a->src[0]), ggml_cpu_tensors_overlap(ad, a->src[1]),
                    ggml_cpu_tensors_overlap(ad, b->src[1]), ggml_cpu_tensors_overlap(ad, gr->src[1]),
                    ggml_cpu_tensors_overlap(ad, a), ggml_cpu_tensors_overlap(ad, gr),
                    ggml_cpu_tensors_overlap(ad, b));
        }
    }

    const int only = ggml_cpu_rope_only_node();
    if (only >= 0 && node_n != only) {
        return 0;   // bisection mode: this matched site is left to the stock kernels
    }

    ggml_compute_forward_rope_fused(params, ad, a, b, gr);

    // Counted only once the fusion really happened (i.e. after the bisection gate), so the reported
    // ropeNodes is the number of sites actually fused.
    ggml_cpu_rope_note_hit(cache, "{MUL,GET_ROWS,MUL,ADD}", node_n);

    // KATAGO_CPU_ROPE_NOSKIP=1 runs the fused kernel but still lets the three nodes run afterwards. This is
    // the NEGATIVE CONTROL that separates "the kernel is wrong" from "skipping is wrong": with it the
    // network output is bit-identical to the baseline, which is how the kernel was shown to be free of side
    // effects. NOTE its epistemic limit -- the stock chain overwrites what the fused kernel wrote, so it
    // does NOT by itself prove the kernel's arithmetic; that comes from KATAGO_CPU_ROPE_AUDIT (see
    // ROPE-FUSION-OUTCOME.md section 2).
    {
        static int noskip = -1;
        if (noskip < 0) {
            const char * e = getenv("KATAGO_CPU_ROPE_NOSKIP");
            noskip = (e != NULL && e[0] == '1') ? 1 : 0;
        }
        if (noskip) {
            return 0;
        }
    }

    return 3;
}

static int ggml_cpu_try_fuse_rope(
        const struct ggml_cgraph * cgraph,
        const int node_n,
        const struct ggml_compute_params * params) {

    return ggml_cpu_rope_match(cgraph, node_n, params);
}

// ---- A1: {MUL_MAT, <free view>, ADD} ------------------------------------------------------------//
// Measured shapes on the production graph (56 sites), two forms:
//     MUL_MAT[384,3]    -> RESHAPE[384,1,3,1]   -> ADD[384,361,3,1]   (broadcast along S)
//     MUL_MAT[192,1083] -> RESHAPE[192,361,3,1] -> ADD[192,361,3,1]   (pure relabel)
// The fusion skips the free view AND the ADD, so it removes ONE non-empty node per site (the ADD); the
// view is already free. Barrier cost is proportional to node count (BARRIER-SPIN-QUANTIFIED.md), and the
// denominator to quote is computed_nodes_per_walk (640), not nonempty_nodes (800).
//
// NOTE ON SCOPE: this is CPU-local and does NOT touch the shared judgement layer (ggml_can_fuse_ext in
// ggml-impl.h). See-through is implemented here instead, so the upstream matcher is unchanged. Vulkan's
// MUL_MAT_ADD is the ADJACENT form only (GV:20175), so this see-through is new on both backends.
//
// The three counters are deliberately separate from the first line of code: hit (predicate matched),
// refuse (matched but rejected by a guard), executed (the kernel actually ran). This project has already
// lost time to "logged 40 hits when 1 site fused".
// The three A1 counters live near the other barrier counters at the top of this file.

static int ggml_cpu_t9_enabled(void) {
    static int on = -1;
    if (on < 0) {
        const char * e = getenv("KATAGO_CPU_T9_FUSE");
        on = (e != NULL && e[0] == '1') ? 1 : 0;
    }
    return on;
}

// Trace switch, read ONCE. This used to be a raw getenv() on the refusal path, which the matcher reaches for
// every node that is not a match -- i.e. several hundred times per walk, on every thread, and getenv on
// Windows walks the whole environment block with string compares. Measured cost of the detection alone was
// +9.01% against the baseline (T=8 B=6), MORE than the whole fusion costs (+7.31%), which is what pointed
// here. Caching it also lets the caller short-circuit before doing any work.
static int ggml_cpu_t9_trace(void) {
    static int v = -1;
    if (v < 0) {
        v = (getenv("KATAGO_CPU_T9_TRACE") != NULL) ? 1 : 0;
    }
    return v;
}

// Timing-only ablation switch, read once. See the call site: it keeps the fusion's structure and its write
// stream and removes the pass's read streams, so it splits a structural cost from a kernel cost.
static int ggml_cpu_t9_nopass(void) {
    static int v = -1;
    if (v < 0) {
        v = (getenv("KATAGO_CPU_T9_NOPASS") != NULL) ? 1 : 0;
    }
    return v;
}

// A free layout op: ggml_op_is_empty skips these, so they cost no barrier and can be stepped over.
static int ggml_cpu_is_free_op(enum ggml_op op) {
    return op == GGML_OP_RESHAPE || op == GGML_OP_VIEW ||
           op == GGML_OP_PERMUTE || op == GGML_OP_TRANSPOSE || op == GGML_OP_NONE;
}

// Raw single-use test for a REAL tensor. cgraph->use_counts is keyed by the tensor's hash slot; this is the
// same question ggml_cpu_rope_uses_ok answers from a node index, asked directly because a caller that starts
// from a tensor (the ADDONLY diagnostic) has no node index for it. Returns 0 when the count cannot be read,
// so an unreadable graph refuses rather than proceeding.
static int ggml_cpu_t9_uses_once(const struct ggml_cgraph * cgraph, const struct ggml_tensor * t) {
    if (cgraph->use_counts == NULL) {
        return 0;
    }
    const size_t hp = ggml_hash_find(&cgraph->visited_hash_set, t);
    if (!ggml_bitset_get(cgraph->visited_hash_set.used, hp)) {
        return 0;
    }
    return cgraph->use_counts[hp] == 1;
}

// Returns the number of nodes to skip (2) on success. `node_n` indexes the MUL_MAT.
static int ggml_cpu_t9_match(
        const struct ggml_cgraph * cgraph,
        const int node_n,
        const struct ggml_compute_params * params) {

    // Every rejection is counted and (under KATAGO_CPU_T9_TRACE=1) named. "hit=0" with no reason is exactly
    // the failure mode this project keeps losing time to, so the reason is recorded at each return.
    const char * why = NULL;
    #define T9_NO(r) do { why = (r); __atomic_fetch_add(&ggml_cpu_t9_refuse, 1, __ATOMIC_RELAXED); \
                          goto t9_done; } while (0)

    if (cgraph->n_nodes - node_n < 3) {
        T9_NO("not enough nodes");
    }
    if (cgraph->nodes[node_n]->op != GGML_OP_MUL_MAT) {
        T9_NO("anchor is not MUL_MAT");
    }

    const struct ggml_tensor * mid = cgraph->nodes[node_n + 1];
    const struct ggml_tensor * add = cgraph->nodes[node_n + 2];

    if (!ggml_cpu_is_free_op(mid->op)) {
        T9_NO("mid is not a free op");
    }
    if (add->op != GGML_OP_ADD) {
        T9_NO("tail is not ADD");
    }
    if (mid->src[0] != cgraph->nodes[node_n]) {
        T9_NO("free op does not consume the MUL_MAT");
    }
    const int mid_in_add = (add->src[0] == mid) ? 0 : ((add->src[1] == mid) ? 1 : -1);
    if (mid_in_add < 0) {
        T9_NO("ADD does not consume the free op");
    }
    if (!ggml_can_repeat(mid, add)) {
        T9_NO("view cannot be broadcast to the ADD's shape");
    }
    if (add->type != GGML_TYPE_F32 || mid->type != GGML_TYPE_F32 ||
        cgraph->nodes[node_n]->type != GGML_TYPE_F32) {
        T9_NO("not all F32 (gemm/view/add)");
    }
    if (!ggml_is_contiguous(add) || !ggml_is_contiguous(mid) || !ggml_is_contiguous(cgraph->nodes[node_n])) {
        T9_NO("not all contiguous");
    }
    const struct ggml_tensor * other = add->src[1 - mid_in_add];
    if (other == NULL || other->type != GGML_TYPE_F32) {
        T9_NO("other operand missing or not F32");
    }
    if (ggml_nelements(cgraph->nodes[node_n]) != ggml_nelements(mid)) {
        T9_NO("gemm output and view differ in element count");
    }
    if (ggml_nelements(mid) <= 0 || ggml_nelements(add) <= 0) {
        T9_NO("empty tensor");   // keeps the row arithmetic below free of division by zero
    }
    // The pass is keyed on the GEMM's own chunk ranges (columns of the MUL_MAT output), so the fast
    // dimension must be shared, unresized, by all three tensors. Contiguity plus the equal element count
    // above then makes the view's element order identical to the MUL_MAT's, which is what lets the pass
    // read the MUL_MAT buffer directly with the view's row mapping.
    if (mid->ne[0] != cgraph->nodes[node_n]->ne[0]) {
        T9_NO("view does not preserve the MUL_MAT's fast dimension");
    }
    if (mid->ne[0] != add->ne[0] || other->ne[0] != add->ne[0]) {
        T9_NO("fast dimension is broadcast");
    }
    if (!ggml_can_repeat(other, add)) {
        T9_NO("other operand cannot be broadcast to the ADD's shape");
    }
    // single-use checks last: they need the per-graph cache, which is expensive to build, so only a chain
    // that already looks like A1 pays for it.
    //
    // NOTE the middle node IS a view by construction, so ggml_cpu_rope_uses_once (which mirrors
    // ggml_node_has_n_uses and rejects any view) cannot be used for it. That rejection is over-strict here:
    // the fused kernel reads the buffer the view aliases, so an existing view_src does not matter -- what
    // matters is that nothing else consumes the view. Hence the raw use-count check below.
    {
        const ggml_cpu_rope_uses * cache = ggml_cpu_rope_use_counts(cgraph);
        if (!ggml_cpu_rope_uses_ok(cache, cgraph, node_n)) {
            T9_NO("MUL_MAT has more than one use");
        }
        if (cgraph->nodes[node_n + 1]->op & GGML_TENSOR_FLAG_OUTPUT) {
            T9_NO("view is a graph output");
        }
        {
            int32_t n_uses = 0;
            if (cgraph->use_counts != NULL) {
                const size_t hp = ggml_hash_find(&cgraph->visited_hash_set, mid);
                if (ggml_bitset_get(cgraph->visited_hash_set.used, hp)) {
                    n_uses = cgraph->use_counts[hp];
                }
            }
            if (n_uses != 1) {
                T9_NO("view is used more than once");
            }
        }
    }

    __atomic_fetch_add(&ggml_cpu_t9_hit, 1, __ATOMIC_RELAXED);

    // The matmul must still run: A1 skips the free view and the ADD, NOT the GEMM. Running it HERE, at the
    // anchor, is exactly what saves the ADD's barrier -- the previous version returned "skip 2" without ever
    // computing the matmul, so the pass read the anchor's output buffer before anything had written it
    // (A1-EPILOGUE-FUSION-STATUS.md 3.2). Publishing the match and calling into the GEMM on this same thread
    // lets the pass ride along in the GEMM's own barrier interval, by whichever of the two routes that GEMM
    // supports; see the block comment above ggml_cpu_t9_pending.
    ggml_cpu_t9_pend.mm     = cgraph->nodes[node_n];
    ggml_cpu_t9_pend.view   = mid;
    ggml_cpu_t9_pend.add    = add;
    ggml_cpu_t9_pend.other  = other;
    ggml_cpu_t9_pend.s_ne1  = mid->ne[1];
    ggml_cpu_t9_pend.s_ne2  = mid->ne[2];
    ggml_cpu_t9_pend.s_ne3  = mid->ne[3];
    ggml_cpu_t9_pend.nr_rows = mid->ne[1] * mid->ne[2] * mid->ne[3];
    ggml_cpu_t9_pend.armed  = 1;
    ggml_cpu_t9_pend.pass_ran = 0;

    // TWO-WAY SPLIT before any per-region timer (user's cheaper division). MATCHONLY runs the WHOLE
    // predicate -- including its use-count lookups -- and then does nothing else: no GEMM here, no skipped
    // nodes. The node loop computes everything exactly as the baseline does, so the difference from the
    // baseline is the COST OF LOOKING, and nothing else. That separates the two classes of cause, which
    // have completely different fixes:
    //   MATCHONLY also ~+7%  => the account is per-node detection / relocating the GEMM call site
    //                           (fixable: short-circuit the preconditions, hoist the decision into
    //                           ggml_graph_plan -- the TODO at the plan site)
    //   MATCHONLY back at 0  => the account is the anchor/skip structure at the 52 sites
    // Do NOT reuse T9_NOSKIP for this: NOSKIP runs the GEMM in the matcher AND lets the node loop run it
    // again, doubling the GEMM and drowning the effect being measured.
    {
        static int t9_matchonly = -1;
        if (t9_matchonly < 0) {
            t9_matchonly = (getenv("KATAGO_CPU_T9_MATCHONLY") != NULL) ? 1 : 0;
        }
        if (t9_matchonly) {
            ggml_cpu_t9_pend.armed = 0;
            #undef T9_NO
            return 0;
        }
    }

    ggml_compute_forward_mul_mat(params, cgraph->nodes[node_n]);

    const int pass_ran = ggml_cpu_t9_pend.pass_ran;
    const int chunk_path = ggml_cpu_t9_pend.diag[3] == 0;
    ggml_cpu_t9_pend.armed  = 0;

    // No path was able to carry the pass: the GEMM left through an early-out whose kernel published no
    // barrier after its writes (see the handshake in llamafile/sgemm.cpp), so the ADD has not been written.
    // Fall back to the stock chain -- the node loop recomputes this same matmul identically and then runs
    // the view and the ADD. Refusing here is the whole point of the handshake: it turns "this kernel's
    // internals changed" into a lost optimisation instead of a silent race.
    if (!pass_ran) {
        __atomic_fetch_add(&ggml_cpu_t9_path_fallback, 1, __ATOMIC_RELAXED);
        {
            static int n_rep = 0;
            if (ggml_cpu_t9_trace() && params->ith == 0 && n_rep < 8) {
                n_rep++;
                const int * d = ggml_cpu_t9_pend.diag;
                fprintf(stderr, "[t9-path] node=%d FALLBACK exit=%d src0=%s src1=%s vec_dot=%s\n",
                        node_n, d[3],
                        ggml_type_name((enum ggml_type) d[0]),
                        ggml_type_name((enum ggml_type) d[1]),
                        ggml_type_name((enum ggml_type) d[2]));
            }
        }
        #undef T9_NO
        return 0;
    }

    if (chunk_path) {
        __atomic_fetch_add(&ggml_cpu_t9_path_chunk, 1, __ATOMIC_RELAXED);
    } else {
        __atomic_fetch_add(&ggml_cpu_t9_path_early, 1, __ATOMIC_RELAXED);
    }
    __atomic_fetch_add(&ggml_cpu_t9_execute, 1, __ATOMIC_RELAXED);

    // KATAGO_CPU_T9_NOSKIP=1 lets the stock chain run afterwards as well, so the fused result and the stock
    // ADD can be diffed inside one run, on identical inputs. Note it recomputes the GEMM (the anchor now
    // always computes it); the second run is redundant but harmless, and this mode is diagnostic only.
    {
        static int t9_noskip = -1;
        if (t9_noskip < 0) {
            t9_noskip = (getenv("KATAGO_CPU_T9_NOSKIP") != NULL) ? 1 : 0;
        }
        if (t9_noskip) {
            #undef T9_NO
            return 0;
        }
    }

    #undef T9_NO
    return 2;

t9_done:
    #undef T9_NO
    if (ggml_cpu_t9_trace() && params->ith == 0) {
        static int n_rep = 0;
        // Report only the refusals that MATTER: a chain that already looks like A1 (MUL_MAT -> free -> ADD).
        // The matcher runs for every node, so without this filter the first 24 refusals are all
        // "anchor is not MUL_MAT" and "tail is not ADD" from unrelated nodes, and the reasons for the sites
        // we actually care about are never printed. That is exactly what happened when this trace was first
        // used to explain why 67 A1-shaped sites yield only 52 hits.
        const int looks_a1 = (node_n + 2 < cgraph->n_nodes) &&
                             (cgraph->nodes[node_n]     != NULL) &&
                             (cgraph->nodes[node_n + 1] != NULL) &&
                             (cgraph->nodes[node_n + 2] != NULL) &&
                             (cgraph->nodes[node_n]->op == GGML_OP_MUL_MAT) &&
                             ggml_cpu_is_free_op(cgraph->nodes[node_n + 1]->op) &&
                             (cgraph->nodes[node_n + 2]->op == GGML_OP_ADD);
        if (looks_a1 && n_rep < 40) {
            n_rep++;
            fprintf(stderr, "[t9-a1blk] node=%d rejected: %s\n", node_n, why ? why : "?");
        }
    }
    return 0;
}

// A1 pass: for each MUL_MAT row r in [r_begin, r_end), write every ADD element that reads that row, over the
// fast-dimension range [c_begin, c_end). `mm` is the anchor MUL_MAT and is read directly instead of through
// the view: the matcher guarantees the view preserves the fast dimension and has the same element count, so
// the two share one flat layout, and `mm` is the buffer whose writes the caller has made visible.
//
// The caller decides the partition. In the chunk loop it passes the GEMM's own ir0/ir1 ranges, so a thread
// only ever reads rows it just wrote. After a llamafile sgemm it passes a plain row split, which is safe
// because that sgemm ended with a barrier ordered after all of its writes.
//
// Both the decomposition and the row mapping come from the stock paths on purpose:
//   * the (i1,i2,i3) decomposition of the ADD's rows and the `%` reductions are apply_binary_op's broadcast
//     rule, which is what makes the result equal the stock ADD;
//   * the view's row index is the same `%` reduction over the view's own ne.
// No gather is involved here (unlike the RoPE chain), which is why this pass does not carry that failure
// mode.
static void ggml_cpu_t9_anchor_epilogue(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * mm,
        int64_t r_begin, int64_t r_end,
        int64_t c_begin, int64_t c_end) {

    GGML_UNUSED(params);

    const struct ggml_tensor * add   = ggml_cpu_t9_pend.add;
    const struct ggml_tensor * other = ggml_cpu_t9_pend.other;

    const int64_t ne0 = add->ne[0];              // == mm->ne[0], enforced by the matcher
    const int64_t a_ne1 = add->ne[1], a_ne2 = add->ne[2], a_ne3 = add->ne[3];
    const int64_t s_ne0 = mm->ne[0];
    const int64_t s_ne1 = ggml_cpu_t9_pend.s_ne1;
    const int64_t s_ne2 = ggml_cpu_t9_pend.s_ne2;
    const int64_t s_ne3 = ggml_cpu_t9_pend.s_ne3;
    const int64_t o_ne0 = other->ne[0], o_ne1 = other->ne[1], o_ne2 = other->ne[2], o_ne3 = other->ne[3];

    if (r_begin >= r_end || c_begin >= c_end) {
        return;
    }

    // Rows are walked in the GEMM's own row space and each such row is fanned out to the ADD rows that
    // alias it. Walking from the GEMM side rather than scanning every ADD row keeps the work proportional
    // to what is actually written: a scan would cost O(nr_add) per call even when a range maps to a
    // handful of rows. The fan-out loop is the general `%` reduction read backwards: i_k runs over the ADD
    // indices congruent to the view's index v_k modulo the view's ne_k, with a step of 1 whenever that
    // dimension is broadcast (s_ne_k == 1, and then v_k == 0).
    const int64_t s_row   = s_ne1 * s_ne2;         // rows per plane of the view
    for (int64_t r = r_begin; r < r_end; ++r) {
        const int64_t v3  = r / s_row;
        const int64_t rem = r - v3 * s_row;
        const int64_t v2  = rem / s_ne1;
        const int64_t v1  = rem - v2 * s_ne1;

        const float * s = (const float *) ((const char *) mm->data + r * s_ne0 * sizeof(float));

        for (int64_t i3 = v3; i3 < a_ne3; i3 += s_ne3) {
            const int64_t o3 = i3 % o_ne3;
            for (int64_t i2 = v2; i2 < a_ne2; i2 += s_ne2) {
                const int64_t o2 = i2 % o_ne2;
                for (int64_t i1 = v1; i1 < a_ne1; i1 += s_ne1) {
                    const int64_t o1    = i1 % o_ne1;
                    const int64_t o_row = o1 + o_ne1 * o2 + o_ne1 * o_ne2 * o3;
                    const float * o     = (const float *) ((const char *) other->data + o_row * o_ne0 * sizeof(float));
                    const int64_t ir    = i1 + a_ne1 * (i2 + a_ne2 * i3);
                    float * d           = (float *) ((char *) add->data + ir * ne0 * sizeof(float));

                    // Timing-only ablation (KATAGO_CPU_T9_NOPASS=1). Keeps the shell, the row fan-out, the
                    // partition and the WRITE stream, and drops the two READ streams plus the add. It writes
                    // zeros rather than leaving the buffer untouched, so nothing downstream can see a denormal
                    // or a NaN and charge its own cost to this experiment. The output is WRONG by
                    // construction: this exists to split "the fusion structure costs something" from "the
                    // pass's memory traffic costs something", which is the difference between a structural
                    // problem and a kernel problem.
                    if (ggml_cpu_t9_nopass()) {
                        memset(d + c_begin, 0, (size_t) (c_end - c_begin) * sizeof(float));
                    } else {
                        // Use the SAME primitive the stock ADD path uses (vec.h:89, explicit AVX2 path).
                        ggml_vec_add_f32((int) (c_end - c_begin), d + c_begin, s + c_begin, o + c_begin);
                    }
                }
            }
        }
    }

    __atomic_fetch_add(&ggml_cpu_t9_epilogue_chunks, 1, __ATOMIC_RELAXED);
}


// Try to fuse the current node with subsequent nodes for better performance.
// Returns the number of nodes skipped by fusion (>=1), or 0 if no fusion was applied.
static bool ggml_cpu_disable_fusion = false;  // initialized once in ggml_cpu_init(), read-only afterwards

static int ggml_cpu_try_fuse_ops(
        const struct ggml_cgraph * cgraph,
        const int node_n,
        const struct ggml_compute_params * params,
        const struct ggml_cplan * cplan) {

    // Full-census trigger, moved here because this call site is known to run (the earlier one in the thread
    // entry did not fire). KATAGO_CPU_ROPE_DUMP=full.
    {
        static int census_done = 0;
        const char * full = getenv("KATAGO_CPU_ROPE_DUMP");
        if (!census_done && full != NULL && full[0] == 'f' && params->ith == 0) {
            census_done = 1;
            for (int i = 0; i < cgraph->n_nodes; i++) {
                const struct ggml_tensor * t = cgraph->nodes[i];
                fprintf(stderr, ": CPU graph node %d %s type=%s ne=[%lld,%lld,%lld,%lld] nb=[%lld,%lld,%lld,%lld] contig=%d src0=%s src1=%s\n",
                        i, ggml_op_name(t->op), ggml_type_name(t->type),
                        (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3],
                        (long long) t->nb[0], (long long) t->nb[1], (long long) t->nb[2], (long long) t->nb[3],
                        (int) ggml_is_contiguous(t),
                        t->src[0] ? ggml_op_name(t->src[0]->op) : "NULL",
                        t->src[1] ? ggml_op_name(t->src[1]->op) : "NULL");
            }
            fprintf(stderr, "[rope-dump] CENSUS n_nodes=%d\n", cgraph->n_nodes);
        }
    }

    if (ggml_cpu_disable_fusion || cplan->use_ref) {
        return 0;
    }

    struct ggml_tensor * node = cgraph->nodes[node_n];

    if (node->op == GGML_OP_RMS_NORM) {
        // RMS_NORM + MUL fusion
        const enum ggml_op fuse_ops[] = { GGML_OP_RMS_NORM, GGML_OP_MUL };
        if (ggml_can_fuse(cgraph, node_n, fuse_ops, 2)) {
            struct ggml_tensor * mul_node = cgraph->nodes[node_n + 1];
            const struct ggml_tensor * mul_w = (mul_node->src[0] == node)
                ? mul_node->src[1] : mul_node->src[0];
            if (node->src[0]->type  == GGML_TYPE_F32 &&
                mul_node->type      == GGML_TYPE_F32 &&
                mul_w->type         == GGML_TYPE_F32 &&
                mul_w->ne[0]        == node->ne[0]   &&
                mul_w->nb[0]        == sizeof(float)) {

                ggml_compute_forward_rms_norm_mul_fused(params, node, mul_node);
                return 1;
            }
        }
    }

    // A1 BISECT (diagnostic, KATAGO_CPU_T9_ADDONLY=1): run the fused pass IN PLACE OF the stock ADD kernel,
    // with the GEMM and the view left completely alone. This isolates ONE variable -- my pass versus
    // ggml_compute_forward_add_non_quantized -- at the same position in the same barrier interval, so it
    // separates "the pass is slower than the kernel it replaces" from "moving the work into the GEMM's
    // interval costs something". The A1 fusion measured ~-7% at T=8 B=6 three times; this says which part.
    // Correctness is preserved (the pass computes the same values, which the bit gate independently proves),
    // so any timing difference here is attributable to the kernel alone.
    {
        static int t9_addonly = -1;
        if (t9_addonly < 0) {
            t9_addonly = (getenv("KATAGO_CPU_T9_ADDONLY") != NULL) ? 1 : 0;
        }
        if (t9_addonly && node->op == GGML_OP_ADD) {
            const struct ggml_tensor * v = NULL;
            const struct ggml_tensor * o = NULL;
            for (int k = 0; k < 2; ++k) {
                const struct ggml_tensor * s = node->src[k];
                if (s != NULL && ggml_cpu_is_free_op(s->op) && s->src[0] != NULL &&
                    s->src[0]->op == GGML_OP_MUL_MAT) {
                    v = s;
                    o = node->src[1 - k];
                }
            }
            if (v != NULL && o != NULL &&
                v->type == GGML_TYPE_F32 && o->type == GGML_TYPE_F32 && node->type == GGML_TYPE_F32 &&
                v->src[0]->type == GGML_TYPE_F32 &&
                ggml_is_contiguous(node) && ggml_is_contiguous(v) && ggml_is_contiguous(o) &&
                ggml_is_contiguous(v->src[0]) &&
                ggml_nelements(v) == ggml_nelements(v->src[0]) &&
                ggml_nelements(v) > 0 && ggml_nelements(node) > 0 &&
                v->ne[0] == v->src[0]->ne[0] && v->ne[0] == node->ne[0] && o->ne[0] == node->ne[0] &&
                ggml_can_repeat(v, node) && ggml_can_repeat(o, node) &&
                !(node->flags & GGML_TENSOR_FLAG_OUTPUT) &&
                // The same single-use guards the matcher applies. Without them the ADDONLY diagnostic matched
                // about 15 more sites than the matcher does, wrote them with a predicate it had not checked,
                // and produced a different network hash -- which is why its first timing number could not be
                // believed. The matcher reaches these through a node index; here the producer has no index,
                // so the same question is asked directly against the graph's use-count table.
                ggml_cpu_t9_uses_once(cgraph, v->src[0]) &&
                ggml_cpu_t9_uses_once(cgraph, v)) {
                if (ggml_cpu_t9_trace() && params->ith == 0) {
                    static int n_rep = 0;
                    if (n_rep < 40) {
                        n_rep++;
                        fprintf(stderr, "[t9-addonly] node=%d mm#? ne=[%lld,%lld,%lld,%lld] "
                                        "v.ne=[%lld,%lld,%lld,%lld] o.ne=[%lld,%lld,%lld,%lld] "
                                        "v.nb=[%lld,%lld,%lld,%lld] o.nb=[%lld,%lld,%lld,%lld]\n",
                                node_n,
                                (long long) v->src[0]->ne[0], (long long) v->src[0]->ne[1],
                                (long long) v->src[0]->ne[2], (long long) v->src[0]->ne[3],
                                (long long) v->ne[0], (long long) v->ne[1], (long long) v->ne[2], (long long) v->ne[3],
                                (long long) o->ne[0], (long long) o->ne[1], (long long) o->ne[2], (long long) o->ne[3],
                                (long long) v->nb[0], (long long) v->nb[1], (long long) v->nb[2], (long long) v->nb[3],
                                (long long) o->nb[0], (long long) o->nb[1], (long long) o->nb[2], (long long) o->nb[3]);
                    }
                }
                ggml_cpu_t9_pend.mm       = v->src[0];
                ggml_cpu_t9_pend.view     = v;
                ggml_cpu_t9_pend.add      = node;
                ggml_cpu_t9_pend.other    = o;
                ggml_cpu_t9_pend.s_ne1    = v->ne[1];
                ggml_cpu_t9_pend.s_ne2    = v->ne[2];
                ggml_cpu_t9_pend.s_ne3    = v->ne[3];
                ggml_cpu_t9_pend.nr_rows  = v->ne[1] * v->ne[2] * v->ne[3];
                ggml_cpu_t9_pend.armed    = 0;      // armed stays 0: this must NOT make the GEMM fuse
                ggml_cpu_t9_pend.pass_ran = 0;
                ggml_cpu_t9_pass_after_sgemm(params, v->src[0]);
                return 1;                           // the ADD is done; skip just this node
            }
        }
    }

    if (ggml_cpu_t9_enabled()) {
        // A1: {MUL_MAT, <free view>, ADD} -- see the block comment above ggml_cpu_t9_match. Called for every
        // node (not just MUL_MAT) so the refusal counters and the trace name the real reason.
        const int t9 = ggml_cpu_t9_match(cgraph, node_n, params);
        if (t9 > 0) {
            return t9;
        }
    }

    if (node->op == GGML_OP_MUL) {
        // RoPE rotation pair, matched as a diamond: {MUL, GET_ROWS, MUL, ADD}, or the gather-less
        // {MUL, MUL, ADD} that a paired-read shape will leave behind.
        //
        // DEFAULT OFF: see the status block above the predicate. The kernel and the predicate are verified
        // correct, but skipping the three nodes changes the network's numbers, so this must not be on by
        // accident.
        static int rope_fuse_enabled = -1;
        if (rope_fuse_enabled < 0) {
            const char * e = getenv("KATAGO_CPU_ROPE_FUSE");
            rope_fuse_enabled = (e != NULL && e[0] == '1') ? 1 : 0;
        }
        if (rope_fuse_enabled) {
            const int skipped = ggml_cpu_try_fuse_rope(cgraph, node_n, params);
            if (skipped > 0) {
                return skipped;
            }
        }
    }

    return 0;
}

static thread_ret_t ggml_graph_compute_thread(void * data) {
    struct ggml_compute_state * state = (struct ggml_compute_state *) data;
    struct ggml_threadpool    * tp    = state->threadpool;

    const struct ggml_cgraph * cgraph = tp->cgraph;
    const struct ggml_cplan  * cplan  = tp->cplan;

#ifdef GGML_USE_CPU_RISCV64_SPACEMIT
    ggml_backend_cpu_riscv64_spacemit_set_numa_thread_affinity(state->ith);
#else
    set_numa_thread_affinity(state->ith);
#endif

    struct ggml_compute_params params = {
        /*.ith        =*/ state->ith,
        /*.nth        =*/ atomic_load_explicit(&tp->n_graph, memory_order_relaxed) & GGML_THREADPOOL_N_THREADS_MASK,
        /*.wsize      =*/ cplan->work_size,
        /*.wdata      =*/ cplan->work_data,
        /*.threadpool =*/ tp,
        /*.use_ref    =*/ cplan->use_ref,
    };

#ifdef GGML_USE_OPENMP
    GGML_PRINT_DEBUG("thread #%d compute-start cplan %p\n", state->ith, (const void *)cplan);
#else
    GGML_PRINT_DEBUG("thread #%d compute-start cplan %p last-graph %d\n", state->ith, (const void *)cplan, state->last_graph);
#endif

    // Resolve the diagnostic switch once, before the node loop, so the per-node counters below read a settled
    // value rather than lazily initialising it mid-loop.
    (void) ggml_cpu_barrier_stats_enabled();
    ggml_cpu_barrier_backoff_init();
    if (ggml_cpu_barrier_stats_on) {
        // Every thread writes the same value, once per graph. Only used to turn waits into barrier sites.
        ggml_cpu_bar_nth = params.nth;
        // ggml_barrier() gets no ith, so the thread index for the site table is stashed here, once per graph.
        ggml_cpu_tls_bar_ith = params.ith;
    }

    // One rdtsc pair per graph per thread: the denominator for the spin share. Deliberately always taken (not
    // only when the diagnostic is on) because it is once per graph, and the branch it would save is worth less
    // than the clarity of an unconditional span.
    const uint64_t span_t0 = ggml_cpu_bar_tsc();

    ggml_cpu_rope_dump_census(cgraph, params.ith);

    for (int node_n = 0; node_n < cgraph->n_nodes && atomic_load_explicit(&tp->abort, memory_order_relaxed) != node_n; node_n++) {
        struct ggml_tensor * node = cgraph->nodes[node_n];

        // Counted only when the diagnostic is on; the counters are shared, but these sites run once per node
        // rather than once per spin iteration, so the contention is negligible (unlike the spin counters,
        // which had to be made thread-local).
        if (ggml_cpu_barrier_stats_on) {
            __atomic_fetch_add(&ggml_cpu_n_nodes_seen, 1, __ATOMIC_RELAXED);
        }
        if (ggml_op_is_empty(node->op)) {
            // skip NOPs
            if (ggml_cpu_barrier_stats_on) {
                __atomic_fetch_add(&ggml_cpu_n_nodes_empty, 1, __ATOMIC_RELAXED);
            }
            continue;
        }

        if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            if (ggml_cpu_barrier_stats_on) {
                __atomic_fetch_add(&ggml_cpu_n_nodes_nocomp, 1, __ATOMIC_RELAXED);
            }
            continue;
        }
        if (ggml_cpu_barrier_stats_on) {
            __atomic_fetch_add(&ggml_cpu_n_nodes_compute, 1, __ATOMIC_RELAXED);
        }

        // TODO: move fused-op detection into ggml_graph_plan so fusion decisions are made once at planning time
        // Try fused ops, fall back to normal compute
        const int n_fused = ggml_cpu_try_fuse_ops(cgraph, node_n, &params, cplan);
        if (n_fused > 0) {
            node_n += n_fused;
        } else {
            ggml_compute_forward(&params, node);
        }

        if (state->ith == 0 && cplan->abort_callback &&
                cplan->abort_callback(cplan->abort_callback_data)) {
            atomic_store_explicit(&tp->abort, node_n + 1, memory_order_relaxed);
            tp->ec    = GGML_STATUS_ABORTED;
        }

        if (node_n + 1 < cgraph->n_nodes) {
            if (ggml_cpu_barrier_stats_on) {
                __atomic_fetch_add(&ggml_cpu_n_nodes_barr, 1, __ATOMIC_RELAXED);
                ggml_cpu_bar_nodes_by_op[params.ith][(unsigned) node->op < (unsigned) GGML_CPU_BAR_OP_SLOTS
                                                  ? (int) node->op : GGML_OP_COUNT]++;
            }
            GGML_CPU_BAR_SITE(GGML_CPU_BAR_SITE_LOOP);
            ggml_barrier(state->threadpool);
            // Attribute this barrier's spin to `node`, captured BEFORE the fusion advance above. For a plain
            // node that is simply "the node whose kernel just ran". For a fused group it is the group's ANCHOR,
            // which is deliberate: the fused pass lives inside the anchor's kernel, and it is what makes the ADD
            // row collapse under KATAGO_CPU_T9_FUSE=1 -- that collapse is the built-in positive control.
            if (ggml_cpu_barrier_stats_on) {
                ggml_cpu_bar_spin_drain(params.ith, (int) node->op);
            }
        }
    }

    ggml_cpu_rope_report_graph(ggml_cpu_rope_use_counts(cgraph), cgraph->n_nodes, params.ith);

#ifdef GGML_USE_OPENMP
    GGML_PRINT_DEBUG("thread #%d compute-done cplan %p\n", state->ith, (const void *)cplan);
#else
    GGML_PRINT_DEBUG("thread #%d compute-done cplan %p last-graph %d\n", state->ith, (const void *)cplan, state->last_graph);
#endif

    GGML_CPU_BAR_SITE(GGML_CPU_BAR_SITE_GRAPH_END);
    ggml_barrier(state->threadpool);

    // The graph-end barrier waits on no node; it gets its own column so the attribution invariant stays exact.
    if (ggml_cpu_barrier_stats_on) {
        ggml_cpu_bar_spin_drain(params.ith, GGML_OP_COUNT);
        __atomic_fetch_add(&ggml_cpu_bar_span_cycles, ggml_cpu_bar_tsc() - span_t0, __ATOMIC_RELAXED);
    }

#ifdef GGML_USE_CPU_RISCV64_SPACEMIT
    ggml_backend_cpu_riscv64_spacemit_clear_numa_thread_affinity_threaded(state->ith);
#endif

    return 0;
}

#ifndef GGML_USE_OPENMP

// check if thread is ready to proceed (exit from polling or sleeping)
// returns true if loops should exit, sets state->pending to indicate new work
static inline bool ggml_graph_compute_thread_ready(struct ggml_compute_state * state) {
    struct ggml_threadpool * threadpool = state->threadpool;

    if (state->pending || threadpool->stop || threadpool->pause) { return true; }

    // check for new graph/work
    int n_graph   = atomic_load_explicit(&threadpool->n_graph, memory_order_relaxed);
    int n_threads = n_graph & GGML_THREADPOOL_N_THREADS_MASK;
    if (n_graph != state->last_graph) {
        state->pending    = (state->ith < n_threads);
        state->last_graph = n_graph;
        return true;
    }

    return false;
}

// sync thread state after polling
static inline void ggml_graph_compute_thread_sync(struct ggml_compute_state * state) {
    // TSAN doesn't support standalone fence yet, we use a dummy read-modify-write instead
    #ifdef GGML_TSAN_ENABLED
    atomic_fetch_add_explicit(&state->threadpool->n_graph, 0, memory_order_seq_cst);
    #else
    atomic_thread_fence(memory_order_seq_cst);
    #endif
    UNUSED(state);
}

static inline bool ggml_graph_compute_poll_for_work(struct ggml_compute_state * state) {
    struct ggml_threadpool * threadpool = state->threadpool;

    // This seems to make 0 ... 100 a decent range for polling level across modern processors.
    // Perhaps, we can adjust it dynamically based on load and things.
    const uint64_t n_rounds = 1024UL * 128 * threadpool->poll;

    for (uint64_t i=0; !ggml_graph_compute_thread_ready(state) && i < n_rounds; i++) {
        // No new work. Keep polling.
        ggml_thread_cpu_relax();
    }

    return state->pending;
}

static inline bool ggml_graph_compute_check_for_work(struct ggml_compute_state * state) {
    struct ggml_threadpool * threadpool = state->threadpool;

    if (ggml_graph_compute_poll_for_work(state)) {
        ggml_graph_compute_thread_sync(state);
        return state->pending;
    }

    ggml_mutex_lock_shared(&threadpool->mutex);
    while (!ggml_graph_compute_thread_ready(state)) {
        // No new work. Wait for the signal.
        GGML_PRINT_DEBUG("thread #%d waiting for work (sleeping)\n", state->ith);
        ggml_cond_wait(&threadpool->cond, &threadpool->mutex);
    }
    ggml_mutex_unlock_shared(&threadpool->mutex);

    return state->pending;
}

static thread_ret_t ggml_graph_compute_secondary_thread(void* data) {
    struct ggml_compute_state * state = (struct ggml_compute_state *) data;
    struct ggml_threadpool * threadpool = state->threadpool;

    ggml_thread_apply_priority(threadpool->prio);
    if (ggml_thread_cpumask_is_valid(state->cpumask)) {
        ggml_thread_apply_affinity(state->cpumask);
    }

    while (true) {
        // Check if we need to sleep
        while (threadpool->pause) {
            GGML_PRINT_DEBUG("thread #%d inside pause loop\n", state->ith);
            ggml_mutex_lock_shared(&threadpool->mutex);
            if (threadpool->pause) {
                ggml_cond_wait(&threadpool->cond, &threadpool->mutex);
            }
            GGML_PRINT_DEBUG("thread #%d resuming after wait\n", state->ith);
            ggml_mutex_unlock_shared(&threadpool->mutex);
        }

        // This needs to be checked for after the cond_wait
        if (threadpool->stop) break;

        // Check if there is new work
        // The main thread is the only one that can dispatch new work

        ggml_graph_compute_check_for_work(state);
        if (state->pending) {
            state->pending = false;
            ggml_graph_compute_thread(state);
        }
    }

    return (thread_ret_t) 0;
}

// Start processing new graph
static void ggml_graph_compute_kickoff(struct ggml_threadpool * threadpool, int n_threads)
{
    // Always take the mutex here because the worker threads are doing hybrid poll/wait

    ggml_mutex_lock(&threadpool->mutex);

    // Update the number of active threads and the graph count
    int n_graph = atomic_load_explicit(&threadpool->n_graph, memory_order_relaxed) >> GGML_THREADPOOL_N_THREADS_BITS;
    n_graph = ((n_graph + 1) << GGML_THREADPOOL_N_THREADS_BITS) | (n_threads & GGML_THREADPOOL_N_THREADS_MASK);

    GGML_PRINT_DEBUG("compute-kickoff: n_threads %d n_graph %d\n", n_threads, n_graph);

    // Indicate the graph is ready to be processed
    // We need the full seq-cst fence here because of the polling threads (used in thread_sync)
    atomic_store_explicit(&threadpool->n_graph, n_graph, memory_order_seq_cst);

    if (threadpool->pause) {
       // Update main thread prio and affinity to match the threadpool settings
       ggml_thread_apply_priority(threadpool->prio);
       if (ggml_thread_cpumask_is_valid(threadpool->workers[0].cpumask)) {
           ggml_thread_apply_affinity(threadpool->workers[0].cpumask);
       }

       // resume does cond broadcast
       ggml_threadpool_resume_locked(threadpool);
    } else {
       ggml_cond_broadcast(&threadpool->cond);
    }

    ggml_mutex_unlock(&threadpool->mutex);
}

#endif // GGML_USE_OPENMP

static struct ggml_threadpool * ggml_threadpool_new_impl(
    struct ggml_threadpool_params * tpp,
               struct ggml_cgraph * cgraph,
                struct ggml_cplan * cplan) {

    struct ggml_threadpool * threadpool =
        ggml_aligned_malloc(sizeof(struct ggml_threadpool));
    {
        threadpool->cgraph           = cgraph;
        threadpool->cplan            = cplan;
        threadpool->n_graph          = 0;
        threadpool->n_barrier        = 0;
        threadpool->n_barrier_passed = 0;
        threadpool->current_chunk    = 0;
        threadpool->stop             = false;
        threadpool->pause            = tpp->paused;
        threadpool->abort            = -1;
        threadpool->workers          = NULL;
        threadpool->n_threads        = tpp->n_threads;
        threadpool->poll             = tpp->poll;
        threadpool->prio             = tpp->prio;
        threadpool->ec               = GGML_STATUS_SUCCESS;
    }

    // Allocate and init workers state
    const size_t workers_size = sizeof(struct ggml_compute_state) * tpp->n_threads;
    struct ggml_compute_state * workers = ggml_aligned_malloc(workers_size);

    memset(workers, 0, workers_size);
    for (int j = 0; j < tpp->n_threads; j++) {
        workers[j].threadpool = threadpool;
        workers[j].ith        = j;
    }

    threadpool->workers = workers;

#ifdef GGML_USE_OPENMP
    int32_t cpumask_iter = 0;

    // Compute CPU masks for each thread
    for (int j = 0; j < tpp->n_threads; j++) {
        ggml_thread_cpumask_next(tpp->cpumask, workers[j].cpumask, tpp->strict_cpu, &cpumask_iter);
    }
#else // GGML_USE_OPENMP
    ggml_mutex_init(&threadpool->mutex);
    ggml_cond_init(&threadpool->cond);

    // Spin the threads for all workers, and update CPU placements.
    // Place the main thread last (towards the higher numbered CPU cores).

    int32_t cpumask_iter = 0;

    for (int j = 1; j < tpp->n_threads; j++) {
        ggml_thread_cpumask_next(tpp->cpumask, workers[j].cpumask, tpp->strict_cpu, &cpumask_iter);

        int32_t rc = ggml_thread_create(&workers[j].thrd, NULL, ggml_graph_compute_secondary_thread, &workers[j]);
        GGML_ASSERT(rc == 0);
    }

    ggml_thread_cpumask_next(tpp->cpumask, workers[0].cpumask, tpp->strict_cpu, &cpumask_iter);

    if (!threadpool->pause) {
        // Update main thread prio and affinity at the start, otherwise we'll do it in resume
        ggml_thread_apply_priority(threadpool->prio);
        if (ggml_thread_cpumask_is_valid(threadpool->workers[0].cpumask)) {
            ggml_thread_apply_affinity(threadpool->workers[0].cpumask);
        }
    }
#endif // GGML_USE_OPENMP

    return threadpool;
}

struct ggml_threadpool * ggml_threadpool_new(struct ggml_threadpool_params * tpp) {
    return ggml_threadpool_new_impl(tpp, NULL, NULL);
}

enum ggml_status ggml_graph_compute(struct ggml_cgraph * cgraph, struct ggml_cplan * cplan) {
    ggml_cpu_init();

    // One-shot activation-consumer census, before any timing and outside every measured arm (G-22: a
    // flag-gated, read-only report that answers how the f16-activation work must be SPLIT).
    if (ggml_cpu_census_enabled() && !ggml_cpu_census_done) {
        ggml_cpu_census_done = 1;
        ggml_cpu_act_consumer_census(cgraph);
    }

    GGML_ASSERT(cplan);
    GGML_ASSERT(cplan->n_threads > 0);
    GGML_ASSERT(cplan->work_size == 0 || cplan->work_data != NULL);

    int n_threads                               = cplan->n_threads;
    struct ggml_threadpool * threadpool = cplan->threadpool;

    bool disposable_threadpool = false;

    if (threadpool == NULL) {
        //GGML_PRINT_DEBUG("Threadpool is not specified. Will create a disposable threadpool : n_threads %d\n", n_threads);
        disposable_threadpool = true;

        struct ggml_threadpool_params ttp = ggml_threadpool_params_default(n_threads);
        threadpool = ggml_threadpool_new_impl(&ttp, cgraph, cplan);
    } else {
        // Reset some of the parameters that need resetting
        // No worker threads should be accessing the parameters below at this stage
        threadpool->cgraph           = cgraph;
        threadpool->cplan            = cplan;
        threadpool->current_chunk    = 0;
        threadpool->abort            = -1;
        threadpool->ec               = GGML_STATUS_SUCCESS;
    }

#ifdef GGML_USE_OPENMP
    if (n_threads > 1) {
        #pragma omp parallel num_threads(n_threads)
        {
            #pragma omp single
            {
                // update the number of threads from the actual number of threads that we got from OpenMP
                n_threads = omp_get_num_threads();
                atomic_store_explicit(&threadpool->n_graph, n_threads, memory_order_relaxed);
            }

            // Apply thread CPU mask and priority
            int ith = omp_get_thread_num();

            ggml_thread_apply_priority(threadpool->prio);
            if (ggml_thread_cpumask_is_valid(threadpool->workers[ith].cpumask)) {
                ggml_thread_apply_affinity(threadpool->workers[ith].cpumask);
            }
            ggml_graph_compute_thread(&threadpool->workers[ith]);
        }
    } else {
        atomic_store_explicit(&threadpool->n_graph, 1, memory_order_relaxed);
        ggml_graph_compute_thread(&threadpool->workers[0]);
    }
#else
    if (n_threads > threadpool->n_threads) {
        GGML_LOG_WARN("cplan requested more threads (%d) than available (%d)\n", n_threads, threadpool->n_threads);
        n_threads = threadpool->n_threads;
    }

    // Kick all threads to start the new graph
    ggml_graph_compute_kickoff(threadpool, n_threads);

    // This is a work thread too
    ggml_graph_compute_thread(&threadpool->workers[0]);
#endif

    // don't leave affinity set on the main thread
    clear_numa_thread_affinity();

    enum ggml_status ret = threadpool->ec;

    if (disposable_threadpool) {
        ggml_threadpool_free(threadpool);
    }

    return ret;
}

enum ggml_status ggml_graph_compute_with_ctx(struct ggml_context * ctx, struct ggml_cgraph * cgraph, int n_threads) {
    struct ggml_cplan cplan = ggml_graph_plan(cgraph, n_threads, NULL);

    cplan.work_data = (uint8_t *)ggml_new_buffer(ctx, cplan.work_size);

    return ggml_graph_compute(cgraph, &cplan);
}

void ggml_cpu_fp32_to_fp32(const float * x, float * y, int64_t n) {
    memcpy(y, x, n * sizeof(float));
}

void ggml_cpu_fp32_to_fp16(const float * x, ggml_fp16_t * y, int64_t n) {
    int64_t i = 0;
#if defined(__F16C__)
#if defined(__AVX512F__)
    for (; i + 15 < n; i += 16) {
        __m512 x_vec = _mm512_loadu_ps(x + i);
        __m256i y_vec = _mm512_cvtps_ph(x_vec, _MM_FROUND_TO_NEAREST_INT);
        _mm256_storeu_si256((__m256i *)(y + i), y_vec);
    }
#endif
    for (; i + 7 < n; i += 8) {
        __m256 x_vec = _mm256_loadu_ps(x + i);
        __m128i y_vec = _mm256_cvtps_ph(x_vec, _MM_FROUND_TO_NEAREST_INT);
        _mm_storeu_si128((__m128i *)(y + i), y_vec);
    }
    for (; i + 3 < n; i += 4) {
        __m128 x_vec = _mm_loadu_ps(x + i);
        __m128i y_vec = _mm_cvtps_ph(x_vec, _MM_FROUND_TO_NEAREST_INT);
        _mm_storel_epi64((__m128i *)(y + i), y_vec);
    }
#elif defined(__riscv_zvfh)
    for (int vl; i < n; i += vl) {
        vl = __riscv_vsetvl_e32m2(n - i);
        vfloat32m2_t vx = __riscv_vle32_v_f32m2(&x[i], vl);
        vfloat16m1_t vy = __riscv_vfncvt_f_f_w_f16m1(vx, vl);
        __riscv_vse16_v_f16m1((_Float16 *)&y[i], vy, vl);
    }
#endif
    for (; i < n; ++i) {
        y[i] = GGML_CPU_FP32_TO_FP16(x[i]);
    }
}

void ggml_cpu_fp16_to_fp32(const ggml_fp16_t * x, float * y, int64_t n) {
    int64_t i = 0;
#if defined(__F16C__)
#if defined(__AVX512F__)
    for (; i + 15 < n; i += 16) {
        __m256i x_vec = _mm256_loadu_si256((const __m256i *)(x + i));
        __m512 y_vec = _mm512_cvtph_ps(x_vec);
        _mm512_storeu_ps(y + i, y_vec);
    }
#endif
    for (; i + 7 < n; i += 8) {
        __m128i x_vec = _mm_loadu_si128((const __m128i *)(x + i));
        __m256 y_vec = _mm256_cvtph_ps(x_vec);
        _mm256_storeu_ps(y + i, y_vec);
    }
    for (; i + 3 < n; i += 4) {
        __m128i x_vec = _mm_loadl_epi64((const __m128i *)(x + i));
        __m128 y_vec = _mm_cvtph_ps(x_vec);
        _mm_storeu_ps(y + i, y_vec);
    }

#elif defined(__riscv_v_intrinsic) && defined(__riscv_zvfhmin)
    // calculate step size
    const int epr = __riscv_vsetvlmax_e16m2();
    const int step = epr * 2;
    const int np = (n & ~(step - 1));

    // unroll by 2
    for (; i < np; i += step) {
        vfloat16m2_t ax0 = __riscv_vle16_v_f16m2((const _Float16*)x + i, epr);
        vfloat32m4_t ay0 = __riscv_vfwcvt_f_f_v_f32m4(ax0, epr);
        __riscv_vse32_v_f32m4(y + i, ay0, epr);

        vfloat16m2_t ax1 = __riscv_vle16_v_f16m2((const _Float16*)x + i + epr, epr);
        vfloat32m4_t ay1 = __riscv_vfwcvt_f_f_v_f32m4(ax1, epr);
        __riscv_vse32_v_f32m4(y + i + epr, ay1, epr);
    }

    // leftovers
    int vl;
    for (i = np; i < n; i += vl) {
        vl = __riscv_vsetvl_e16m2(n - i);
        vfloat16m2_t ax0 = __riscv_vle16_v_f16m2((const _Float16*)x + i, vl);
        vfloat32m4_t ay0 = __riscv_vfwcvt_f_f_v_f32m4(ax0, vl);
        __riscv_vse32_v_f32m4(y + i, ay0, vl);
    }

#endif

    for (; i < n; ++i) {
        y[i] = GGML_CPU_FP16_TO_FP32(x[i]);
    }
}

void ggml_cpu_fp32_to_bf16(const float * x, ggml_bf16_t * y, int64_t n) {
    int64_t i = 0;
    for (; i < n; ++i) {
        y[i] = GGML_FP32_TO_BF16(x[i]);
    }
}

void ggml_cpu_fp32_to_i32(const float * x, int32_t * y, int64_t n) {
    int64_t i = 0;
    for (; i < n; ++i) {
        y[i] = x[i];
    }
}

void ggml_cpu_bf16_to_fp32(const ggml_bf16_t * x, float * y, int64_t n) {
    int64_t i = 0;
#if defined(__AVX2__)
#if defined(__AVX512F__)
    for (; i + 15 < n; i += 16) {
        _mm512_storeu_ps(y + i,
                        _mm512_castsi512_ps(
                            _mm512_slli_epi32(
                                _mm512_cvtepu16_epi32(
                                    _mm256_loadu_si256(
                                        (const __m256i *)(x + i))),
                                16)));
    }
#endif
    for (; i + 7 < n; i += 8) {
        _mm256_storeu_ps(y + i,
                        _mm256_castsi256_ps(
                            _mm256_slli_epi32(
                                _mm256_cvtepu16_epi32(
                                    _mm_loadu_si128(
                                        (const __m128i *)(x + i))),
                                16)));
    }
#elif defined(__riscv_v_intrinsic) && defined(__riscv_zvfbfmin)
    // calculate step size
    const int epr = __riscv_vsetvlmax_e16m2();
    const int step = epr * 2;
    const int np = (n & ~(step - 1));

    // unroll by 2
    for (; i < np; i += step) {
        vbfloat16m2_t ax0 = __riscv_vle16_v_bf16m2((const __bf16*)x + i, epr);
        vfloat32m4_t ay0 = __riscv_vfwcvtbf16_f_f_v_f32m4(ax0, epr);
        __riscv_vse32_v_f32m4(y + i, ay0, epr);

        vbfloat16m2_t ax1 = __riscv_vle16_v_bf16m2((const __bf16*)x + i + epr, epr);
        vfloat32m4_t ay1 = __riscv_vfwcvtbf16_f_f_v_f32m4(ax1, epr);
        __riscv_vse32_v_f32m4(y + i + epr, ay1, epr);
    }

    // leftovers
    int vl;
    for (i = np; i < n; i += vl) {
        vl = __riscv_vsetvl_e16m2(n - i);
        vbfloat16m2_t ax0 = __riscv_vle16_v_bf16m2((const __bf16*)x + i, vl);
        vfloat32m4_t ay0 = __riscv_vfwcvtbf16_f_f_v_f32m4(ax0, vl);
        __riscv_vse32_v_f32m4(y + i, ay0, vl);
    }
#endif
    for (; i < n; i++) {
        y[i] = GGML_BF16_TO_FP32(x[i]);
    }
}

int ggml_cpu_has_avx(void) {
#if defined(__AVX__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_avx_vnni(void) {
#if defined(__AVXVNNI__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_avx2(void) {
#if defined(__AVX2__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_avx512(void) {
#if defined(__AVX512F__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_avx512_vbmi(void) {
#if defined(__AVX512VBMI__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_avx512_vnni(void) {
#if defined(__AVX512VNNI__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_avx512_bf16(void) {
#if defined(__AVX512BF16__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_amx_int8(void) {
#if defined(__AMX_INT8__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_bmi2(void) {
#if defined(__BMI2__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_fma(void) {
#if defined(__FMA__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_arm_fma(void) {
#if defined(__ARM_FEATURE_FMA)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_riscv_v(void) {
#if defined(__riscv_v_intrinsic)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_get_rvv_vlen(void) {
#if defined(__riscv) && defined(__riscv_v_intrinsic)
    return ggml_riscv_arch_features.rvv_vlen;
#else
    return 0;
#endif
}

int ggml_cpu_has_f16c(void) {
#if defined(__F16C__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_fp16_va(void) {
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_wasm_simd(void) {
#if defined(__wasm_simd128__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_llamafile(void) {
#if defined(GGML_USE_LLAMAFILE)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_sse3(void) {
#if defined(__SSE3__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_ssse3(void) {
#if defined(__SSSE3__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_vsx(void) {
#if defined(__POWER9_VECTOR__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_vxe(void) {
#if defined(__VXE__) || defined(__VXE2__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_neon(void) {
#if defined(__ARM_ARCH) && defined(__ARM_NEON)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_dotprod(void) {
#if defined(__ARM_ARCH) && defined(__ARM_FEATURE_DOTPROD)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_sve(void) {
#if defined(__ARM_ARCH) && defined(__ARM_FEATURE_SVE)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_matmul_int8(void) {
#if defined(__ARM_ARCH) && defined(__ARM_FEATURE_MATMUL_INT8)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_get_sve_cnt(void) {
#if defined(__ARM_ARCH) && defined(__ARM_FEATURE_SVE)
    return ggml_arm_arch_features.sve_cnt;
#else
    return 0;
#endif
}

int ggml_cpu_has_sme(void) {
#if defined(__ARM_ARCH) && defined(__ARM_FEATURE_SME)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_sme2(void) {
#if defined(__ARM_ARCH) && defined(__ARM_FEATURE_SME2)
    return 1;
#else
    return 0;
#endif
}

void ggml_cpu_init(void) {
    // needed to initialize ggml_time
    {
        struct ggml_init_params params = { 0, NULL, false };
        struct ggml_context * ctx = ggml_init(params);
        ggml_free(ctx);
    }

    ggml_critical_section_start();

    static bool is_first_call = true;

    if (is_first_call) {
        // initialize GELU, Quick GELU, SILU and EXP F32 tables
        {
            const uint64_t t_start = ggml_time_us(); UNUSED(t_start);

            for (int i = 0; i < (1 << 16); ++i) {
                union {
                    uint16_t u16;
                    ggml_fp16_t fp16;
                } u = {i};
                float f = GGML_COMPUTE_FP16_TO_FP32(u.fp16);
                ggml_table_f32_f16[i] = f;
                ggml_table_gelu_f16[i] = GGML_CPU_FP32_TO_FP16(ggml_gelu_f32(f));
                ggml_table_gelu_quick_f16[i] = GGML_CPU_FP32_TO_FP16(ggml_gelu_quick_f32(f));
            }

            // initialize E8M0 half table (256 entries)
            for (int i = 0; i < (1 << 8); ++i) {
                ggml_table_f32_e8m0_half[i] = GGML_E8M0_TO_FP32_HALF(i);
            }

            // initialize UE4M3 table (256 entries)
            for (int i = 0; i < (1 << 8); ++i) {
                ggml_table_f32_ue4m3[i] = ggml_ue4m3_to_fp32(i);
            }

            const uint64_t t_end = ggml_time_us(); UNUSED(t_end);

            GGML_PRINT_DEBUG("%s: GELU, Quick GELU, SILU and EXP tables initialized in %f ms\n", __func__, (t_end - t_start)/1000.0);

#ifdef GGML_USE_OPENMP
            //if (!getenv("OMP_WAIT_POLICY")) {
            //    // set the wait policy to active, so that OpenMP threads don't sleep
            //    setenv("OMP_WAIT_POLICY", "active", 0)
            //}

            if (!getenv("KMP_BLOCKTIME")) {
                // set the time to wait before sleeping a thread
                // this is less aggressive than setting the wait policy to active, but should achieve similar results in most cases
#ifdef _WIN32
                _putenv_s("KMP_BLOCKTIME", "200"); // 200ms
#else
                setenv("KMP_BLOCKTIME", "200", 0); // 200ms
#endif
            }
#endif
        }

#if defined(__ARM_ARCH)
        ggml_init_arm_arch_features();
#endif

#if defined(__riscv)
        ggml_init_riscv_arch_features();
#endif

        {
            const char * env = getenv("GGML_CPU_DISABLE_FUSION");
            ggml_cpu_disable_fusion = (env != NULL && atoi(env) == 1);
        }

        is_first_call = false;
    }

    ggml_critical_section_end();
}
