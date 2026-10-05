#include "binary-ops.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#if defined(_MSC_VER) || defined(__clang__)
#include <intrin.h>
#endif

// The P instrument is OFF by default and must stay that way: its branch sat inside the row loop, one
// execution per call, and with ne10 == 1 that is one execution per ELEMENT -- 120,187,008 of them in the
// ON arm and ZERO in the off arm, because the f16 destination only exists there. So it was not a symmetric
// branch: it added cycles to exactly one arm, which is the one shape of instrument error that survives a
// paired A/B. Measured consequence: the work profile named MUL +48.6% on a DLL that contained this branch,
// so that reading has to be reproduced on a build without it before it can be quoted. Enabling it takes a
// build with -DKATAGO_MUL_ST_INSTRUMENT=1, i.e. an instrument build that is never a timing build.
#ifndef KATAGO_MUL_ST_INSTRUMENT
#define KATAGO_MUL_ST_INSTRUMENT 0
#endif

// ---------------------------------------------------------------------------------------------------
// KATAGO_CPU_MUL_STORE_STATS=1 -- price the PRODUCER side of the f16 store, in situ, per element.
//
// Why in situ rather than by disassembly: the f16 contiguous loop compiles to three packed instructions
// per 8 elements (vmovups + vmulps + vcvtps2ph writing 16 B straight to memory, no stack round-trip, the
// alignas float[8] promoted into a register, the function-pointer `op` devirtualised). That bounds the
// ARITHMETIC near zero and says nothing about what a half-width destination costs the MEMORY SYSTEM --
// fewer bytes is a different write pattern, not obviously a cheaper one. The disassembly was read as
// "P ~ 0"; this counter is the check on that reading, and P is the number lead B's arithmetic rests on
// (d(mulf16) = +1.035 - P - D).
//
// Why this shape: BOTH contiguous store loops run in the SAME process, because the retype only touches
// some of the MUL nodes -- the rest keep an f32 destination. So the f32 loop and the f16 loop can be timed
// against each other with no cross-arm, cross-session or cross-DLL comparison anywhere, which is the only
// kind of difference this project has been able to trust. Cycles per ELEMENT is what is reported, so node
// sizes and thread counts cancel.
// ---------------------------------------------------------------------------------------------------
static inline uint64_t katago_mul_st_tsc(void) {
#if defined(__x86_64__) || defined(_M_X64)
    return __rdtsc();
#else
    return 0; // no cycle counter here; the element counts stay valid and the per-element cost reads 0
#endif
}
static int katago_mul_store_stats(void) {
    static int v = -1;
    if (v < 0) { v = (getenv("KATAGO_CPU_MUL_STORE_STATS") != NULL) ? 1 : 0; }
    return v;
}
// [mode][bucket] where mode 0 = f32 destination, 1 = f16, and bucket is the CALL LENGTH:
//   1 = n == 1        (the scalar tail: the 8-wide loop never runs, every element converts through the
//                      type-trait function pointer)
//   2 = 2 <= n < 8    (the scalar tail again, a few elements at a time)
//   3 = n >= 8        (the packed loop actually executes)
// Bucketing is not decoration. The first version of this counter reported ONE number per mode and it
// compared an all-n==1 population (every retyped tensor) against a mixed one (6.5 elements per call on
// average), whose per-call overhead is amortised over six times more elements -- so the reported P was
// mostly the difference in call shape, not in store cost. That is the same disease as reading the TU's
// `2ph` count as if it were the hot loop: a population that is not the population in question.
static uint64_t katago_mul_st_cyc  [2][4][GGML_MAX_N_THREADS];
static uint64_t katago_mul_st_elems[2][4][GGML_MAX_N_THREADS];
static uint64_t katago_mul_st_calls[2][4][GGML_MAX_N_THREADS];

static inline int katago_mul_st_bucket(int64_t n) { return n <= 1 ? 1 : (n < 8 ? 2 : 3); }

extern "C" void katago_mul_store_report(void) {
    static const char * const mode_name[2] = { "f32 dst", "f16 dst" };
    uint64_t c[2][4] = {{0}}, e[2][4] = {{0}}, k[2][4] = {{0}};
    for (int t = 0; t < GGML_MAX_N_THREADS; t++) {
        for (int m = 0; m < 2; m++) {
            for (int b = 1; b < 4; b++) {
                c[m][b] += katago_mul_st_cyc  [m][b][t];
                e[m][b] += katago_mul_st_elems[m][b][t];
                k[m][b] += katago_mul_st_calls[m][b][t];
            }
        }
    }
    if (k[0][1] + k[0][2] + k[0][3] + k[1][1] + k[1][2] + k[1][3] == 0) { return; }
    fprintf(stderr, "[mul-store] contiguous store loops, ONE process, bucketed by CALL LENGTH "
                    "(n==1 scalar tail | 2<=n<8 | n>=8 packed loop):\n");
    for (int m = 0; m < 2; m++) {
        for (int b = 1; b < 4; b++) {
            if (k[m][b] == 0) { continue; }
            fprintf(stderr, "[mul-store]   %s bucket %d: calls=%llu elems=%llu cycles=%llu (%.3f cyc/elem)\n",
                    mode_name[m], b, (unsigned long long) k[m][b], (unsigned long long) e[m][b],
                    (unsigned long long) c[m][b], e[m][b] ? (double) c[m][b] / (double) e[m][b] : 0.0);
        }
    }
    // Like with like: the n==1 bucket on both sides. That is the pair the retyped MULs actually live in.
    if (e[0][1] && e[1][1]) {
        const double a = (double) c[0][1] / (double) e[0][1];
        const double b = (double) c[1][1] / (double) e[1][1];
        fprintf(stderr, "[mul-store] P (n==1 vs n==1) = %+.3f cyc/elem  =>  %+.1f Mcycles over the "
                        "15,023,376 retyped elements  (11.9 Gcycles of thread span = 1%%)\n", b - a,
                (b - a) * 15.023376);
    } else {
        fprintf(stderr, "[mul-store] P NOT READABLE: one side has no n==1 calls, so the like-for-like pair "
                        "does not exist in this run\n");
    }
}

// Debug-only hook, defined (with C linkage) in ops.cpp. With KATAGO_CPU_ROPE_HASH=1 the ADD kernel hashes
// its own output block so it can be compared, inside a single run, against what the fused RoPE kernel wrote.
extern "C" int (*ggml_cpu_rope_dbg_check)(uint64_t h);
// ROPE_SHADOW independent reference: hand the stock ADD's result to ops.cpp, which diffs it against what
// the fused kernel published. Only thread 0 calls it; the arming/disarming is single-threaded by design.
extern "C" void ggml_cpu_rope_shadow_cmp(const float * got, int64_t ne0, int64_t ne1, const void * key);

#if defined(GGML_USE_ACCELERATE)
#include <Accelerate/Accelerate.h>

using vDSP_fn_t = void (*)(const float *, vDSP_Stride, const float *, vDSP_Stride, float *, vDSP_Stride, vDSP_Length);
#endif

static inline float op_add(float a, float b) {
    return a + b;
}

static inline float op_sub(float a, float b) {
    return a - b;
}

static inline float op_mul(float a, float b) {
    return a * b;
}

static inline float op_div(float a, float b) {
    return a / b;
}

template <float (*op)(float, float), typename src0_t, typename src1_t, typename dst_t>
static inline void vec_binary_op_contiguous(const int64_t n, dst_t * z, const src0_t * x, const src1_t * y) {
    constexpr auto src0_to_f32 = type_conversion_table<src0_t>::to_f32;
    constexpr auto src1_to_f32 = type_conversion_table<src1_t>::to_f32;
    constexpr auto f32_to_dst  = type_conversion_table<dst_t >::from_f32;

    for (int i = 0; i < n; i++) {
        z[i] = f32_to_dst(op(src0_to_f32(x[i]), src1_to_f32(y[i])));
    }
}

template <float (*op)(float, float), typename src0_t, typename src1_t, typename dst_t>
static inline void vec_binary_op_non_contiguous(const int64_t n, const int64_t ne10, const int64_t nb10, dst_t * z, const src0_t * x, const src1_t * y) {
    constexpr auto src0_to_f32 = type_conversion_table<src0_t>::to_f32;
    constexpr auto src1_to_f32 = type_conversion_table<src1_t>::to_f32;
    constexpr auto f32_to_dst  = type_conversion_table<dst_t >::from_f32;

    for (int i = 0; i < n; i++) {
        int i10 = i % ne10;
        const src1_t * y_ptr = (const src1_t *)((const char *)y + i10*nb10);
        z[i] = f32_to_dst(op(src0_to_f32(x[i]), src1_to_f32(*y_ptr)));
    }
}

// ---------------------------------------------------------------------------------------------------
// PACKED f32 -> f16 store for the (float, float, ggml_fp16_t) case, used by KATAGO_CPU_MUL_F16.
//
// WHY this exists: the generic path stores through type_conversion_table<ggml_fp16_t>::from_f32, whose
// definition depends on whether __F16C__ was in effect when simd-mappings.h was preprocessed. MEASURED:
// binary-ops.cpp.obj held ZERO '2ph' instructions while ggml-cpu.c.obj -- where the repack's conversion
// lives -- held seven. The same rounding operation therefore cost ~5 cycles/element here and ~nothing
// there, and for the 15,023,376 elements this change retypes that is ~3.1 ms/batch against a total
// saving of 0.6-1.2 ms. The conversion has to be PACKED, not merely present in the instruction set: the
// loop below rounds eight elements per instruction.
//
// imm=0 is round-to-nearest-even, which is what GGML_CPU_FP32_TO_FP16 does, so this is bit-identical to
// the conversion it replaces -- and the bit gate proves that, not this comment.
// ---------------------------------------------------------------------------------------------------
template <float (*op)(float, float)>
static inline void vec_binary_op_contiguous_f16out(const int64_t n, ggml_fp16_t * z, const float * x, const float * y) {
#if defined(__F16C__)
#pragma message("KATAGO MUL->F16 store: PACKED (F16C)")
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        alignas(32) float r[8];
        for (int k = 0; k < 8; k++) {
            r[k] = op(x[i + k], y[i + k]);
        }
        const __m128i h = _mm256_cvtps_ph(_mm256_load_ps(r), 0);
        _mm_storeu_si128((__m128i *)(z + i), h);
    }
    for (; i < n; i++) {
        z[i] = type_conversion_table<ggml_fp16_t>::from_f32(op(x[i], y[i]));
    }
#else
#pragma message("KATAGO MUL->F16 store: scalar fallback (no __F16C__ in this translation unit)")
    for (int64_t i = 0; i < n; i++) {
        z[i] = type_conversion_table<ggml_fp16_t>::from_f32(op(x[i], y[i]));
    }
#endif
}

// ---------------------------------------------------------------------------------------------------
// KATAGO_CPU_MUL_BCAST=1 -- the variant that removes the per-ELEMENT call structure of a broadcast mul.
//
// What the ablation is for: the per-op work profile says MUL is +51..67% on the retype arm in every run
// while every other big op sits within ~1.4% of its counterpart. Inside MUL the cost is NOT the conversion
// instruction -- `type_conversion_table<ggml_fp16_t>::from_f32` is GGML_CPU_FP32_TO_FP16, which
// simd-mappings.h:58-64 defines as `_cvtss_sh(x, 0)` whenever __F16C__ is set, and ggml-cpu-impl.h:50-51
// force __F16C__ on for AVX2/AVX512. So the scalar tail already uses the hardware conversion and swapping
// it would change nothing.
//
// What is left is the STRUCTURE. Every retyped tensor is a broadcast multiply with ne10 == 1, i.e. the
// source has ONE element for the whole row, so `vec_binary_op_contiguous_f16out` is entered once per
// ELEMENT with n == 1: the 8-wide loop never runs and each element pays a call, a one-iteration loop and
// the pointer arithmetic around it. 120,187,008 elements per process.
//
// Why a variant and not a segmented instrument: instrumenting four segments inside a path that is entered
// once per element would cost 120M x 4 x ~40 cycles ~= 19 Gcycles, which is the entire thread span. The
// measurement would be the instrument, and this time it would contaminate BOTH arms, so the symmetry gate
// could not catch it either. A variant costs nothing to measure: the existing work profile already reports
// the MUL row.
//
// Why an env rather than a separate build: it keeps V1 and V4 inside ONE dll, so the comparison does not
// carry a code-layout or ASLR difference on top of the thing being measured -- which matters here because
// the same retype arm has already been measured 1.5 pp apart across two builds of the same source.
//
// Bit identity is NOT inherited from the packed path's earlier gate: that gate covered whichever tensors
// happened to pass through the 8-wide loop, and these broadcast tensors never did. The conversion is the
// same round-to-nearest-even operation, so the expectation is that the hash does not move -- but the gate
// has to be run on THIS population (and if it moves, subnormals are the first place to look).
// FOLDED INTO THE F16 WRITE-BACK PATH (landed). It is no longer a mode: with ne10 == 1 the loop below
// degenerates into one call per ELEMENT, each running a one-iteration loop, and this replaces the whole
// row-set with a single packed loop. Measured on the same arm, same DLL, with element totals verified
// identical: MUL work 1,615,713,548 -> 565,797,374 cycles (-65.0%), and the wall-clock A/B that followed
// turned the retype from -4.56% into +5.49% (6/6, every gate PASS, bit-identical on this population).
//
// Default ON, with a way back (G-22): KATAGO_CPU_MUL_BCAST_OFF=1 restores the per-element form. The old
// opt-in name KATAGO_CPU_MUL_BCAST is still ACCEPTED and is now a no-op, deliberately: the A/B script that
// measured it passes that name on the ON arm, and keeping it harmless means that script now measures the
// retype with this fold already baked into BOTH arms -- which is exactly the confirmation run required
// before the retype itself becomes the default.
//
// It is applied only where the destination is f16 (the if-constexpr below): the f32 destination has its own
// path and the earlier reading that the knob was "slightly negative" there was run-to-run scatter, since
// this code cannot be reached with dst_t == float at all.
static int katago_mul_bcast_off(void) {
    static int v = -1;
    if (v < 0) { v = (getenv("KATAGO_CPU_MUL_BCAST_OFF") != NULL) ? 1 : 0; }
    return v;
}

// One packed loop over the whole row instead of one call per element. The body deliberately applies `op`
// element by element inside the 8-wide group rather than calling _mm256_mul_ps: that keeps the helper
// correct for any op, and for op_mul clang still folds it into vmovups + vmulps + vcvtps2ph (exactly the
// three-instruction loop the packed path compiles to), so nothing is given up for the generality.
template <float (*op)(float, float)>
static inline void katago_mul_bcast_f16out(const int64_t n, ggml_fp16_t * z, const float * x, const float * y) {
#if defined(__F16C__)
    const float s = y[0];
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        alignas(32) float r[8];
        for (int k = 0; k < 8; k++) {
            r[k] = op(x[i + k], s);
        }
        _mm_storeu_si128((__m128i *)(z + i), _mm256_cvtps_ph(_mm256_load_ps(r), 0));
    }
    for (; i < n; i++) {
        z[i] = GGML_CPU_FP32_TO_FP16(op(x[i], s));
    }
#else
    for (int64_t i = 0; i < n; i++) {
        z[i] = GGML_CPU_FP32_TO_FP16(op(x[i], y[0]));
    }
#endif
}

template <float (*op)(float, float), typename src0_t, typename src1_t, typename dst_t>
static void apply_binary_op(const ggml_compute_params * params, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_can_repeat(src1, src0) && ggml_are_same_shape(src0, dst));

    GGML_TENSOR_BINARY_OP_LOCALS

    GGML_ASSERT( nb0 == sizeof(dst_t));
    GGML_ASSERT(nb00 == sizeof(src0_t));

    const auto [ir0, ir1] = get_thread_range(params, src0);
    const bool is_src1_contiguous_rows = ggml_is_contiguous_rows(src1);

#ifdef GGML_USE_ACCELERATE
    vDSP_fn_t vDSP_op = nullptr;
    // TODO - avoid the f32-only check using type 'trait' lookup tables and row-based src-to-float conversion functions
    if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
        if (op == op_add) {
            vDSP_op = vDSP_vadd;
        } else if (op == op_sub) {
            vDSP_op = vDSP_vsub;
        } else if (op == op_mul) {
            vDSP_op = vDSP_vmul;
        } else if (op == op_div) {
            vDSP_op = vDSP_vdiv;
        }
    }
#endif

    for (int64_t ir = ir0; ir < ir1; ++ir) {
        const int64_t i03 = ir/(ne02*ne01);
        const int64_t i02 = (ir - i03*ne02*ne01)/ne01;
        const int64_t i01 = (ir - i03*ne02*ne01 - i02*ne01);

        const int64_t i13 = i03 % ne13;
        const int64_t i12 = i02 % ne12;
        const int64_t i11 = i01 % ne11;

        dst_t        * dst_ptr  = (dst_t  *)       ((char *)       dst->data  + i03*nb3  + i02*nb2  + i01*nb1 );
        const src0_t * src0_ptr = (const src0_t *) ((const char *) src0->data + i03*nb03 + i02*nb02 + i01*nb01);
        const src1_t * src1_ptr = (const src1_t *) ((const char *) src1->data + i13*nb13 + i12*nb12 + i11*nb11);

        // Shadow diff is gated by the KEY, not by a size heuristic: an earlier version sat behind
        // ne00 > 1000 and therefore never fired for the A1 epilogue shapes (ne0 = 192/384), which looked
        // exactly like "the fusion is fine". Safety now comes from the key match inside shadow_cmp.
        if (params->ith == 0 && ir == 0 && op == op_add) {
            if constexpr (std::is_same_v<dst_t, float>) {
                ggml_cpu_rope_shadow_cmp((const float *) dst->data, ne00, ggml_nrows(dst), (const void *) dst);
            }
        }

        if (params->ith == 0 && ir == 0 && ne00 > 1000 && op == op_add) {
            if constexpr (std::is_same_v<dst_t, float>) {
                if (ggml_cpu_rope_dbg_check != nullptr) {
                    // Address-independent hash, same shape as the fused kernel's, for KATAGO_CPU_ROPE_HASH.
                    uint64_t h = 1469598103934665603ULL;
                    for (int64_t r = 0; r < ggml_nrows(dst); r++) {
                        const int64_t r03 = r/(ne02*ne01);
                        const int64_t r02 = (r - r03*ne02*ne01)/ne01;
                        const int64_t r01 = (r - r03*ne02*ne01 - r02*ne01);
                        for (int64_t c = 0; c < ne00; ++c) {
                            const unsigned char * q = (const unsigned char *) dst->data + r03*nb3 + r02*nb2 + r01*nb1 + c*nb0;
                            for (int64_t k = 0; k < (int64_t) sizeof(float); k++) {
                                h ^= q[k];
                                h *= 1099511628211ULL;
                            }
                        }
                    }
                    ggml_cpu_rope_dbg_check(h);
                }
            }
        }

        if (is_src1_contiguous_rows) {
            // src1 is broadcastable across src0 and dst in i1, i2, i3
            const int64_t nr0 = ne00 / ne10;

            // Folded into the F16 write-back path -- no longer a mode. See the note above
            // katago_mul_bcast_off(); KATAGO_CPU_MUL_BCAST_OFF=1 is the way back (G-22).
            if (ne10 == 1 && !katago_mul_bcast_off()) {
                if constexpr (std::is_same_v<src0_t, float> && std::is_same_v<dst_t, ggml_fp16_t>) {
                    katago_mul_bcast_f16out<op>(ne00, dst_ptr, src0_ptr, src1_ptr);
                    continue;
                }
            }

            for (int64_t r = 0; r < nr0; ++r) {
#ifdef GGML_USE_ACCELERATE
                if constexpr (std::is_same_v<src0_t, float> && std::is_same_v<src1_t, float> && std::is_same_v<dst_t, float>) {
                    if (vDSP_op != nullptr) {
                        vDSP_op(src1_ptr, 1, src0_ptr + r*ne10, 1, dst_ptr + r*ne10, 1, ne10);
                        continue;
                    }
                }
#endif
                if constexpr (std::is_same_v<src0_t, float> && std::is_same_v<src1_t, float> &&
                              std::is_same_v<dst_t, ggml_fp16_t>) {
                    // KATAGO_CPU_MUL_F16's store: packed, because the generic path's conversion is not.
                    // See the note above vec_binary_op_contiguous_f16out.
#if KATAGO_MUL_ST_INSTRUMENT
                    if (katago_mul_store_stats()) {
                        const int stb = katago_mul_st_bucket(ne10);
                        const uint64_t st0 = katago_mul_st_tsc();
                        vec_binary_op_contiguous_f16out<op>(ne10, dst_ptr + r*ne10, src0_ptr + r*ne10, src1_ptr);
                        katago_mul_st_cyc  [1][stb][params->ith] += katago_mul_st_tsc() - st0;
                        katago_mul_st_elems[1][stb][params->ith] += (uint64_t) ne10;
                        katago_mul_st_calls[1][stb][params->ith] ++;
                    } else
#endif
                    {
                        vec_binary_op_contiguous_f16out<op>(ne10, dst_ptr + r*ne10, src0_ptr + r*ne10, src1_ptr);
                    }
                } else if constexpr (std::is_same_v<src0_t, float> && std::is_same_v<src1_t, float> &&
                                     std::is_same_v<dst_t, float>) {
                    // The f32 counterpart, timed by the same counter so P is a same-process difference.
#if KATAGO_MUL_ST_INSTRUMENT
                    if (katago_mul_store_stats()) {
                        const int stb = katago_mul_st_bucket(ne10);
                        const uint64_t st0 = katago_mul_st_tsc();
                        vec_binary_op_contiguous<op>(ne10, dst_ptr + r*ne10, src0_ptr + r*ne10, src1_ptr);
                        katago_mul_st_cyc  [0][stb][params->ith] += katago_mul_st_tsc() - st0;
                        katago_mul_st_elems[0][stb][params->ith] += (uint64_t) ne10;
                        katago_mul_st_calls[0][stb][params->ith] ++;
                    } else
#endif
                    {
                        vec_binary_op_contiguous<op>(ne10, dst_ptr + r*ne10, src0_ptr + r*ne10, src1_ptr);
                    }
                } else {
                    vec_binary_op_contiguous<op>(ne10, dst_ptr + r*ne10, src0_ptr + r*ne10, src1_ptr);
                }
            }
        } else {
            vec_binary_op_non_contiguous<op>(ne0, ne10, nb10, dst_ptr, src0_ptr, src1_ptr);
        }
    }
}

// TODO: Use the 'traits' lookup table (for type conversion fns), instead of a mass of 'if' conditions with long templates
template <float (*op)(float, float)>
static void binary_op(const ggml_compute_params * params, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    /*  */ if (src0->type == GGML_TYPE_F32  && src1->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_F32) { // all f32
        apply_binary_op<op, float, float, float>(params, dst);
    } else if (src0->type == GGML_TYPE_F32  && src1->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_F16) {
        // KATAGO_CPU_MUL_F16 (R2 batch 1): f32 inputs, f16 STORE. The op still runs in f32 and only the
        // store rounds -- through type_conversion_table<ggml_fp16_t>::from_f32, i.e. the very conversion
        // mul_mat applies when it converts this activation to its vec_dot_type. A GEMM that reads this
        // tensor as src1 therefore gets the exact bits it used to compute for itself.
        apply_binary_op<op, float, float, ggml_fp16_t>(params, dst);
    } else if (src0->type == GGML_TYPE_F16  && src1->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F16) { // all f16
        apply_binary_op<op, ggml_fp16_t, ggml_fp16_t, ggml_fp16_t>(params, dst);
    } else if (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_BF16) { // all bf16
        apply_binary_op<op, ggml_bf16_t, ggml_bf16_t, ggml_bf16_t>(params, dst);
    } else if (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_BF16) {
        apply_binary_op<op, ggml_bf16_t, float, ggml_bf16_t>(params, dst);
    } else if (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_F32) {
        apply_binary_op<op, ggml_bf16_t, float, float>(params, dst);
    } else if (src0->type == GGML_TYPE_F16  && src1->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_F16) {
        apply_binary_op<op, ggml_fp16_t, float, ggml_fp16_t>(params, dst);
    } else if (src0->type == GGML_TYPE_F16  && src1->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_F32) {
        apply_binary_op<op, ggml_fp16_t, float, float>(params, dst);
    } else {
        GGML_ABORT("%s: unsupported types: dst: %s, src0: %s, src1: %s\n", __func__,
            ggml_type_name(dst->type), ggml_type_name(src0->type), ggml_type_name(src1->type));
    }
}

void ggml_compute_forward_add_non_quantized(const ggml_compute_params * params, ggml_tensor * dst) {
    binary_op<op_add>(params, dst);
}

void ggml_compute_forward_sub(const ggml_compute_params * params, ggml_tensor * dst) {
    binary_op<op_sub>(params, dst);
}

void ggml_compute_forward_mul(const ggml_compute_params * params, ggml_tensor * dst) {
    binary_op<op_mul>(params, dst);
}

void ggml_compute_forward_div(const ggml_compute_params * params, ggml_tensor * dst) {
    binary_op<op_div>(params, dst);
}
