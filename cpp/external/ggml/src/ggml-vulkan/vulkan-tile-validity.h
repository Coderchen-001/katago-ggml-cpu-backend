// vulkan-tile-validity.h -- pure tile-validity rules for the Vulkan tuner.
//
// WHY THIS HEADER EXISTS (t3-shell-plan.md section 1.7)
// Legality used to live in three places: the rule table in t1-predicate.md, the loader
// check in ggml_vk_tuned_row_check(), and the enumerator in the T3 shell.  That is a
// standing invitation to "the enumerator let it through but the loader rejected it" (or
// the reverse).  The rules are the backend's -- they mirror what the create path and the
// shaders actually accept -- so they live here, dependency-free, and both sides include
// this header:
//
//   * the backend: the KATAGO_VK_MM_TILE=custom: parser
//   * the shell (KataGO side): the candidate enumerator and the IMPORT parser
//
// Deliberately small API (authority order: class, then reason, then canonical key):
//   1. vk_tile_candidate_reason() -- is this concrete geometry legal, and if not, why
//   2. vk_tile_canonical_key()    -- dedup identity (dedup is NOT validity)
//   3. vk_tile_axis_class()       -- per-path axis classification (the AUTHORITY);

//      vk_tile_axis_is_live() is a thin wrapper answering only "may it be swept?"
//
// No vk::, no ggml, no std::string: plain ints in, plain const char* out.  That is what
// lets both sides share it without either one reaching into the other.

#pragma once

#include <stdint.h>

// Which creation path the geometry is destined for: the rules differ per path.
enum vk_tile_path {
    VK_TILE_PATH_SCALAR = 0,  // no cooperative matrix (f16/f32 scalar GEMM)
    VK_TILE_PATH_CM1    = 1,  // VK_KHR_cooperative_matrix
    VK_TILE_PATH_CM2    = 2,  // NV_cooperative_matrix2 (cm2 shader)
};

// Axis ids for vk_tile_axis_class().  These follow the REAL spec layout of mul_mm.comp
// (verified 2026-09-13), not a tuner invention:
//   warptile = { BS, BM, BN, BK, WM, WN, WMITER, TM, TN, TK, WARP }
//     0=BLOCK_SIZE 1=BM 2=BN [3=BK, quant branch only] 4=WM 5=WN 6=WMITER 7=TM 8=TN
//     9=TK ("Only needed for coopmat") 10=WARP
//   11=ALIGNED (host-set, never swept)  12=SHMEM_STRIDE_PAD  13=APPLY_SLM_A_RESHAPE
//     -> note there is NO "10=PAD" slot; PAD is spec 12.
//   cm2's own spec is {BLOCK_SIZE, BM, BN, BK, smaller}, where `smaller` is spec id 4
//   in the shader -- the same number as scalar WM, which is exactly why axis ids must
//   NOT be reused across paths: this enum keeps distinct numbers for distinct meanings,
//   and 14 was chosen for SMALLER because 11 is ALIGNED.
enum vk_tile_axis {
    VK_TILE_AXIS_BS      = 0,
    VK_TILE_AXIS_BM      = 1,
    VK_TILE_AXIS_BN      = 2,
    VK_TILE_AXIS_BK      = 3,
    VK_TILE_AXIS_WM      = 4,
    VK_TILE_AXIS_WN      = 5,
    VK_TILE_AXIS_WMITER  = 6,
    VK_TILE_AXIS_TM      = 7,
    VK_TILE_AXIS_TN      = 8,
    VK_TILE_AXIS_TK      = 9,
    VK_TILE_AXIS_WARP    = 10,
    VK_TILE_AXIS_SPEC_PAD     = 12,
    VK_TILE_AXIS_SPEC_RESHAPE = 13,
    VK_TILE_AXIS_SMALLER = 14,   // cm2 only (spec id 4 in the shader)
};

// What an axis means on a given path.  Two-state "live/dead" was not enough: an axis
// can be a free enumeration axis, a value that must be filled in correctly but must not
// be swept, a derived quantity, or simply absent on that path.
enum vk_tile_axis_class {
    VK_TILE_AXIS_INERT        = 0,  // does not exist on this path (sweeping it is a no-op)
    VK_TILE_AXIS_ENUMERABLE   = 1,  // free axis: may be swept
    VK_TILE_AXIS_PINNED       = 2,  // must be set correctly, must NOT be swept
    VK_TILE_AXIS_DERIVED      = 3,  // computed from other axes
    VK_TILE_AXIS_NOT_APPLICABLE = 4 // not an axis of this path at all
};

// One concrete tile geometry.  Scalar / cm1 use the warptile fields; cm2 uses
// BM, BN, BK, BS (BLOCK_SIZE) and `smaller`.  Unset spec constants are -1.
struct vk_tile_geometry {
    uint32_t BM, BN, BK, WM, WN, WMITER, TM, TN, BS;
    int pad;      // -1 = unspecified (shader default applies)
    int reshape;  // -1 = unspecified
    int smaller;  // -1 = unspecified (cm2 only)
};

// The wire form of ONE tuned row: identity + geometry, and nothing else.  No handles and
// no pointers (t3-shell-plan.md section 7.1-3): the backend resolves the geometry into
// pipelines when the row is installed, so a row is something the cache can store verbatim
// and the shell can hand across the interface without either side knowing the other's
// types.  It lives in this header so both sides share ONE definition of a row -- two
// definitions of a wire format is exactly the drift this header exists to prevent.
//
// `src0_type`/`src1_type` are ggml_type values carried as int on purpose: this header
// knows nothing about ggml, and the backend is the only side that has to interpret them.
// `key_m`/`key_k` are 0 for "whole slot" (matches any shape), which is the v1 form; see
// ggml_vk_tuned_row_matches for the matching rule.
struct vk_tune_row_spec {
    int src0_type;
    int src1_type;
    uint32_t key_m;
    uint32_t key_k;
    int path;                  // vk_tile_path
    struct vk_tile_geometry geom;
};

// ---------------------------------------------------------------------------
// B1 (t3-shell-plan.md section 6.7): install-time outcome reporting.
//
// A row set is USER DATA (a cache file or KATAGO_VK_TUNE_IMPORT), and the loader has
// to be able to say "declared N, installed M, dropped for this reason".  Two reasons
// why a single drop counter is not enough:
//
//   1. "every row in the file is hit at least once" is the acceptance for "no row was
//      tuned for nothing", and its denominator must be M (the rows that installed): a
//      row that cannot install can never be hit, so grading against N would make that
//      leg red for a reason that has nothing to do with tuning.
//   2. the write side (step 5) must write only rows this process really installed, so
//      the reason is what tells a transient failure from a permanent one.
//
// The reason set is closed on purpose: these counts reach the log and (later) the
// durable-effect decision, so free text must not be able to enter them.
enum vk_tune_row_drop {
    VK_TUNE_ROW_DROP_NONE         = 0,
    VK_TUNE_ROW_DROP_ILLEGAL      = 1,  // vk_tile_candidate_reason() refused it
    VK_TUNE_ROW_DROP_UNSUPPORTED  = 2,  // family / path / capability (I6) not usable
    VK_TUNE_ROW_DROP_BUILD_FAILED = 3,  // the pipeline could not be created
};

// Row-level counts, and they are an EQUALITY:
//
//     installed + dropped_illegal + dropped_unsupported + dropped_build_failed == declared
//
// A row is counted as installed when it landed on at least one non-empty target object
// (union, so a row that is usable is never thrown away), and it is counted as dropped only
// when it landed NOWHERE.  That is what keeps the three numbers addable: an earlier shape
// counted `installed` per row but the drops per TARGET, so one row whose two accumulator
// variants failed differently could push installed + dropped past declared.
//
// The per-target detail lives in two fields that are deliberately OUTSIDE that equality
// (they describe one row twice, they do not partition the declared set):
//
//   * partial        -- rows that installed, but not on every non-empty target (a SUBSET
//                       flag of `installed`, not a third bucket: a third bucket would
//                       reopen the "which side does it fall on" question)
//   * variant_failed -- (row, target) build failures
//
// Why "installed" is a union and the authority for "was it really used" is the HIT, not any
// install-time predicate: the accumulator variant is chosen PER NODE from dst->op_params[0],
// and KATAGO_VK_MMPREC_F32 (a policy knob, i.e. a legitimate production configuration) can
// rewrite the whole graph or a subset of nodes (vulkanbackend.cpp:2748-2758), so both
// variants may be in use inside one process and no predicate evaluated at install time can
// name "the" required one.  A dead row -- installed on a variant production never resolves,
// or on a variant that is empty while the other is not -- is what the per-row hit assertion
// catches.
struct vk_tune_install_result {
    unsigned declared;
    unsigned installed;
    unsigned dropped_illegal;
    unsigned dropped_unsupported;
    unsigned dropped_build_failed;
    unsigned partial;          // subset of `installed` (see above)
    unsigned variant_failed;   // (row, target) build failures (see above)
};

// ---------------------------------------------------------------------------
// Which (family, path) combinations have an implementation at all.
//
// Same sharing rule as vk_tile_candidate_reason: the enumerator and the loader MUST
// answer this identically, because a candidate the loader would always drop is a
// permanent loud line per run -- i.e. "the enumerator proposed something that can
// never install" is a bug in the enumerator, not a device property.
//
// The ggml_type numbers are transcribed deliberately (this header cannot include
// ggml): GGML_TYPE_F32 = 0, GGML_TYPE_F16 = 1.  The backend asserts both values at
// compile time (see ggml_backend_vk_tune_install_rows), so a renumbering upstream
// becomes a build error instead of a wrong-family install.
enum vk_tune_src_type {
    VK_TUNE_SRC_F32 = 0,
    VK_TUNE_SRC_F16 = 1,
};

// Returns nullptr when a row with this (family, path) has an implementation, else a
// static reason naming the missing piece.  `path` is only checked for existence here;
// whether this DEVICE can use that path is capability (I6), which is the backend's to
// answer at install time -- the shell must not guess it from the fingerprint.
static inline const char * vk_tune_row_family_reason(int src0_type, int src1_type, int path) {
    const int f16_f16 = (src0_type == VK_TUNE_SRC_F16 && src1_type == VK_TUNE_SRC_F16);
    const int f16_f32 = (src0_type == VK_TUNE_SRC_F16 && src1_type == VK_TUNE_SRC_F32);
    if (!f16_f16 && !f16_f32) {
        return "no tuned-row implementation for this (src0, src1) family "
               "(only f16xf16 and f16xf32 are wired)";
    }
    if (path == VK_TILE_PATH_CM1) {
        return "cm1 rows are not wired (no cm1 machine to measure on; refusing beats "
               "installing an unmeasured geometry)";
    }
    if (path != VK_TILE_PATH_SCALAR && path != VK_TILE_PATH_CM2) {
        return "unknown creation path";
    }
    return nullptr;
}

// The ONE place that names where a wave/subgroup size comes from.  On the scalar path stock fills
// the warptile's WARP field with subgroup_size_8 = max(device->subgroup_size, 8)
// (ggml-vulkan.cpp:4735-4742, used at :4823-4825 and :5854-5855).  The row build and the custom
// probe must put the SAME value into their warptile, so that the predicate, the build and stock
// agree by construction; writing max(subgroup_size, 8) at each call site is how two truths are born.
static inline uint32_t vk_tile_wave_for(uint32_t device_subgroup_size) {
    return device_subgroup_size > 8 ? device_subgroup_size : 8;
}

// Returns nullptr when the geometry is legal, otherwise a static reason string naming
// the first violated clause.  Callers log it as-is, which is what makes the enumerator
// and the loader produce the SAME reason for the same bad input.
static inline const char * vk_tile_candidate_reason(int path, const vk_tile_geometry & g, uint32_t wave) {
    // The wave (subgroup) size is an INPUT, never a constant: on a wave64 device the stock scalar
    // warptiles set the warptile's WARP field to 64 (subgroup_size_8 = max(device->subgroup_size, 8),
    // ggml-vulkan.cpp:4735-4742, used at :4823-4825 and :5854-5855), so a hardcoded 32 here would
    // refuse every legal wave64 geometry and accept illegal ones -- and for BS "refused" is the
    // silent-wrong-answer direction.  Callers pass vk_tile_wave_for(device->subgroup_size).  Zero is
    // a caller-contract violation and is refused rather than divided by (fail-closed, same style as
    // the clause list below: an unknown input is never "legal").
    // wave == 0 is refused for EVERY path, including cm2.  cm2's spec (BLOCK_SIZE/BM/BN/BK/smaller)
    // has no wave semantics at all, so this is a deliberate over-refusal: the wave is a property of
    // the DEVICE, and "this device's wave is unknown" means the row cannot be attributed to a device
    // class yet -- refusing it is the safe direction (the row is dropped loudly, not judged wrongly).
    //
    // Fail-closed vs fail-open, side by side, so nobody reads them as a contradiction:
    //   * the JUDGEMENT side (here) is fail-closed -- an unknown input is never "legal";
    //   * the FUNCTION side is fail-open -- the shell refuses to tune at all (no rows injected) and
    //     the production path keeps the stock heuristic.
    if (wave == 0) {
        return "wave == 0 (the device's wave/subgroup size must be known before validating)";
    }
    if (path == VK_TILE_PATH_CM2) {
        // cm2's spec is {BLOCK_SIZE, BM, BN, BK, enable_smaller_matrices}
        // (mul_mm_cm2.comp:31-36).  The scalar warptile clauses below do NOT describe
        // it, so this branch stands alone.  Bounds beyond these sanity checks (BK
        // sweep verdict, bucket membership) belong to the candidate table, not here.
        if (g.BM == 0 || g.BN == 0 || g.BK == 0 || g.BS == 0) {
            return "cm2: zero dimension (BM/BN/BK/BLOCK_SIZE must be > 0)";
        }
        if (g.BS > 1024) {
            return "cm2: BLOCK_SIZE > 1024";
        }
        if (g.smaller >= 0 && g.smaller != 0 && g.smaller != 1) {
            return "cm2: smaller not in {0,1}";
        }
        if (g.smaller == 1 && (g.BN % 2) != 0) {
            return "cm2: smaller=1 needs an even BN (effective BN = BN/2)";
        }
        return nullptr;
    }

    // Scalar / cm1 warptile family.  Clauses transcribed from the
    // KATAGO_VK_MM_TILE=custom: parser (same accept/reject set; the leading zero check
    // also removes a divide-by-zero the original expression had on WM/WN == 0).
    if (g.BM == 0 || g.BN == 0 || g.WM == 0 || g.WN == 0 || g.WMITER == 0 || g.TM == 0 || g.TN == 0) {
        return "zero dimension (BM/BN/WM/WN/WMITER/TM/TN must be > 0)";
    }
    const uint32_t num_warps = (g.BM / g.WM) * (g.BN / g.WN);
    const uint32_t wniter = g.WM * g.WN / (wave * g.TM * g.TN * g.WMITER);
    if (g.BM % g.WM != 0)               return "BM % WM != 0";
    if (g.BN % g.WN != 0)               return "BN % WN != 0";
    if (g.WM % g.WMITER != 0)           return "WM % WMITER != 0";
    if ((g.WM / g.WMITER) % g.TM != 0)  return "(WM / WMITER) % TM != 0";
    if (wniter < 1)                       return "wniter < 1 (TM*TN*WMITER too large for WM*WN/wave)";
    if (g.WN % wniter != 0)               return "WN % wniter != 0";
    if ((g.WN / wniter) % g.TN != 0)      return "(WN / wniter) % TN != 0";
    if (g.BS != num_warps * wave)         return "BS != num_warps * wave (BS is derived, not free)";
    if (g.TM * g.TN > g.WM * g.WN / wave) return "TM * TN > WM * WN / wave";
    if (g.BS > 1024)                    return "BS > 1024";
    if (g.TM % 2 != 0)                  return "TM % 2 != 0";
    if (g.pad >= 0 && g.pad > 4)        return "PAD outside [0,4]";
    if (g.reshape >= 0 && g.reshape != 0 && g.reshape != 1) return "RESHAPE not in {0,1}";
    return nullptr;
}

// Dedup identity: two geometries that create the same effective tile must share a key.
// cm2 halves BN when `smaller` is set (mul_mm_cm2.comp:37-38), which is why
// {BN=256,smaller=1} and {BN=128,smaller=0} are the same candidate -- the same
// phenomenon that makes the driver report identical shared memory for cm2 l and m
// (t2-report.md section 5.1).
static inline uint64_t vk_tile_canonical_key(int path, const vk_tile_geometry & g) {
    uint32_t bn_eff = g.BN;
    if (path == VK_TILE_PATH_CM2 && g.smaller == 1 && (g.BN % 2) == 0) {
        bn_eff = g.BN / 2;
    }
    const uint32_t parts[10] = {
        (uint32_t) path, g.BM, bn_eff, g.BK, g.WM, g.WN, g.WMITER, g.TM, g.TN, g.BS,
    };
    // Standard FNV-1a 64 (offset basis 0xCBF29CE484222325, prime 0x100000001B3), fed one BYTE at
    // a time in little-endian order -- the same wire protocol the shell's rowSetHash() uses, so
    // both hashes are "FNV-1a" in the standard sense.  R0 changed the basis to the standard value
    // and the feed from one u32 per step to bytes; the function has no callers, is never
    // persisted and never compared across repos, so this only changes an in-process dedup key.
    // NOTE: `bn_eff` above exists for DEDUP only (two geometries that create the same tile must
    // share a key) -- it must NOT be reused as the pipeline align (t3-shell-plan.md 6.9: the
    // align rule is per path and is derived from the stock buckets, not from this identity).
    uint64_t h = 14695981039346656037ull;
    for (int i = 0; i < 10; i++) {
        const uint32_t v = parts[i];
        for (int b = 0; b < 4; b++) {
            h ^= (uint64_t) ((v >> (8 * b)) & 0xffu);
            h *= 1099511628211ull;
        }
    }
    return h;
}

// The align a row's ALIGNED variant must be created with, per path (t3-shell-plan.md 6.9).
// Derived from the stock buckets so that a row and the bucket it shadows agree:
//   cm2   : warptiles are {BLOCK_SIZE,BM,BN,BK,smaller} = {256,128,256,64,1} / {256,128,128,64,0}
//           / {128,64,64,64,0} and stock aligns l/m/s = 128/64/32 = BN/2 (ggml-vulkan.cpp:4806-4808);
//   scalar: warptiles have BM == BN and stock aligns 128/64/32 (ggml-vulkan.cpp:4875-4877),
//           which max(BM,BN) reproduces exactly.
// DIRECTION DISCIPLINE (the same one KATAGO_VK_MM_TILE=custom: states): too small means the
// shader reads across rows -- a silent wrong answer; too large only means fewer aligned hits --
// slower.  So there is nothing to measure when the two candidates disagree: take the larger.
// Taking the smaller needs shader-level evidence, never an A/B run.  An odd cm2 BN falls back to
// BN itself because BN/2 would truncate (i.e. round down = the dangerous direction).
static inline uint32_t vk_tile_row_align_for(int path, const vk_tile_geometry & g) {
    if (path == VK_TILE_PATH_CM2) {
        return (g.BN % 2 == 0) ? (g.BN / 2) : g.BN;
    }
    return g.BM > g.BN ? g.BM : g.BN;
}

// Per-path axis classification: the enumeration space, not validity.  Kept here so the
// enumerator and any validator read one table.  Sources: t1-predicate.md section 2
// (2026-09-13 rewrite), t1-candidates.md section 2.2, VENDOR-BASE.md section 2, and the
// measured BK verdict (census-t05-shapes.md section 4.4).
//
// Scalar: BM/BN/WM/WN/WMITER/TM/TN are ENUMERABLE; BS is DERIVED (num_warps*wave);
//   TK and WARP are PINNED (TK==1, WARP==vk_tile_wave_for(subgroup_size) -- fill them in from the
//   device, do not sweep: the wave is a hardware property, not a tunable);
//   BK is INERT on the f16 path (compile-time constant; spec id 3 exists only for the
//   quant branch), and SHMEM_STRIDE_PAD / APPLY_SLM_A_RESHAPE are INERT because on the
//   scalar path they are NOT specs at all (mul_mm.comp's COOPMAT guard and its explicit
//   "DO NOT spec-ize" note -- lesson #41: a multi-level spec array made the Blackwell
//   driver emit BM-dependent garbage).  That is a stronger statement than "measured
//   inert": the earlier experiment could not have shown anything else.
// cm1: as scalar, plus SHMEM_STRIDE_PAD / APPLY_SLM_A_RESHAPE are ENUMERABLE (they are
//   spec-ized there) but unmeasured -- we have no cm1 machine.
// cm2: BM/BN/BK/BS/smaller are ENUMERABLE (BLOCK_SIZE is a real cm2 spec, not derived).
static inline int vk_tile_axis_class(int path, int axis) {
    if (path == VK_TILE_PATH_CM2) {
        switch (axis) {
            case VK_TILE_AXIS_BS:
            case VK_TILE_AXIS_BM:
            case VK_TILE_AXIS_BN:
            case VK_TILE_AXIS_BK:
            case VK_TILE_AXIS_SMALLER:
                return VK_TILE_AXIS_ENUMERABLE;
            case VK_TILE_AXIS_TK:
            case VK_TILE_AXIS_WARP:
                // NOT_APPLICABLE, not INERT: these are not axes of the cm2 path at all
                // (there is no warptile semantics there), so the enumerator should not
                // even consider them.  INERT would mean "exists but sweeping does nothing".
                return VK_TILE_AXIS_NOT_APPLICABLE;
            default:
                return VK_TILE_AXIS_NOT_APPLICABLE;
        }
    }
    if (path != VK_TILE_PATH_SCALAR && path != VK_TILE_PATH_CM1) {
        return VK_TILE_AXIS_NOT_APPLICABLE;
    }
    switch (axis) {
        case VK_TILE_AXIS_BM:
        case VK_TILE_AXIS_BN:
        case VK_TILE_AXIS_WM:
        case VK_TILE_AXIS_WN:
        case VK_TILE_AXIS_WMITER:
        case VK_TILE_AXIS_TM:
        case VK_TILE_AXIS_TN:
            return VK_TILE_AXIS_ENUMERABLE;
        case VK_TILE_AXIS_BS:
            return VK_TILE_AXIS_DERIVED;
        case VK_TILE_AXIS_TK:
        case VK_TILE_AXIS_WARP:
            return VK_TILE_AXIS_PINNED;
        case VK_TILE_AXIS_SPEC_PAD:
        case VK_TILE_AXIS_SPEC_RESHAPE:
            return path == VK_TILE_PATH_CM1 ? VK_TILE_AXIS_ENUMERABLE : VK_TILE_AXIS_INERT;
        case VK_TILE_AXIS_BK:
            return VK_TILE_AXIS_INERT;   // f16 family: compile-time constant in the shader
        default:
            return VK_TILE_AXIS_NOT_APPLICABLE;
    }
}

// Convenience for the enumerator: may this axis be swept on this path?
static inline int vk_tile_axis_is_live(int path, int axis) {
    return vk_tile_axis_class(path, axis) == VK_TILE_AXIS_ENUMERABLE;
}
