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
// Deliberately small API:
//   1. vk_tile_candidate_reason() -- is this concrete geometry legal, and if not, why
//   2. vk_tile_canonical_key()    -- dedup identity (dedup is NOT validity)
//   3. vk_tile_axis_is_live()     -- per-path live-axis table (the enumeration space)
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

// Axis ids for vk_tile_axis_is_live().  These are the warptile/spec slots as the create
// path sees them (ggml_vk_mul_mm_spec order), not a tuner invention:
//   warptile = { BS, BM, BN, BK, WM, WN, WMITER, TM, TN, WARP, PAD }
//   spec ids 12/13 = PAD / RESHAPE (shader spec constants)
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
    VK_TILE_AXIS_WARP    = 9,
    VK_TILE_AXIS_PAD     = 10,
    VK_TILE_AXIS_SPEC_PAD     = 12,
    VK_TILE_AXIS_SPEC_RESHAPE = 13,
};

// One concrete tile geometry.  Scalar / cm1 use the warptile fields; cm2 uses
// BM, BN, BK, BS (BLOCK_SIZE) and `smaller`.  Unset spec constants are -1.
struct vk_tile_geometry {
    uint32_t BM, BN, BK, WM, WN, WMITER, TM, TN, BS;
    int pad;      // -1 = unspecified (shader default applies)
    int reshape;  // -1 = unspecified
    int smaller;  // -1 = unspecified (cm2 only)
};

// Returns nullptr when the geometry is legal, otherwise a static reason string naming
// the first violated clause.  Callers log it as-is, which is what makes the enumerator
// and the loader produce the SAME reason for the same bad input.
static inline const char * vk_tile_candidate_reason(int path, const vk_tile_geometry & g) {
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
    const uint32_t wniter = g.WM * g.WN / (32 * g.TM * g.TN * g.WMITER);
    if (g.BM % g.WM != 0)               return "BM % WM != 0";
    if (g.BN % g.WN != 0)               return "BN % WN != 0";
    if (g.WM % g.WMITER != 0)           return "WM % WMITER != 0";
    if ((g.WM / g.WMITER) % g.TM != 0)  return "(WM / WMITER) % TM != 0";
    if (wniter < 1)                     return "wniter < 1 (TM*TN*WMITER too large for WM*WN/32)";
    if (g.WN % wniter != 0)             return "WN % wniter != 0";
    if ((g.WN / wniter) % g.TN != 0)    return "(WN / wniter) % TN != 0";
    if (g.BS != num_warps * 32)         return "BS != num_warps * 32 (BS is derived, not free)";
    if (g.TM * g.TN > g.WM * g.WN / 32) return "TM * TN > WM * WN / 32";
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
    uint64_t h = 1469598103934665603ull;   // FNV-1a: stable across compilers/platforms
    for (int i = 0; i < 10; i++) {
        h ^= parts[i];
        h *= 1099511628211ull;
    }
    return h;
}

// Per-path live axes: the enumeration space, not validity.  Kept here so the enumerator
// and any validator read one table.  Sources: t1-predicate.md (live/dead axis list, BS
// derived), t1-candidates.md section 2.2 (cm2's {BS,BM,BN,BK,smaller}), and the measured
// BK verdict (t1-candidates.md section 4 / census-t05-shapes.md section 4.4).
static inline int vk_tile_axis_is_live(int path, int axis) {
    if (path == VK_TILE_PATH_CM2) {
        switch (axis) {
            case VK_TILE_AXIS_BS:   // BLOCK_SIZE is a real cm2 spec field
            case VK_TILE_AXIS_BM:
            case VK_TILE_AXIS_BN:
            case VK_TILE_AXIS_BK:
                return 1;
            default:
                return 0;
        }
    }
    switch (axis) {
        case VK_TILE_AXIS_BM:
        case VK_TILE_AXIS_BN:
        case VK_TILE_AXIS_WM:
        case VK_TILE_AXIS_WN:
        case VK_TILE_AXIS_WMITER:
        case VK_TILE_AXIS_TM:
        case VK_TILE_AXIS_TN:
            return 1;
        // BS is derived (BS == num_warps*32); BK is a compile-time constant in the
        // shader (mul_mm.comp defines BK 32); WARP/PAD/RESHAPE were measured inert.
        case VK_TILE_AXIS_BS:
        case VK_TILE_AXIS_BK:
        case VK_TILE_AXIS_WARP:
        case VK_TILE_AXIS_PAD:
        case VK_TILE_AXIS_SPEC_PAD:
        case VK_TILE_AXIS_SPEC_RESHAPE:
        default:
            return 0;
    }
}
