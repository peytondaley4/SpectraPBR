#pragma once

//------------------------------------------------------------------------------
// Path-guide cell layout — single source of truth, shared by:
//   - optix_programs/path_guide_grid_device.h  (raygen-side access)
//   - src/rendering/path_guide_kernels.cu      (refit / subdivision / init)
//   - src/rendering/path_guide_grid.h          (host mirror + inspector)
//
// One cell = a K-lobe von Mises–Fisher MIXTURE over incident radiance
// (Ruppert et al. 2020 flavored, hard-assignment stepwise EM on device):
// a single lobe cannot represent multimodal incidence (two windows lighting
// one spot), and subdividing SPACE to compensate explodes the cell count in
// genuinely-wide-radiance regions. K narrow lobes capture the modes; spatial
// subdivision is reserved for actual spatial variation (sample-count
// criterion in the subdivision kernel).
//
// === Cell data layout (64 floats = 256 bytes per cell) ===
//
//  [0..23]  PG_NUM_LOBES=4 lobes x PG_LOBE_STRIDE=6 floats:
//             +0..2  mu (unit mean direction)     — written by refit kernel
//             +3     kappa (<= 0: lobe untrained) — written by refit kernel
//             +4     exp(-2*kappa) cache          — written by refit kernel
//             +5     weight pi_k (mixture weight, normalized) — refit kernel
//           The sampling hot data is the leading 96B segment.
//  [24..39] per-lobe interval sums {sx, sy, sz, sw} — atomicAdd by shaders
//           into the hard-assigned lobe, consumed/zeroed by the refit kernel
//  [40..55] per-lobe cumulative sums — EMA-decayed, owned by the refit kernel
//  [56]     lastHitFrame — atomicMax (positive floats sort like ints)
//  [57]     interval deposit count — atomicAdd 1 per deposit
//  [58]     cumulative deposit count (EMA) — owned by refit kernel; drives
//           the guide confidence ramp and the subdivision criterion
//  [59..63] reserved
//------------------------------------------------------------------------------

#define PG_NUM_LOBES         4
#define PG_LOBE_STRIDE       6
#define PG_L_MU_X            0   // lobe-relative offsets (base = k * PG_LOBE_STRIDE)
#define PG_L_MU_Y            1
#define PG_L_MU_Z            2
#define PG_L_KAPPA           3
#define PG_L_EXP_NEG2K       4
#define PG_L_WEIGHT          5

#define PG_INT_BASE          24  // + k*4: {sumX, sumY, sumZ, sumW}
#define PG_CUMS_BASE         40  // + k*4: {sumX, sumY, sumZ, sumW}
#define PG_LAST_HIT_FRAME    56
#define PG_INT_COUNT         57
#define PG_CUM_COUNT         58
#define PG_ENTRY_STRIDE      64

#if defined(__CUDACC__) || defined(__CUDA_ARCH__)
// Initialize a cell's lobes to the tetrahedral starting configuration:
// distinct directions so the hard assignment partitions the sphere from the
// first deposit (all-zero lobes would collapse every deposit onto lobe 0),
// kappa = 0 so the cell is never sampled before its first refit. Called by
// the insert winner (raygen), and by the clear/reinit kernel. Writes only
// the lobe-parameter fields — disjoint from the interval sums other threads
// may concurrently atomicAdd into.
__forceinline__ __device__ void pgInitCellLobes(float* cell)
{
    const float t = 0.57735027f;  // 1/sqrt(3)
    const float dirs[PG_NUM_LOBES][3] = {
        {  t,  t,  t }, {  t, -t, -t }, { -t,  t, -t }, { -t, -t,  t }
    };
    for (int k = 0; k < PG_NUM_LOBES; k++) {
        float* l = cell + k * PG_LOBE_STRIDE;
        l[PG_L_MU_X] = dirs[k][0];
        l[PG_L_MU_Y] = dirs[k][1];
        l[PG_L_MU_Z] = dirs[k][2];
        l[PG_L_KAPPA] = 0.0f;
        l[PG_L_EXP_NEG2K] = 1.0f;   // exp(-2*0)
        l[PG_L_WEIGHT] = 1.0f / PG_NUM_LOBES;
    }
}
#endif
