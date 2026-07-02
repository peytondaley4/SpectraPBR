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
// === Cell data layout (84 floats = 336 bytes per cell) ===
//
//  [0..27]  PG_NUM_LOBES=4 lobes x PG_LOBE_STRIDE=7 floats:
//             +0..2  mu (unit mean direction)     — written by refit kernel
//             +3     kappa (<= 0: lobe untrained) — written by refit kernel
//             +4     exp(-2*kappa) cache          — written by refit kernel
//             +5     weight pi_k (mixture weight, normalized) — refit kernel
//             +6     meanDist — mean distance to the incident radiance along
//                    mu, used for PARALLAX-AWARE reprojection at lookup
//                    (Ruppert et al. 2020). 0 = untrained / treat as distant.
//           The sampling hot data is the leading 112B segment.
//  [28..47] per-lobe interval sums {sx, sy, sz, sw, sDist} (PG_SUM_STRIDE=5) —
//           atomicAdd by shaders into the hard-assigned lobe, consumed/zeroed
//           by the refit kernel. sDist = sum of weight*hitDistance.
//  [48..67] per-lobe cumulative sums {sx,sy,sz,sw,sDist} — EMA-decayed, owned
//           by the refit kernel
//  [68]     lastHitFrame — atomicMax (positive floats sort like ints)
//  [69]     interval deposit count — atomicAdd 1 per deposit
//  [70]     cumulative deposit count (EMA) — owned by refit kernel; drives
//           the guide confidence ramp and the min-count gate of subdivision
//  === Spatial-contrast statistics ([71..78], [80..81]) ===
//  All weighted by mw = log1p(w), NOT the raw deposit weight w: raw Li/pdf
//  weights are heavy-tailed BY NATURE in exactly the regions subdivision
//  matters most (caustics, near-field pools — the tail IS the signal there,
//  not firefly noise), which collapses the effective sample size and lets the
//  noise floor block refinement forever. log1p tames the tails ~1000x -> ~7x
//  while preserving the bright-vs-dark asymmetry the centroid detects, so
//  caustic edges split and single fireflies still cannot fake structure.
//
//  [71..73] interval spatial first moments {Sum(mw*relX), Sum(mw*relY),
//           Sum(mw*relZ)} — atomicAdd per deposit, rel in [-1,1] = deposit
//           position within the cell from its center. Folded to the EMA below.
//  [74..76] cumulative spatial first moments (EMA) — owned by refit kernel.
//           With the log-weight sum MW ([80..81]) these give the weighted
//           radiance CENTROID c_a = S_a / MW. The subdivision kernel splits a
//           cell when |centroid|^2 = sum_a S_a^2/MW^2 exceeds a threshold:
//           i.e. only where the radiance is spatially OFF-CENTER (a barrier /
//           edge inside the cell), independent of absolute brightness. A
//           uniform cell (bright OR dark) has centroid ~0 and is never split.
//  [77]     interval Sum(mw^2) — atomicAdd per deposit
//  [78]     cumulative Sum(mw^2) (EMA) — refit kernel. With MW this gives the
//           Kish EFFECTIVE sample size nEff = MW^2 / Sum(mw^2), bounding the
//           centroid's estimation noise; the subdivision kernel requires the
//           contrast to also clear a noise floor ~ 1/nEff.
//  [79]     maturity — SLOW-decayed deposit count (decay 0.98/refit vs 0.85
//           for cumCount), owned by the refit kernel. Drives the guide
//           confidence ramp and the kappa evidence gate INSTEAD of the fast
//           EMA count: cumCount measures recent deposit RATE, which starves
//           exactly the cells that need guiding most — in dim regions lit by
//           a small/hard light, deposits only occur when a path FINDS light,
//           which is rare under BSDF sampling, so rate-based confidence never
//           rises, guiding never activates, and successes stay rare (a
//           feedback deadlock; the visible symptom is one resolved region
//           around the light while everywhere else stays noisy). Maturity
//           accumulates evidence over a ~50-refit window, so consistent
//           trickle deposits eventually activate guiding and break the loop.
//------------------------------------------------------------------------------

#define PG_NUM_LOBES         4
#define PG_LOBE_STRIDE       7
#define PG_L_MU_X            0   // lobe-relative offsets (base = k * PG_LOBE_STRIDE)
#define PG_L_MU_Y            1
#define PG_L_MU_Z            2
#define PG_L_KAPPA           3
#define PG_L_EXP_NEG2K       4
#define PG_L_WEIGHT          5
#define PG_L_MEAN_DIST       6

#define PG_SUM_STRIDE        5   // per-lobe sum group: {sx, sy, sz, sw, sDist}
#define PG_S_DIST            4   // distance sub-offset within a sum group
#define PG_INT_BASE          28  // + k*PG_SUM_STRIDE
#define PG_CUMS_BASE         48  // + k*PG_SUM_STRIDE
#define PG_LAST_HIT_FRAME    68
#define PG_INT_COUNT         69
#define PG_CUM_COUNT         70
#define PG_INT_SR_X          71   // interval  Sum(mw*relX), mw=log1p(w) (atomicAdd)
#define PG_INT_SR_Y          72   // interval  Sum(mw*relY)
#define PG_INT_SR_Z          73   // interval  Sum(mw*relZ)
#define PG_CUM_SR_X          74   // cumulative EMA Sum(mw*relX) (refit kernel)
#define PG_CUM_SR_Y          75   // cumulative EMA Sum(mw*relY)
#define PG_CUM_SR_Z          76   // cumulative EMA Sum(mw*relZ)
#define PG_INT_SW2           77   // interval  Sum(mw^2), mw=log1p(w) (atomicAdd)
#define PG_CUM_SW2           78   // cumulative EMA Sum(mw^2) (refit kernel)
#define PG_MATURITY          79   // slow-decayed deposit count (refit kernel)
#define PG_INT_SMW           80   // interval  Sum(mw) (atomicAdd by shaders)
#define PG_CUM_SMW           81   // cumulative EMA Sum(mw) (refit kernel) —
                                  // the centroid/nEff normalizer; MUST use the
                                  // same weighting as the moments and SW2
#define PG_ENTRY_STRIDE      84   // [82..83] reserved

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
        l[PG_L_MEAN_DIST] = 0.0f;   // untrained: no parallax reprojection
    }
}
#endif
