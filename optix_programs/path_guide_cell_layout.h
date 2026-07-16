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
// === Cell data layout (98 floats = 392 bytes per cell) ===
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
//  === Deposit-position statistics ([71..78], [80..81]) — jittered cell ===
//  mw-weighted (mw = log1p(w)) first and second spatial moments of the
//  deposits, rel in [-1,1] from the cell center. These do NOT drive
//  subdivision (see below for why every weighted-centroid criterion failed);
//  they serve the GUIDE itself:
//  [71..73] interval Sum(mw*rel_a) — atomicAdd per deposit
//  [74..76] cumulative EMA — with MW ([80..81]) gives the deposit centroid
//           c_w = S_a/MW, used as the PARALLAX PIVOT: lobes reproject around
//           where the radiance was actually deposited, not the cell center,
//           so coarse cells aim correctly (Ruppert 2020 spirit).
//  [77..78] Sum(mw*|rel|^2) interval/EMA — with the centroid gives the
//           deposit SPREAD, which bounds the lobe's positional uncertainty
//           and therefore the kappa cap (a compact pool in a big cell may be
//           sharp; a wall-to-wall glow may not).
//
//  === Why subdivision does NOT use weighted-centroid contrast ===
//  Three structural reasons, established numerically (2026-07): (1) deposits
//  lie on 2D surfaces inside 3D cells, so the raw weighted centroid carries
//  a pure-geometry offset up to ~0.5 with uniform radiance (false positive in
//  ~1/3 of cells at every level); (2) subtracting the unweighted centroid
//  cancels geometry but also the signal — deposit DENSITY is radiance-
//  correlated, so edges live in the density channel the subtraction removes;
//  (3) any statistic on w = Li/pdf is self-erasing: the guide's convergence
//  flattens Li/pdf by design. No threshold separated real edges from noise
//  for any centroid variant (inverted 5-10x). Subdivision instead uses the
//  half-cell log-RADIANCE statistics below ([82..97]).
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
#define PG_INT_SRR           77   // interval  Sum(mw*|rel|^2) (atomicAdd) —
#define PG_CUM_SRR           78   // cumulative EMA (refit kernel). With the
                                  // centroid this gives the deposit SPREAD
                                  // about it: spread^2 = SRR/SMW - |SR/SMW|^2.
                                  // Drives the geometry-aware kappa cap — the
                                  // lobe's positional uncertainty is the
                                  // spread about its parallax pivot, NOT the
                                  // whole cell size, so a compact light pool
                                  // in a coarse cell still earns a sharp lobe.
#define PG_MATURITY          79   // slow-decayed deposit count (refit kernel)
#define PG_INT_SMW           80   // interval  Sum(mw) (atomicAdd by shaders)
#define PG_CUM_SMW           81   // cumulative EMA Sum(mw) (refit kernel) —
                                  // the centroid/spread normalizer; MUST use
                                  // the same weighting as SR and SRR
#define PG_INT_VISITS        82   // interval  guided-vertex VISIT count —
#define PG_CUM_VISITS        83   // cumulative EMA. Incremented for every
                                  // guided vertex of a TRAINING-SUBSAMPLED
                                  // path (PG_TRAIN_PROB = 1/2, uncompensated;
                                  // gated on path_guide_training; first
                                  // MAX_TRAIN_VERTICES vertices only), with
                                  // NO radiance gate: the refinement budget
                                  // must follow traffic (Mueller-style), not
                                  // brightness. Absolute visit thresholds
                                  // are denominated in these subsampled
                                  // fast-EMA units (sustained rates, not
                                  // cumulative counts). Written to the EXACT
                                  // (unjittered) cell.
// Half-cell split statistics (written to the EXACT cell at the EXACT vertex
// position — the box-filter jitter would smear the very edge being detected):
// per axis a, the positive half-cell's visit count and its Sum(log1p(Llum)),
// where Llum is the vertex's reconstructed incident radiance luminance (the
// RAW radiance, NOT Li/pdf — importance sampling flattens Li/pdf by design,
// so any pdf-divided statistic self-erases as the guide converges). Negative
// halves derive by subtraction from the totals. The split criterion is the
// per-axis log-ratio of conditional mean log-radiance between the halves:
// density-invariant (conditional means), geometry-invariant (both halves are
// measured, not a centroid), and IS-invariant (no pdf anywhere).
#define PG_INT_HC_X          84   // interval  visits with relX > 0
#define PG_INT_HC_Y          85
#define PG_INT_HC_Z          86
#define PG_CUM_HC_X          87   // cumulative EMA (refit kernel)
#define PG_CUM_HC_Y          88
#define PG_CUM_HC_Z          89
#define PG_INT_HL_X          90   // interval  Sum(log1p(Llum)) over relX > 0
#define PG_INT_HL_Y          91
#define PG_INT_HL_Z          92
#define PG_CUM_HL_X          93   // cumulative EMA (refit kernel)
#define PG_CUM_HL_Y          94
#define PG_CUM_HL_Z          95
#define PG_INT_SL            96   // interval  Sum(log1p(Llum)) over ALL visits
#define PG_CUM_SL            97   // cumulative EMA (refit kernel)
#define PG_ENTRY_STRIDE      98

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
