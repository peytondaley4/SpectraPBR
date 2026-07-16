#pragma once

//------------------------------------------------------------------------------
// Device-side path-guide maintenance kernels (launchers).
//
// The cell table is device-resident (path_guide_hash_device.h): shaders
// allocate cells on first touch, so there is no structure build, readback,
// or grid swap to maintain. What remains host-driven is launching:
//   - the refit kernel (fold interval sums into EMA cumulative sums, refit
//     each cell's vMF lobe in place), every few frames, and
//   - the subdivision kernel (insert the 8 children of any cell passing the
//     level-normalized visit gate AND the half-cell radiance structure
//     test — see launchSubdivideCells), every refine interval.
// Both are bounded by the device-side allocation counter — the host never
// needs to know the cell count to launch them.
//------------------------------------------------------------------------------

#include <cuda_runtime.h>
#include <cstdint>

namespace spectra {

// Subdivision statistics buffer layout (uint32 counters, zeroed per pass):
//   [0] parents split (passed both the visit gate and the structure test)
//   [1] parents that passed the level-normalized visit gate
//   [2] gate-passed parents whose half-cell radiance structure test failed
//   [3] children actually inserted (new cells)
//   [4..19] live cells per level 0..15 (histogram over the snapshot)
constexpr uint32_t PG_SUBDIV_STAT_SPLIT      = 0;
constexpr uint32_t PG_SUBDIV_STAT_ELIGIBLE   = 1;
constexpr uint32_t PG_SUBDIV_STAT_NOSTRUCT   = 2;
constexpr uint32_t PG_SUBDIV_STAT_CHILDREN   = 3;
constexpr uint32_t PG_SUBDIV_STAT_LEVEL0     = 4;
constexpr uint32_t PG_SUBDIV_STATS_SIZE      = 20;

// Fold per-lobe interval sums into EMA cumulative sums and refit the cell's
// vMF mixture (hard-assignment stepwise EM M-step; Banerjee/Sra kappa
// approximation; dead-lobe re-seeding). kappa is capped geometry-aware,
// denominated in the measured DEPOSIT SPREAD about the parallax pivot (the
// deposit centroid), not the cell size: sigmaPos = spreadRel * halfCell,
// kappa <= (meanDist/sigmaPos)^2, floored at 8 — a compact light pool in a
// coarse cell still earns a sharp, correctly-aimed lobe. The flat 2000
// ceiling additionally requires slow-decayed maturity evidence AND the
// cell's own (post-split) log-tamed weight mass. Must run on the render
// stream so it is ordered against optixLaunch (shaders and the refit never
// touch the cell data concurrently).
void launchRefitCells(float* data, const uint64_t* cellKeys,
                      const uint32_t* cellCounter, uint32_t cellCapacity,
                      float emaDecay, float baseCellSize, uint32_t currentFrame,
                      cudaStream_t stream);

// Subdivide cells with (1) enough guided-vertex VISITS — traffic counted
// with no radiance gate, threshold minVisits at startLevel and halved per
// axis per level below it (floored at min(256, minVisits)) — and (2) either
// genuine radiance STRUCTURE (per-axis half-cell conditional mean
// log1p(radiance) ratio/difference at exact positions exceeding
// hlrThreshold, per-half minimum visit floor) or a RESOLUTION-LIMITED guide
// (a well-evidenced lobe's fitted kappa demand exceeds the spread-based cap
// severalfold while the cap is still low — smooth near-field illumination
// needs finer cells even without a radiance edge). The visit
// gate makes ELIGIBILITY brightness-neutral (the old radiance-gated deposit
// count only ever admitted bright cells — the root cause of brightness-
// correlated refinement); the structure test makes the TRIGGER density-,
// geometry-, and importance-sampling-invariant (every weighted-centroid
// contrast variant failed on at least one of those — see the layout
// header). There is no escape hatch and no lineage lock: the structure test
// needs neither.
// Children warm-start with the parent's full mixture, 1/8 of its per-lobe
// evidence, and its FULL maturity (the mixture is verbatim, so confidence
// must not drop); parent-frame spatial/visit statistics reset to zero.
// Idempotent: existing children are left untouched, so re-running on an
// already-subdivided parent only costs hash probes. Runs on the render
// stream for the same ordering reason as the refit.
//
// countSnapshot must hold a copy of *cellCounter taken (stream-ordered)
// BEFORE the launch: bounding by the live counter would let late-scheduled
// blocks process children inserted earlier in the same pass — a warm-started
// child inherits parent/8 evidence and could cascade-subdivide within one
// pass, and its payload may still be mid-write.
// stats (optional, may be null): PG_SUBDIV_STATS_SIZE uint32 counters,
// zeroed by the caller before the pass (layout above).
void launchSubdivideCells(uint64_t* hashKeys, uint32_t* hashValues,
                          uint32_t hashTableSize, uint32_t hashShift,
                          uint64_t* cellKeys, uint32_t* cellCounter, uint32_t cellCapacity,
                          const uint32_t* countSnapshot,
                          float* data, uint32_t entryStride,
                          uint32_t maxLevel, uint32_t startLevel,
                          float minVisits, float hlrThreshold,
                          float baseCellSize,
                          uint32_t currentFrame, uint32_t* stats,
                          cudaStream_t stream);

// Reinitialize every allocated cell's lobes to the tetrahedral starting
// mixture. Required after clear(): an all-zero mixture would collapse the
// deposit hard-assignment onto lobe 0. Stream-ordered after the clearing
// memset.
void launchInitCells(float* data,
                     const uint32_t* cellCounter, uint32_t cellCapacity,
                     cudaStream_t stream);

} // namespace spectra
