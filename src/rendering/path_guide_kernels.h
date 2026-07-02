#pragma once

//------------------------------------------------------------------------------
// Device-side path-guide maintenance kernels (launchers).
//
// The cell table is device-resident (path_guide_hash_device.h): shaders
// allocate cells on first touch, so there is no structure build, readback,
// or grid swap to maintain. What remains host-driven is launching:
//   - the refit kernel (fold interval sums into EMA cumulative sums, refit
//     each cell's vMF lobe in place), every few frames, and
//   - the subdivision kernel (insert the 8 children of any mature cell whose
//     radiance is spatially off-center — count gate x noise-floored contrast
//     test), every refine interval.
// Both are bounded by the device-side allocation counter — the host never
// needs to know the cell count to launch them.
//------------------------------------------------------------------------------

#include <cuda_runtime.h>
#include <cstdint>

namespace spectra {

// Fold per-lobe interval sums into EMA cumulative sums and refit the cell's
// vMF mixture (hard-assignment stepwise EM M-step; Banerjee/Sra kappa
// approximation; dead-lobe re-seeding). kappa is capped geometry-aware:
// a lobe fit over a cell of edge `cellSize` (derived from cellKeys level +
// baseCellSize, halving per level) and consumed up to ~1 cell away by the
// box-filter jitter must not be narrower than the borrowing error
// cellSize/meanDist — so kappa <= (2*meanDist/cellSize)^2, with the flat
// 2000 ceiling unlocked only past the slow-decayed maturity count. Must run
// on the render stream so it is ordered against optixLaunch (shaders and the
// refit never touch the cell data concurrently).
void launchRefitCells(float* data, const uint64_t* cellKeys,
                      const uint32_t* cellCounter, uint32_t cellCapacity,
                      float emaDecay, float baseCellSize, uint32_t currentFrame,
                      cudaStream_t stream);

// Subdivide cells straddling a spatial barrier: for every cell with level <
// maxLevel that has >= minCount deposits and whose radiance centroid is
// off-center (|centroid|^2 = sum_a S_a^2 / W^2 above BOTH contrastThreshold and
// a noise floor ~4/nEff, nEff = W^2/Sum(w^2) — so heavy-tailed Li/pdf weight
// spikes cannot fake spatial structure), insert its 8 children into the table.
// Cells far past maturity (8x minCount) split regardless of contrast: a
// first-moment centroid is blind to even-symmetric variation, and the hatch
// costs at most one surplus level on genuinely uniform hot cells. The
// scale-invariant centroid test refines only the boundary of a difference
// (caustic edge, shadow line), not uniform regions.
// Children warm-start with the parent's mixture and 1/8 of its cumulative
// statistics so guiding (and the confidence ramp) survive the split.
// Idempotent: existing children are left untouched, so re-running on an
// already-subdivided parent only costs hash probes. Runs on the render
// stream for the same ordering reason as the refit.
//
// countSnapshot must hold a copy of *cellCounter taken (stream-ordered)
// BEFORE the launch: bounding by the live counter would let late-scheduled
// blocks process children inserted earlier in the same pass — a warm-started
// child inherits parent/8 evidence and could cascade-subdivide within one
// pass, and its payload may still be mid-write.
void launchSubdivideCells(uint64_t* hashKeys, uint32_t* hashValues,
                          uint32_t hashTableSize, uint32_t hashShift,
                          uint64_t* cellKeys, uint32_t* cellCounter, uint32_t cellCapacity,
                          const uint32_t* countSnapshot,
                          float* data, uint32_t entryStride,
                          uint32_t maxLevel, float minCount, float contrastThreshold,
                          uint32_t currentFrame,
                          cudaStream_t stream);

// Reinitialize every allocated cell's lobes to the tetrahedral starting
// mixture. Required after clear(): an all-zero mixture would collapse the
// deposit hard-assignment onto lobe 0. Stream-ordered after the clearing
// memset.
void launchInitCells(float* data,
                     const uint32_t* cellCounter, uint32_t cellCapacity,
                     cudaStream_t stream);

} // namespace spectra
