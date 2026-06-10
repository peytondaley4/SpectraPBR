#pragma once

//------------------------------------------------------------------------------
// Device-side path-guide maintenance kernels (launchers).
//
// These replace the previous CPU build pipeline's per-build full readback /
// vMF fit / full upload (10–300 ms CPU plus a training-loss window between
// snapshot and swap). The refit kernel runs in-place on the render stream in
// well under a millisecond, so lobes can be refit every few frames and no
// training deposit is ever lost.
//------------------------------------------------------------------------------

#include <cuda_runtime.h>
#include <cstdint>

namespace spectra {

// Gather-map sentinel/flags (see launchGatherCells)
constexpr uint32_t PG_GATHER_NEW_CELL   = 0xFFFFFFFFu;  // zero-init, lastHitFrame = current
constexpr uint32_t PG_GATHER_LOBE_ONLY  = 0x80000000u;  // copy lobe from (map & 0x7FFFFFFF), zero stats
constexpr uint32_t PG_GATHER_INDEX_MASK = 0x7FFFFFFFu;

// Fold interval sums into EMA cumulative sums, refit each cell's vMF lobe
// (Banerjee/Sra kappa approximation), zero the interval sums, and initialize
// lastHitFrame for cells that have never been hit. Must run on the render
// stream so it is ordered against optixLaunch (shaders and the refit never
// touch the cell data concurrently).
void launchRefitCells(float* data, uint32_t totalCells,
                      float emaDecay, uint32_t currentFrame,
                      cudaStream_t stream);

// Re-layout cell data after a structure change (new cells / refinement):
// dst[i] = src[map[i]] for surviving cells; PG_GATHER_NEW_CELL entries are
// zero-initialized; PG_GATHER_LOBE_ONLY entries copy mu/kappa from the parent
// (warm start for subdivided children) with fresh statistics. Runs on the
// render stream at swap time, so it picks up every deposit made since the
// staging snapshot — the old pipeline silently dropped those.
void launchGatherCells(float* dst, const float* src, const uint32_t* map,
                       uint32_t totalNewCells, uint32_t currentFrame,
                       cudaStream_t stream);

} // namespace spectra
