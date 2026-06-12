#pragma once

//------------------------------------------------------------------------------
// Sparse path guide grid — device-side (collision-free hash, device-resident
// allocation). Layout must match src/rendering/path_guide_grid.h. Used from
// raygen; the allocation/maintenance kernels live in path_guide_kernels.cu.
//
// Cells are allocated ON DEVICE the first time a path vertex needs one
// (pgTableInsert in path_guide_hash_device.h): no staging buffer, no host
// readback, no CPU merge, no grid swap. Cell indices are stable for the
// lifetime of the grid, and a fresh cell (zeroed payload, kappa = 0) is never
// sampled before its first refit, so sampling PDFs stay consistent within a
// launch.
//
// Lookup is TOP-DOWN: probe the base (start) level, then descend while a
// finer cell containing the position exists. Cells only exist at the base
// level or as complete 8-child sets created by the subdivision kernel, and
// per_level_scale is fixed at 2 (resolutions double per level), so the cell
// containing a position at level L+1 is always a child of the one at L. The
// common unsubdivided case costs one hash hit plus one miss (~2 probes).
//
// Path guiding: a 4-lobe von Mises–Fisher MIXTURE per cell for importance
// sampling of incident radiance (layout in path_guide_cell_layout.h).
// References:
//   - Müller et al., "Practical Path Guiding for Efficient Light-Transport
//     Simulation", EGSR 2017 (Computer Graphics Forum).
//   - Ruppert, Herholz, Lensch, "Robust Fitting of Parallax-Aware Mixtures
//     for Path Guiding", SIGGRAPH 2020 (vMF mixtures, stepwise EM).
//   - Von Mises–Fisher distribution; sampling via Wood/Ulrich.
//------------------------------------------------------------------------------

#include <cuda_runtime.h>
#include "vmf_device.h"
#include "path_guide_hash_device.h"
#include "path_guide_cell_layout.h"


// Device descriptor: the cell table plus world-space bounds. Level
// resolutions are read from __constant__ params (sparseResolutionAtLevel) to
// avoid copying a 16-element array into per-thread local storage.
struct SparsePathGuideDescriptorDevice {
    PathGuideTableDevice table;
    float bounds_min[3];
    float bounds_max[3];
};

__forceinline__ __device__ float sparseResolutionAtLevel(
    const SparsePathGuideDescriptorDevice& grid,
    unsigned int level)
{
    return (float)params.path_guide_level_resolutions[level < 16 ? level : 15];
}

__forceinline__ __device__ void worldToNormalized(
    const SparsePathGuideDescriptorDevice& grid,
    float px, float py, float pz,
    float& nx, float& ny, float& nz)
{
    float inv_dx = 1.0f / (grid.bounds_max[0] - grid.bounds_min[0]);
    float inv_dy = 1.0f / (grid.bounds_max[1] - grid.bounds_min[1]);
    float inv_dz = 1.0f / (grid.bounds_max[2] - grid.bounds_min[2]);
    nx = (px - grid.bounds_min[0]) * inv_dx;
    ny = (py - grid.bounds_min[1]) * inv_dy;
    nz = (pz - grid.bounds_min[2]) * inv_dz;
}

__forceinline__ __device__ void normalizedToCell(
    float nx, float ny, float nz,
    unsigned int level,
    const SparsePathGuideDescriptorDevice& grid,
    int& ix, int& iy, int& iz)
{
    float res = sparseResolutionAtLevel(grid, level);
    int resU = (int)floorf(res);
    if (resU < 1) resU = 1;
    nx = fminf(fmaxf(nx, 0.0f), 0.9999999f);
    ny = fminf(fmaxf(ny, 0.0f), 0.9999999f);
    nz = fminf(fmaxf(nz, 0.0f), 0.9999999f);
    ix = (int)floorf(nx * (float)resU);
    iy = (int)floorf(ny * (float)resU);
    iz = (int)floorf(nz * (float)resU);
    if (ix >= resU) ix = resU - 1;
    if (iy >= resU) iy = resU - 1;
    if (iz >= resU) iz = resU - 1;
}

// Pointer to cell data or nullptr if not found
__forceinline__ __device__ float* sparseCellDataPtr(
    const SparsePathGuideDescriptorDevice& grid,
    unsigned int global_index)
{
    if (grid.table.data == nullptr || global_index == PG_INVALID_CELL) return nullptr;
    return grid.table.data + (unsigned long long)global_index * grid.table.entry_stride;
}

// Top-down lookup: probe the base level, descend while a finer cell exists.
// Returns the deepest cell index (level via *outLevel), or PG_INVALID_CELL
// when the position is outside the grid bounds or the base cell does not
// exist yet (callers may then insert it via pathGuideInsertBaseCell).
__forceinline__ __device__ unsigned int topDownCellLookup(
    const SparsePathGuideDescriptorDevice& grid,
    float px, float py, float pz,
    unsigned int startLevel, unsigned int maxLevel,
    unsigned int* outLevel)
{
    *outLevel = startLevel;

    float nx, ny, nz;
    worldToNormalized(grid, px, py, pz, nx, ny, nz);
    if (nx < 0.0f || nx > 1.0f || ny < 0.0f || ny > 1.0f || nz < 0.0f || nz > 1.0f)
        return PG_INVALID_CELL;

    int ix, iy, iz;
    normalizedToCell(nx, ny, nz, startLevel, grid, ix, iy, iz);
    unsigned int idx = pgTableLookup(grid.table, startLevel,
        pgMortonEncode64((unsigned int)ix, (unsigned int)iy, (unsigned int)iz));
    if (idx == PG_INVALID_CELL) return PG_INVALID_CELL;

    unsigned int level = startLevel;
    for (unsigned int lev = startLevel + 1; lev <= maxLevel && lev < 16u; ++lev) {
        normalizedToCell(nx, ny, nz, lev, grid, ix, iy, iz);
        unsigned int childIdx = pgTableLookup(grid.table, lev,
            pgMortonEncode64((unsigned int)ix, (unsigned int)iy, (unsigned int)iz));
        if (childIdx == PG_INVALID_CELL) break;
        idx = childIdx;
        level = lev;
    }
    *outLevel = level;
    return idx;
}

// First-touch allocation of the base-level cell containing a position.
// Bounds-checked (out-of-grid positions never allocate). Returns the cell
// index, or PG_INVALID_CELL while another thread's insert is pending or the
// table is full. pgTableInsert initializes the lobes (tetrahedral, kappa 0)
// before publishing; the fresh cell is a valid training target immediately
// but cannot be sampled until its first refit fits a lobe.
__forceinline__ __device__ unsigned int pathGuideInsertBaseCell(
    const SparsePathGuideDescriptorDevice& grid,
    float px, float py, float pz,
    unsigned int startLevel)
{
    float nx, ny, nz;
    worldToNormalized(grid, px, py, pz, nx, ny, nz);
    if (nx < 0.0f || nx > 1.0f || ny < 0.0f || ny > 1.0f || nz < 0.0f || nz > 1.0f)
        return PG_INVALID_CELL;

    int ix, iy, iz;
    normalizedToCell(nx, ny, nz, startLevel, grid, ix, iy, iz);
    bool inserted = false;
    return pgTableInsert(grid.table, startLevel,
        pgMortonEncode64((unsigned int)ix, (unsigned int)iy, (unsigned int)iz),
        &inserted);
}

// Train a cell: hard-assign the deposit to the most responsible lobe
// (argmax_k pi_k * vmf_k(d), kappa floored so freshly initialized lobes
// partition the sphere by direction) and accumulate the importance-weighted
// direction into that lobe's interval sums. The refit kernel periodically
// folds these into the cumulative sums and refits each lobe (stepwise EM
// with hard assignment) — shaders never write the fields the sampling path
// reads, so sampling PDFs stay consistent within a launch.
__forceinline__ __device__ void pathGuideTrainCell(
    float* cell,
    float dx, float dy, float dz, float weight,
    unsigned int frameIndex)
{
    // Reject non-finite or non-positive weights — atomicAdd of Inf/NaN
    // permanently corrupts cell sums with no way to recover.
    if (!(weight > 0.0f) || !isfinite(weight) ||
        !isfinite(dx) || !isfinite(dy) || !isfinite(dz))
        return;

    int best = 0;
    float bestScore = -1.0f;
    for (int k = 0; k < PG_NUM_LOBES; k++) {
        const float* l = cell + k * PG_LOBE_STRIDE;
        // Kappa floor: untrained lobes (kappa 0) score by direction affinity
        // alone instead of uniformly — without it every deposit would
        // hard-assign to lobe 0 and the mixture would collapse.
        float kr = fmaxf(l[PG_L_KAPPA], 0.5f);
        float cosT = l[PG_L_MU_X] * dx + l[PG_L_MU_Y] * dy + l[PG_L_MU_Z] * dz;
        // Weight floor keeps dead lobes able to recapture nearby deposits.
        float pi = fmaxf(l[PG_L_WEIGHT], 0.01f);
        float score = pi * vmfPdf(kr, cosT);
        if (score > bestScore) { bestScore = score; best = k; }
    }

    float* sums = cell + PG_INT_BASE + best * 4;
    atomicAdd(&sums[0], dx * weight);
    atomicAdd(&sums[1], dy * weight);
    atomicAdd(&sums[2], dz * weight);
    atomicAdd(&sums[3], weight);
    atomicAdd(&cell[PG_INT_COUNT], 1.0f);

    // atomicMax on lastHitFrame: positive floats have same ordering as ints
    int frameAsInt = __float_as_int((float)frameIndex);
    atomicMax((int*)&cell[PG_LAST_HIT_FRAME], frameAsInt);
}

// Sampling and PDF evaluation read the per-vertex GuideLobe cached in
// registers by the integrator (raygen.cu) — see vmfSampleCached /
// vmfPdfCached in vmf_device.h. No per-evaluation cell reads remain here.
