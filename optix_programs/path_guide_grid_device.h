#pragma once

//------------------------------------------------------------------------------
// Sparse path guide grid — device-side (collision-free, hash lookup)
// Layout must match src/rendering/path_guide_grid.h. Used from raygen.
//
// Path guiding: one von Mises–Fisher (vMF) lobe per cell for importance
// sampling of incident radiance. References:
//   - Müller et al., "Practical Path Guiding for Efficient Light-Transport
//     Simulation", EGSR 2017 (Computer Graphics Forum).
//   - Von Mises–Fisher distribution; sampling via Wood/Ulrich.
//
// A single lobe per cell is deliberate: the aggregate sums (sumX/Y/Z/W) the
// shaders accumulate are sufficient statistics for ONE vMF lobe only — they
// cannot separate the modes of a genuinely bimodal distribution, so a second
// lobe fit from the same sums never represented real bimodality (it only
// reacted to temporal lighting changes). Spatial subdivision (refinement)
// handles multimodal regions instead; the BSDF leg of the MIS covers the
// rest. A future per-lobe-statistics online EM can reintroduce mixtures.
//------------------------------------------------------------------------------

#include <cuda_runtime.h>
#include "vmf_device.h"

// ─── Cell data layout (named offsets, 16 floats = 64 bytes per cell) ─────────
//
//  [0-2]:  fitted vMF mean direction mu (unit vector); written by refit kernel
//  [3]:    fitted vMF kappa; <= 0 means no valid lobe yet
//  [4-7]:  interval sums (sumX, sumY, sumZ, sumW) — atomicAdd from shaders,
//          consumed and zeroed by the device refit kernel
//  [8-11]: cumulative sums — EMA-decayed lifetime totals, owned by the refit
//          kernel (never touched by shaders)
//  [12]:   lastHitFrame — atomicMax (positive floats sort like ints)
//  [13]:   interval deposit count — atomicAdd 1 per deposit
//  [14]:   cumulative deposit count — EMA-decayed, owned by refit kernel
//  [15]:   reserved
//
// The hot fields for sampling (mu + kappa) are the first 16 bytes, so the
// sampling/PDF path touches a single aligned segment per cell.
// ─────────────────────────────────────────────────────────────────────────────

#define PG_MU_X              0
#define PG_MU_Y              1
#define PG_MU_Z              2
#define PG_KAPPA             3
#define PG_INT_SUM_X         4
#define PG_INT_SUM_Y         5
#define PG_INT_SUM_Z         6
#define PG_INT_SUM_W         7
#define PG_CUM_SUM_X         8
#define PG_CUM_SUM_Y         9
#define PG_CUM_SUM_Z         10
#define PG_CUM_SUM_W         11
#define PG_LAST_HIT_FRAME    12
#define PG_INT_COUNT         13
#define PG_CUM_COUNT         14
#define PG_RESERVED          15
#define PG_ENTRY_STRIDE      16

// Sparse grid: lightweight descriptor with pointers into __constant__ params.
// Does NOT copy level_resolutions[16] — reads directly from params to avoid
// 64 bytes of register pressure / local memory spill per hit.
struct SparsePathGuideDescriptorDevice {
    const unsigned long long* morton_codes;  // uint64_t, sorted per level
    float* data;                             // entry_stride floats per cell
    const unsigned int* level_offsets;       // [0 .. num_levels]
    unsigned int num_levels;
    unsigned int entry_stride;
    float bounds_min[3];
    float bounds_max[3];

    // Hash table for O(1) cell lookup (replaces binary search)
    const unsigned long long* hash_keys;     // (level<<48 | morton), empty = 0xFFFFFFFFFFFFFFFF
    const unsigned int* hash_values;         // flat cell index
    unsigned int hash_table_size;            // power of 2
    unsigned int hash_shift;                 // 64 - log2(hash_table_size)
};

// Staging: append (level, ix, iy, iz) for build
struct PathGuideStagingDevice {
    unsigned int* buffer;   // 4 uints per entry
    unsigned int* count;    // atomic
    unsigned int capacity;
};


// Reads level resolution directly from __constant__ params to avoid copying
// the 16-element array into each thread's local storage.
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

// 3D Morton encode (64-bit), up to 21 bits per axis. Must match host.
__forceinline__ __device__ unsigned long long mortonSpread3(unsigned long long x) {
    x &= 0x1fffffULL;
    x = (x | x << 32) & 0x001f00000000ffffULL;
    x = (x | x << 16) & 0x001f0000ff0000ffULL;
    x = (x | x << 8)  & 0x010f00f00f00f00fULL;
    x = (x | x << 4)  & 0x10c30c30c30c30c3ULL;
    x = (x | x << 2)  & 0x1249249249249249ULL;
    return x;
}
__forceinline__ __device__ unsigned long long mortonEncode64(unsigned int ix, unsigned int iy, unsigned int iz) {
    return mortonSpread3(ix) | (mortonSpread3(iy) << 1) | (mortonSpread3(iz) << 2);
}

// Hash table lookup: O(1) average, linear probing. Returns flat cell index or ~0u.
__forceinline__ __device__ unsigned int sparseCellIndexHash(
    const SparsePathGuideDescriptorDevice& grid,
    unsigned int level,
    unsigned long long morton)
{
    if (grid.hash_table_size == 0 || grid.hash_keys == nullptr) return 0xFFFFFFFFu;

    unsigned long long key = ((unsigned long long)level << 48) | morton;
    unsigned int mask = grid.hash_table_size - 1;
    // Fibonacci hashing: multiply by golden ratio, take top bits
    unsigned int slot = (unsigned int)((key * 0x9E3779B97F4A7C15ULL) >> grid.hash_shift) & mask;

    for (unsigned int i = 0; i < 32; i++) {  // max 32 probes (generous for 50% load)
        unsigned long long k = grid.hash_keys[slot];
        if (k == key) return grid.hash_values[slot];
        if (k == 0xFFFFFFFFFFFFFFFFULL) return 0xFFFFFFFFu;  // empty slot
        slot = (slot + 1) & mask;
    }
    return 0xFFFFFFFFu;
}

// Binary search fallback (used only when hash table is not available)
__forceinline__ __device__ unsigned int sparseCellIndex(
    const SparsePathGuideDescriptorDevice& grid,
    unsigned int level,
    unsigned long long morton)
{
    // Prefer hash table when available
    if (grid.hash_keys != nullptr && grid.hash_table_size > 0) {
        return sparseCellIndexHash(grid, level, morton);
    }
    unsigned int start = grid.level_offsets[level];
    unsigned int end = grid.level_offsets[level + 1];
    while (start < end) {
        unsigned int mid = start + (end - start) / 2;
        unsigned long long m = grid.morton_codes[mid];
        if (m < morton)
            start = mid + 1;
        else if (m > morton)
            end = mid;
        else
            return mid;
    }
    return 0xFFFFFFFFu;
}

// Pointer to cell data or nullptr if not found
__forceinline__ __device__ float* sparseCellDataPtr(
    const SparsePathGuideDescriptorDevice& grid,
    unsigned int global_index)
{
    if (grid.data == nullptr || global_index == 0xFFFFFFFFu) return nullptr;
    return grid.data + (global_index * grid.entry_stride);
}

// Hierarchical lookup: find finest existing cell for position (searches fine to coarse)
// Returns global index and level, or 0xFFFFFFFF if no cell exists at any level
__forceinline__ __device__ unsigned int hierarchicalCellLookup(
    const SparsePathGuideDescriptorDevice& grid,
    float px, float py, float pz,
    unsigned int maxLevel,
    unsigned int minLevel,
    unsigned int* outLevel)
{
    if (grid.morton_codes == nullptr || grid.level_offsets == nullptr) {
        *outLevel = 0;
        return 0xFFFFFFFFu;
    }

    float nx, ny, nz;
    worldToNormalized(grid, px, py, pz, nx, ny, nz);

    // Clamp to valid range
    if (nx < 0.0f || nx > 1.0f || ny < 0.0f || ny > 1.0f || nz < 0.0f || nz > 1.0f) {
        *outLevel = 0;
        return 0xFFFFFFFFu;
    }

    // Search from finest to coarsest level
    unsigned int searchMax = (maxLevel < grid.num_levels) ? maxLevel : (grid.num_levels - 1);
    unsigned int searchMin = (minLevel < grid.num_levels) ? minLevel : 0;

    for (int lev = (int)searchMax; lev >= (int)searchMin; --lev) {
        int ix, iy, iz;
        normalizedToCell(nx, ny, nz, (unsigned int)lev, grid, ix, iy, iz);
        unsigned long long morton = mortonEncode64((unsigned int)ix, (unsigned int)iy, (unsigned int)iz);
        unsigned int idx = sparseCellIndex(grid, (unsigned int)lev, morton);
        if (idx != 0xFFFFFFFFu) {
            *outLevel = (unsigned int)lev;
            return idx;
        }
    }

    *outLevel = 0;
    return 0xFFFFFFFFu;
}

// Append (level, ix, iy, iz) to staging; no-op if staging null or full
__forceinline__ __device__ void pathGuideStagingAppend(
    const PathGuideStagingDevice& staging,
    unsigned int level, int ix, int iy, int iz)
{
    if (staging.buffer == nullptr || staging.count == nullptr || staging.capacity == 0) return;
    unsigned int idx = atomicAdd(staging.count, 1u);
    if (idx >= staging.capacity) return;
    unsigned int* slot = staging.buffer + idx * 4;
    slot[0] = level;
    slot[1] = (unsigned int)ix;
    slot[2] = (unsigned int)iy;
    slot[3] = (unsigned int)iz;
}

// Train a cell: accumulate an importance-weighted direction deposit into the
// interval sums. The refit kernel (path_guide_kernels.cu) periodically folds
// these into the cumulative sums and refits mu/kappa — shaders never write
// the fields the sampling path reads, so sampling PDFs stay consistent
// within a launch.
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

    atomicAdd(&cell[PG_INT_SUM_X], dx * weight);
    atomicAdd(&cell[PG_INT_SUM_Y], dy * weight);
    atomicAdd(&cell[PG_INT_SUM_Z], dz * weight);
    atomicAdd(&cell[PG_INT_SUM_W], weight);
    atomicAdd(&cell[PG_INT_COUNT], 1.0f);

    // atomicMax on lastHitFrame: positive floats have same ordering as ints
    int frameAsInt = __float_as_int((float)frameIndex);
    atomicMax((int*)&cell[PG_LAST_HIT_FRAME], frameAsInt);
}

// Sample a direction from the cell's vMF lobe. Returns false if no valid lobe.
__forceinline__ __device__ bool pathGuideSampleDirection(
    const SparsePathGuideDescriptorDevice& grid,
    unsigned int global_index,
    float u1, float u2,
    float& ox, float& oy, float& oz)
{
    float* cell = sparseCellDataPtr(grid, global_index);
    if (cell == nullptr) return false;
    float kappa = cell[PG_KAPPA];
    if (kappa <= 1e-6f) return false;
    vmfSample(cell[PG_MU_X], cell[PG_MU_Y], cell[PG_MU_Z], kappa, u1, u2, ox, oy, oz);
    return true;
}

// PDF of the cell's vMF lobe at direction (ox, oy, oz).
// Must match pathGuideSampleDirection exactly.
__forceinline__ __device__ float pathGuidePdfDirection(
    const SparsePathGuideDescriptorDevice& grid,
    unsigned int global_index,
    float ox, float oy, float oz)
{
    float* cell = sparseCellDataPtr(grid, global_index);
    if (cell == nullptr) return 0.07957747154f;  // 1/(4π) uniform
    float kappa = cell[PG_KAPPA];
    if (kappa <= 1e-6f) return 0.07957747154f;
    float cosTheta = cell[PG_MU_X] * ox + cell[PG_MU_Y] * oy + cell[PG_MU_Z] * oz;
    return fmaxf(vmfPdf(kappa, cosTheta), 1e-10f);
}

//------------------------------------------------------------------------------
// Trilinear interpolation for smooth cell boundary transitions.
// Eliminates visible grid squares on flat surfaces by blending between
// neighboring cells' vMF distributions.
//------------------------------------------------------------------------------

struct TrilinearInfo {
    unsigned int cellIdx[8];  // Global cell index per corner (0xFFFFFFFF if missing/invalid)
    float weight[8];          // Trilinear weight per corner
    float weightSum;          // Sum of valid weights
};

// Compute trilinear neighbors at a given level. Uses cell-centered interpolation
// so a position at the center of a cell gets 100% of that cell. Includes ALL
// existing cells (no lobe filtering). Use filterTrilinearByValidLobes() to get
// a guiding-safe subset.
__forceinline__ __device__ TrilinearInfo computeTrilinearNeighbors(
    const SparsePathGuideDescriptorDevice& grid,
    float px, float py, float pz,
    unsigned int level)
{
    TrilinearInfo info;
    for (int i = 0; i < 8; i++) {
        info.cellIdx[i] = 0xFFFFFFFFu;
        info.weight[i] = 0.0f;
    }
    info.weightSum = 0.0f;

    float nx, ny, nz;
    worldToNormalized(grid, px, py, pz, nx, ny, nz);
    nx = fminf(fmaxf(nx, 0.0f), 0.9999999f);
    ny = fminf(fmaxf(ny, 0.0f), 0.9999999f);
    nz = fminf(fmaxf(nz, 0.0f), 0.9999999f);

    float res = sparseResolutionAtLevel(grid, level);
    int resU = (int)floorf(res);
    if (resU < 1) resU = 1;

    // Cell-centered: shift by -0.5 so cell centers map to integer coordinates.
    // At cell center -> fx=0 -> 100% this cell. At boundary -> fx=0.5 -> 50/50.
    float cxf = nx * (float)resU - 0.5f;
    float cyf = ny * (float)resU - 0.5f;
    float czf = nz * (float)resU - 0.5f;

    int ix0 = (int)floorf(cxf);
    int iy0 = (int)floorf(cyf);
    int iz0 = (int)floorf(czf);
    float fx = cxf - (float)ix0;
    float fy = cyf - (float)iy0;
    float fz = czf - (float)iz0;

    float wx[2] = { 1.0f - fx, fx };
    float wy[2] = { 1.0f - fy, fy };
    float wz[2] = { 1.0f - fz, fz };

    for (int dz = 0; dz < 2; dz++) {
        for (int dy = 0; dy < 2; dy++) {
            for (int dx = 0; dx < 2; dx++) {
                int ci = dz * 4 + dy * 2 + dx;
                int cx = ix0 + dx;
                int cy = iy0 + dy;
                int cz = iz0 + dz;
                float w = wx[dx] * wy[dy] * wz[dz];

                if (cx < 0 || cx >= resU || cy < 0 || cy >= resU || cz < 0 || cz >= resU)
                    continue;

                unsigned long long morton = mortonEncode64((unsigned int)cx, (unsigned int)cy, (unsigned int)cz);
                unsigned int idx = sparseCellIndex(grid, level, morton);
                if (idx == 0xFFFFFFFFu) continue;

                info.cellIdx[ci] = idx;
                info.weight[ci] = w;
                info.weightSum += w;
            }
        }
    }

    return info;
}

// Filter trilinear info to only include cells with valid vMF lobes (kappa > 0).
// Use this for guiding (sampling + PDF) — ensures we only sample from trained cells.
__forceinline__ __device__ TrilinearInfo filterTrilinearByValidLobes(
    const SparsePathGuideDescriptorDevice& grid,
    const TrilinearInfo& all)
{
    TrilinearInfo filtered;
    filtered.weightSum = 0.0f;
    for (int i = 0; i < 8; i++) {
        filtered.cellIdx[i] = 0xFFFFFFFFu;
        filtered.weight[i] = 0.0f;
        if (all.cellIdx[i] != 0xFFFFFFFFu) {
            float* cell = sparseCellDataPtr(grid, all.cellIdx[i]);
            if (cell != nullptr && cell[PG_KAPPA] > 1e-6f) {
                filtered.cellIdx[i] = all.cellIdx[i];
                filtered.weight[i] = all.weight[i];
                filtered.weightSum += all.weight[i];
            }
        }
    }
    return filtered;
}

// Stochastically select one cell from trilinear neighbors, weighted by
// trilinear weights. Returns 0xFFFFFFFF if no valid neighbor exists.
__forceinline__ __device__ unsigned int stochasticSelectCell(
    const TrilinearInfo& info, float rand)
{
    if (info.weightSum <= 0.0f) return 0xFFFFFFFFu;

    float target = rand * info.weightSum;
    float cumulative = 0.0f;
    for (int i = 0; i < 8; i++) {
        if (info.cellIdx[i] == 0xFFFFFFFFu) continue;
        cumulative += info.weight[i];
        if (target <= cumulative) return info.cellIdx[i];
    }
    // Fallback: return last valid cell
    for (int i = 7; i >= 0; i--) {
        if (info.cellIdx[i] != 0xFFFFFFFFu) return info.cellIdx[i];
    }
    return 0xFFFFFFFFu;
}

// Trilinear-weighted PDF for MIS. Computes the weighted average of all valid
// neighbor cells' PDFs, matching the stochastic sampling distribution exactly:
// the sampler picks cell i with probability weight[i]/weightSum and then
// samples that cell's vMF, so the marginal density is this weighted mixture.
__forceinline__ __device__ float trilinearGuidePdf(
    const SparsePathGuideDescriptorDevice& grid,
    const TrilinearInfo& info,
    float ox, float oy, float oz)
{
    if (info.weightSum <= 0.0f) return 0.07957747154f;  // 1/(4*pi) uniform

    float pdfSum = 0.0f;
    for (int i = 0; i < 8; i++) {
        if (info.cellIdx[i] == 0xFFFFFFFFu) continue;
        float cellPdf = pathGuidePdfDirection(grid, info.cellIdx[i], ox, oy, oz);
        pdfSum += info.weight[i] * cellPdf;
    }

    return fmaxf(pdfSum / info.weightSum, 1e-10f);
}
