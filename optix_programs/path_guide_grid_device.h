#pragma once

//------------------------------------------------------------------------------
// Sparse path guide grid — device-side (collision-free, binary search lookup)
// Layout must match path_guide_grid.h. Used from raygen and closesthit.
//
// Path guiding: per-cell mixture of 2 von Mises–Fisher (vMF) lobes for
// importance sampling of directions. References:
//   - Müller et al., "Practical Path Guiding for Efficient Light-Transport
//     Simulation", EGSR 2017 (Computer Graphics Forum).
//   - Von Mises–Fisher distribution: Wikipedia; sampling via Wood/Ulrich:
//     w = 1 + (1/κ)*ln(ξ + (1-ξ)*exp(-2κ)), then orthonormal tangent + v on circle.
//------------------------------------------------------------------------------

#include <cuda_runtime.h>
#include "vmf_device.h"

#define PATH_GUIDE_VMF_FLOATS_PER_LOBE 3
#define PATH_GUIDE_LOBES_PER_SLOT      2
#define PATH_GUIDE_VMF_FLOATS          6   // 2 lobes * 3 floats
#define PATH_GUIDE_STATS_FLOATS        6   // sumX, sumY, sumZ, sumW, pi_0 (mixture weight), lastHitFrame
#define PATH_GUIDE_ENTRY_STRIDE        12  // vMF (6) + stats (6)
#define PATH_GUIDE_MIX_WEIGHT_OFFSET   10  // offset of pi_0 within cell data

// Sparse grid: per-level sorted Morton codes + data
struct SparsePathGuideDescriptorDevice {
    const unsigned long long* morton_codes;  // uint64_t, sorted per level
    float* data;                             // entry_stride floats per cell
    const unsigned int* level_offsets;       // [0 .. num_levels]
    unsigned int num_levels;
    unsigned int entry_stride;
    unsigned int base_resolution;
    float per_level_scale;
    float bounds_min[3];
    float bounds_max[3];
};

// Staging: append (level, ix, iy, iz) for build
struct PathGuideStagingDevice {
    unsigned int* buffer;   // 4 uints per entry
    unsigned int* count;    // atomic
    unsigned int capacity;
};

// Training: append (level, ix, iy, iz, dir_x, dir_y, dir_z, weight, frame) for vMF fit. 9 floats per entry.
struct PathGuideTrainingStagingDevice {
    float* buffer;   // 9 floats per entry
    unsigned int* count;
    unsigned int capacity;
};

__forceinline__ __device__ float sparseResolutionAtLevel(
    const SparsePathGuideDescriptorDevice& grid,
    unsigned int level)
{
    return (float)grid.base_resolution * powf(grid.per_level_scale, (float)level);
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

// Binary search for Morton code in level; returns index into level's array or ~0u if not found
__forceinline__ __device__ unsigned int sparseCellIndex(
    const SparsePathGuideDescriptorDevice& grid,
    unsigned int level,
    unsigned long long morton)
{
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

// True if cell (ix, iy, iz) at level exists in sparse grid
__forceinline__ __device__ bool sparseCellExists(
    const SparsePathGuideDescriptorDevice& grid,
    unsigned int level,
    int ix, int iy, int iz)
{
    if (grid.morton_codes == nullptr || grid.level_offsets == nullptr) return false;
    if (level >= grid.num_levels) return false;
    unsigned long long m = mortonEncode64((unsigned int)ix, (unsigned int)iy, (unsigned int)iz);
    return sparseCellIndex(grid, level, m) != 0xFFFFFFFFu;
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

__forceinline__ __device__ void sparseCellAABB(
    const SparsePathGuideDescriptorDevice& grid,
    unsigned int level,
    int ix, int iy, int iz,
    float& out_minx, float& out_miny, float& out_minz,
    float& out_maxx, float& out_maxy, float& out_maxz)
{
    float res = sparseResolutionAtLevel(grid, level);
    float resU = floorf(res);
    if (resU < 1.0f) resU = 1.0f;
    float inv_res = 1.0f / resU;
    out_minx = grid.bounds_min[0] + (float)ix * inv_res * (grid.bounds_max[0] - grid.bounds_min[0]);
    out_maxx = grid.bounds_min[0] + (float)(ix + 1) * inv_res * (grid.bounds_max[0] - grid.bounds_min[0]);
    out_miny = grid.bounds_min[1] + (float)iy * inv_res * (grid.bounds_max[1] - grid.bounds_min[1]);
    out_maxy = grid.bounds_min[1] + (float)(iy + 1) * inv_res * (grid.bounds_max[1] - grid.bounds_min[1]);
    out_minz = grid.bounds_min[2] + (float)iz * inv_res * (grid.bounds_max[2] - grid.bounds_min[2]);
    out_maxz = grid.bounds_min[2] + (float)(iz + 1) * inv_res * (grid.bounds_max[2] - grid.bounds_min[2]);
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

// Append (level, ix, iy, iz, dir_x, dir_y, dir_z, weight, frame) to training staging
// 9 floats per entry to include frame index for coarsening decisions
__forceinline__ __device__ void pathGuideTrainingAppend(
    const PathGuideTrainingStagingDevice& staging,
    unsigned int level, int ix, int iy, int iz,
    float dx, float dy, float dz, float weight,
    unsigned int frameIndex)
{
    if (staging.buffer == nullptr || staging.count == nullptr || staging.capacity == 0) return;
    unsigned int idx = atomicAdd(staging.count, 1u);
    if (idx >= staging.capacity) return;
    float* slot = staging.buffer + idx * 9;
    slot[0] = (float)level;
    slot[1] = (float)ix;
    slot[2] = (float)iy;
    slot[3] = (float)iz;
    slot[4] = dx;
    slot[5] = dy;
    slot[6] = dz;
    slot[7] = weight;
    slot[8] = (float)frameIndex;
}

// Cell data: 6 floats = lobe0(theta, phi, kappa), lobe1(theta, phi, kappa). kappa<=0 means lobe inactive.
// Sample from 2-lobe mixture (equal weight). Caller provides randoms u_lobe, u1, u2 in [0,1). If both lobes inactive returns false.
__forceinline__ __device__ bool pathGuideSampleDirection(
    const SparsePathGuideDescriptorDevice& grid,
    unsigned int global_index,
    float u_lobe, float u1, float u2,
    float& ox, float& oy, float& oz)
{
    float* cell = sparseCellDataPtr(grid, global_index);
    if (cell == nullptr) return false;
    float k0 = cell[2], k1 = cell[5];
    bool use0 = (k0 > 1e-6f);
    bool use1 = (k1 > 1e-6f);
    if (!use0 && !use1) return false;

    if (use0 && !use1) {
        float mx, my, mz;
        vmfSphericalToCartesian(cell[0], cell[1], mx, my, mz);
        vmfSample(mx, my, mz, k0, u1, u2, ox, oy, oz);
        return true;
    }
    if (!use0 && use1) {
        float mx, my, mz;
        vmfSphericalToCartesian(cell[3], cell[4], mx, my, mz);
        vmfSample(mx, my, mz, k1, u1, u2, ox, oy, oz);
        return true;
    }
    float pi0 = cell[PATH_GUIDE_MIX_WEIGHT_OFFSET];
    if (pi0 <= 0.0f || pi0 >= 1.0f) pi0 = 0.5f;  // safety fallback
    if (u_lobe < pi0) {
        float mx, my, mz;
        vmfSphericalToCartesian(cell[0], cell[1], mx, my, mz);
        vmfSample(mx, my, mz, k0, u1, u2, ox, oy, oz);
    } else {
        float mx, my, mz;
        vmfSphericalToCartesian(cell[3], cell[4], mx, my, mz);
        vmfSample(mx, my, mz, k1, u1, u2, ox, oy, oz);
    }
    return true;
}

// PDF of 2-lobe mixture at direction (ox, oy, oz)
// Weights must match pathGuideSampleDirection: when both lobes active, 50/50;
// when only one active, that lobe has weight 1.0.
__forceinline__ __device__ float pathGuidePdfDirection(
    const SparsePathGuideDescriptorDevice& grid,
    unsigned int global_index,
    float ox, float oy, float oz)
{
    float* cell = sparseCellDataPtr(grid, global_index);
    if (cell == nullptr) return 0.07957747154f;  // 1/(4π) uniform
    float k0 = cell[2], k1 = cell[5];
    bool use0 = (k0 > 1e-6f);
    bool use1 = (k1 > 1e-6f);
    if (!use0 && !use1) return 0.07957747154f;

    float p0 = 0.0f, p1 = 0.0f;
    if (use0) {
        float mx, my, mz;
        vmfSphericalToCartesian(cell[0], cell[1], mx, my, mz);
        p0 = vmfPdf(k0, mx*ox + my*oy + mz*oz);
    }
    if (use1) {
        float mx, my, mz;
        vmfSphericalToCartesian(cell[3], cell[4], mx, my, mz);
        p1 = vmfPdf(k1, mx*ox + my*oy + mz*oz);
    }

    // Match sampling: both active → use fitted mixture weight; one active → weight 1.0
    float p;
    if (use0 && use1) {
        float pi0 = cell[PATH_GUIDE_MIX_WEIGHT_OFFSET];
        if (pi0 <= 0.0f || pi0 >= 1.0f) pi0 = 0.5f;  // safety fallback
        p = pi0 * p0 + (1.0f - pi0) * p1;
    } else if (use0) {
        p = p0;
    } else {
        p = p1;
    }

    return fmaxf(p, 1e-10f);
}
