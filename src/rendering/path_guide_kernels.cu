#include "path_guide_kernels.h"
#include "path_guide_grid.h"   // PG_* cell layout constants (host mirror)

namespace spectra {

namespace {

constexpr uint32_t BLOCK_SIZE = 256;

//------------------------------------------------------------------------------
// Refit: interval sums -> EMA cumulative sums -> vMF (mu, kappa).
//
// Runs between launches on the render stream: no shader atomics are in
// flight, so plain loads/stores are safe. kappa uses the Banerjee/Sra
// approximation kappa = Rbar*(3 - Rbar^2)/(1 - Rbar^2), clamped to 300
// (the Wood/Ulrich sampler and the stable vMF PDF both handle large kappa).
//------------------------------------------------------------------------------
__global__ void refitCellsKernel(float* data, uint32_t totalCells,
                                 float emaDecay, float currentFrame)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalCells) return;

    float* c = data + (size_t)idx * PG_ENTRY_STRIDE;

    // Fold the interval window into the EMA lifetime totals
    float cumX = emaDecay * c[PG_CUM_SUM_X] + c[PG_INT_SUM_X];
    float cumY = emaDecay * c[PG_CUM_SUM_Y] + c[PG_INT_SUM_Y];
    float cumZ = emaDecay * c[PG_CUM_SUM_Z] + c[PG_INT_SUM_Z];
    float cumW = emaDecay * c[PG_CUM_SUM_W] + c[PG_INT_SUM_W];
    float cumN = emaDecay * c[PG_CUM_COUNT] + c[PG_INT_COUNT];

    c[PG_CUM_SUM_X] = cumX;
    c[PG_CUM_SUM_Y] = cumY;
    c[PG_CUM_SUM_Z] = cumZ;
    c[PG_CUM_SUM_W] = cumW;
    c[PG_CUM_COUNT] = cumN;

    c[PG_INT_SUM_X] = 0.0f;
    c[PG_INT_SUM_Y] = 0.0f;
    c[PG_INT_SUM_Z] = 0.0f;
    c[PG_INT_SUM_W] = 0.0f;
    c[PG_INT_COUNT] = 0.0f;

    // Fit a single vMF lobe from the cumulative sums
    if (cumW >= 1.0f) {
        float len = sqrtf(cumX * cumX + cumY * cumY + cumZ * cumZ);
        if (len > 1e-9f) {
            float invLen = 1.0f / len;
            float rbar = fminf(len / cumW, 0.9999f);
            float kappa = rbar * (3.0f - rbar * rbar) / fmaxf(1.0f - rbar * rbar, 0.01f);
            kappa = fminf(kappa, 300.0f);

            c[PG_MU_X] = cumX * invLen;
            c[PG_MU_Y] = cumY * invLen;
            c[PG_MU_Z] = cumZ * invLen;
            c[PG_KAPPA] = kappa;
        }
    }

    // Cells that have never been hit keep lastHitFrame = 0 and would be
    // coarsened immediately; stamp them with the current frame instead.
    if (c[PG_LAST_HIT_FRAME] == 0.0f) {
        c[PG_LAST_HIT_FRAME] = currentFrame;
    }
}

//------------------------------------------------------------------------------
// Gather: re-layout cell data after a structure change.
//------------------------------------------------------------------------------
__global__ void gatherCellsKernel(float* dst, const float* src, const uint32_t* map,
                                  uint32_t totalNewCells, float currentFrame)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalNewCells) return;

    float* d = dst + (size_t)idx * PG_ENTRY_STRIDE;
    uint32_t m = map[idx];

    if (m == PG_GATHER_NEW_CELL) {
        // Fresh cell: zero everything, full coarsening grace period
        #pragma unroll
        for (uint32_t k = 0; k < PG_ENTRY_STRIDE; ++k) d[k] = 0.0f;
        d[PG_LAST_HIT_FRAME] = currentFrame;
        return;
    }

    if (m & PG_GATHER_LOBE_ONLY) {
        // Subdivided child: warm-start with the parent's lobe so guiding
        // keeps working in the region while the child trains its own fit.
        const float* s = src + (size_t)(m & PG_GATHER_INDEX_MASK) * PG_ENTRY_STRIDE;
        d[PG_MU_X] = s[PG_MU_X];
        d[PG_MU_Y] = s[PG_MU_Y];
        d[PG_MU_Z] = s[PG_MU_Z];
        d[PG_KAPPA] = s[PG_KAPPA];
        #pragma unroll
        for (uint32_t k = PG_INT_SUM_X; k < PG_ENTRY_STRIDE; ++k) d[k] = 0.0f;
        d[PG_LAST_HIT_FRAME] = currentFrame;
        return;
    }

    // Surviving cell: carry everything (including deposits made after the
    // structure snapshot — this gather runs at swap time on live data)
    const float* s = src + (size_t)m * PG_ENTRY_STRIDE;
    #pragma unroll
    for (uint32_t k = 0; k < PG_ENTRY_STRIDE; ++k) d[k] = s[k];
    if (d[PG_LAST_HIT_FRAME] == 0.0f) {
        d[PG_LAST_HIT_FRAME] = currentFrame;
    }
}

} // anonymous namespace

void launchRefitCells(float* data, uint32_t totalCells,
                      float emaDecay, uint32_t currentFrame,
                      cudaStream_t stream)
{
    if (!data || totalCells == 0) return;
    uint32_t blocks = (totalCells + BLOCK_SIZE - 1) / BLOCK_SIZE;
    refitCellsKernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
        data, totalCells, emaDecay, static_cast<float>(currentFrame));
}

void launchGatherCells(float* dst, const float* src, const uint32_t* map,
                       uint32_t totalNewCells, uint32_t currentFrame,
                       cudaStream_t stream)
{
    if (!dst || !map || totalNewCells == 0) return;
    uint32_t blocks = (totalNewCells + BLOCK_SIZE - 1) / BLOCK_SIZE;
    gatherCellsKernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
        dst, src, map, totalNewCells, static_cast<float>(currentFrame));
}

} // namespace spectra
