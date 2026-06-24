#include "path_guide_kernels.h"
#include "path_guide_cell_layout.h"  // PG_* cell layout + lobe init (shared with raygen)
#include "path_guide_hash_device.h"  // device cell table (shared with raygen)

namespace spectra {

namespace {

constexpr uint32_t BLOCK_SIZE = 256;

//------------------------------------------------------------------------------
// Refit: per-lobe interval sums -> EMA cumulative sums -> vMF mixture
// (mu_k, kappa_k, pi_k). One hard-assignment stepwise-EM M-step per call:
// shaders hard-assigned each deposit to a lobe (E-step at deposit time
// against the then-current lobes), this kernel folds those per-lobe sums and
// refits every lobe in place.
//
// Runs between launches on the render stream: no shader atomics are in
// flight, so plain loads/stores are safe. kappa uses the Banerjee/Sra
// approximation kappa = Rbar*(3 - Rbar^2)/(1 - Rbar^2), clamped to 300
// (the Wood/Ulrich sampler and the stable vMF PDF both handle large kappa).
//
// Dead lobes (no real evidence in a mature cell) are re-seeded near the
// strongest lobe with a wide exploratory kappa so they can recapture
// emerging secondary modes; kappa 1 keeps them below the sampling
// eligibility gate (kappa >= 2) until they earn evidence.
//
// Launched over the full capacity; threads past the live allocation count
// exit immediately (the counter read broadcasts from L2).
//------------------------------------------------------------------------------
__global__ void refitCellsKernel(float* data,
                                 const uint32_t* cellCounter, uint32_t cellCapacity,
                                 float emaDecay, float currentFrame)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t totalCells = *cellCounter;
    if (totalCells > cellCapacity) totalCells = cellCapacity;
    if (idx >= totalCells) return;

    float* c = data + (size_t)idx * PG_ENTRY_STRIDE;

    // Fold each lobe's interval window into its EMA lifetime totals
    float lobeW[PG_NUM_LOBES];
    float totalW = 0.0f;
    for (int k = 0; k < PG_NUM_LOBES; k++) {
        float* intS = c + PG_INT_BASE + k * PG_SUM_STRIDE;
        float* cumS = c + PG_CUMS_BASE + k * PG_SUM_STRIDE;
        float cumX = emaDecay * cumS[0] + intS[0];
        float cumY = emaDecay * cumS[1] + intS[1];
        float cumZ = emaDecay * cumS[2] + intS[2];
        float cumW = emaDecay * cumS[3] + intS[3];
        float cumD = emaDecay * cumS[PG_S_DIST] + intS[PG_S_DIST];  // sum of weight*dist
        cumS[0] = cumX; cumS[1] = cumY; cumS[2] = cumZ; cumS[3] = cumW; cumS[PG_S_DIST] = cumD;
        intS[0] = 0.0f; intS[1] = 0.0f; intS[2] = 0.0f; intS[3] = 0.0f; intS[PG_S_DIST] = 0.0f;
        lobeW[k] = cumW;
        totalW += cumW;
    }

    float cumN = emaDecay * c[PG_CUM_COUNT] + c[PG_INT_COUNT];
    c[PG_CUM_COUNT] = cumN;
    c[PG_INT_COUNT] = 0.0f;

    // M-step: refit every lobe with evidence; normalize mixture weights
    if (totalW >= 1.0f) {
        float invTotal = 1.0f / totalW;
        int strongest = 0;
        float strongestPi = -1.0f;

        for (int k = 0; k < PG_NUM_LOBES; k++) {
            float* l = c + k * PG_LOBE_STRIDE;
            float pi = lobeW[k] * invTotal;
            l[PG_L_WEIGHT] = pi;
            if (pi > strongestPi) { strongestPi = pi; strongest = k; }

            if (lobeW[k] >= 0.5f) {
                const float* cumS = c + PG_CUMS_BASE + k * PG_SUM_STRIDE;
                float len = sqrtf(cumS[0] * cumS[0] + cumS[1] * cumS[1] + cumS[2] * cumS[2]);
                if (len > 1e-9f) {
                    float invLen = 1.0f / len;
                    float rbar = fminf(len / lobeW[k], 0.99999f);
                    float kappa = rbar * (3.0f - rbar * rbar) / fmaxf(1.0f - rbar * rbar, 1e-4f);
                    // Confidence-gated sharpness ceiling. The old flat 300
                    // (~5 deg lobe) could not importance-sample compact bright
                    // emitters (a sun disc is ~0.5 deg; neon/env hotspots
                    // similar), capping exactly the "hard light" gains. Allow
                    // tight lobes only once the cell has real evidence: a
                    // too-tight, slightly miscentered lobe knocked off by the
                    // +-0.5 cell box-filter jitter has near-zero pdf at the true
                    // direction and raises its own variance. (Sampler + PDF
                    // already handle large kappa numerically — vmf_device.h.)
                    float kappaMax = (cumN >= 64.0f) ? 2000.0f : 300.0f;
                    kappa = fminf(kappa, kappaMax);
                    l[PG_L_MU_X] = cumS[0] * invLen;
                    l[PG_L_MU_Y] = cumS[1] * invLen;
                    l[PG_L_MU_Z] = cumS[2] * invLen;
                    l[PG_L_KAPPA] = kappa;
                    // Cache the vMF normalization term: halves the expf count
                    // in the shader's PDF/sampler hot path.
                    l[PG_L_EXP_NEG2K] = expf(-2.0f * kappa);
                    // Weighted mean distance to the incident radiance, for
                    // parallax-aware reprojection at lookup. cumS[PG_S_DIST] is
                    // the EMA sum of weight*dist; dividing by the weight sum
                    // (lobeW[k]) gives the mean — decay-invariant like rbar.
                    l[PG_L_MEAN_DIST] = cumS[PG_S_DIST] / lobeW[k];
                }
            }
        }

        // Re-seed dead lobes in mature cells toward UNDER-represented parts of
        // the sphere (deterministic; no RNG in this kernel). The old reseed
        // perturbed the strongest direction by ~20 deg, so all dead lobes
        // orbited the dominant mode and a far/opposite second emitter (another
        // window, a bounce light behind the receiver) was never discovered —
        // the mixture decayed toward unimodal. Instead, build an orthonormal
        // frame around the strongest mu and seed dead lobes at the opposite
        // hemisphere and the tangent directions (~90-180 deg away), spreading
        // coverage so emerging secondary modes can be captured. kappa stays 1
        // (below the sampling-eligibility gate) until the lobe earns evidence.
        if (cumN >= 32.0f) {
            const float* ls = c + strongest * PG_LOBE_STRIDE;
            float sx = ls[PG_L_MU_X], sy = ls[PG_L_MU_Y], sz = ls[PG_L_MU_Z];
            // Tangent frame around the strongest direction. up = +Y unless mu
            // is near-vertical, then +X (avoids a degenerate cross product).
            float upx = (fabsf(sy) < 0.9f) ? 0.0f : 1.0f;
            float upy = (fabsf(sy) < 0.9f) ? 1.0f : 0.0f;
            // T = cross(up, s), with up = (upx, upy, 0)
            float tx = upy * sz;
            float ty = -upx * sz;
            float tz = upx * sy - upy * sx;
            float tinv = rsqrtf(fmaxf(tx * tx + ty * ty + tz * tz, 1e-12f));
            tx *= tinv; ty *= tinv; tz *= tinv;
            float bx = sy * tz - sz * ty;   // cross(s, T)
            float by = sz * tx - sx * tz;
            float bz = sx * ty - sy * tx;
            const float seeds[4][3] = {
                { -sx, -sy, -sz },   // opposite hemisphere
                {  tx,  ty,  tz },   // +tangent
                {  bx,  by,  bz },   // +bitangent
                { -tx, -ty, -tz },   // -tangent
            };
            int seedIdx = 0;
            for (int k = 0; k < PG_NUM_LOBES; k++) {
                if (k == strongest || lobeW[k] >= 0.5f) continue;
                float* l = c + k * PG_LOBE_STRIDE;
                int si = seedIdx & 3; seedIdx++;
                float mx = seeds[si][0], my = seeds[si][1], mz = seeds[si][2];
                float invLen = rsqrtf(fmaxf(mx * mx + my * my + mz * mz, 1e-12f));
                l[PG_L_MU_X] = mx * invLen;
                l[PG_L_MU_Y] = my * invLen;
                l[PG_L_MU_Z] = mz * invLen;
                l[PG_L_KAPPA] = 1.0f;                 // exploratory: below the
                l[PG_L_EXP_NEG2K] = 0.13533528f;      // sampling gate; exp(-2)
                l[PG_L_WEIGHT] = 0.02f;
                l[PG_L_MEAN_DIST] = 0.0f;             // no distance evidence yet
                // Small synthetic evidence so the re-seed survives a few
                // refits while it competes for deposits.
                float* cumS = c + PG_CUMS_BASE + k * PG_SUM_STRIDE;
                float seedW = 0.02f * totalW;
                cumS[0] = l[PG_L_MU_X] * seedW * 0.5f;
                cumS[1] = l[PG_L_MU_Y] * seedW * 0.5f;
                cumS[2] = l[PG_L_MU_Z] * seedW * 0.5f;
                cumS[3] = seedW;
                cumS[PG_S_DIST] = 0.0f;               // distant-light default until trained
            }
        }
    }

    // Cells that have never been hit keep lastHitFrame = 0; stamp them with
    // the current frame so the inspector's age display starts at creation.
    if (c[PG_LAST_HIT_FRAME] == 0.0f) {
        c[PG_LAST_HIT_FRAME] = currentFrame;
    }
}

//------------------------------------------------------------------------------
// Subdivision: insert the 8 children of any cell whose EMA deposit count
// crossed the threshold (sample-count criterion, PPG-flavored — spatial
// refinement follows where the samples are; directional complexity is the
// mixture's job, not subdivision's).
//
// Once children exist, the top-down lookup targets them, the parent stops
// receiving deposits, and its EMA count decays below the threshold within a
// few refits — so re-examining subdivided parents costs only hash probes
// until then (pgTableInsert is idempotent).
//------------------------------------------------------------------------------
__global__ void subdivideCellsKernel(PathGuideTableDevice table,
                                     const uint32_t* countSnapshot,
                                     uint32_t maxLevel, float countThreshold,
                                     float currentFrame)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Bound by the pre-launch snapshot, NOT the live counter: children
    // inserted by this pass must not be processed in the same pass (their
    // warm-started evidence could cascade-subdivide, and their payload may
    // still be mid-write by the inserting thread).
    uint32_t totalCells = *countSnapshot;
    if (totalCells > table.cell_capacity) totalCells = table.cell_capacity;
    if (idx >= totalCells) return;

    float* c = table.data + (size_t)idx * table.entry_stride;
    if (c[PG_CUM_COUNT] < countThreshold) return;

    unsigned long long key = table.cell_keys[idx];
    uint32_t level = (uint32_t)(key >> 48);
    if (level >= maxLevel) return;
    unsigned long long morton = key & ((1ull << 48) - 1);

    for (uint32_t k = 0; k < 8; k++) {
        bool inserted = false;
        uint32_t childIdx = pgTableInsert(table, level + 1, pgChildMorton(morton, k), &inserted);
        if (inserted && childIdx != PG_INVALID_CELL) {
            float* d = table.data + (size_t)childIdx * table.entry_stride;
            // Warm start: parent's full mixture so guiding keeps working in
            // the region, 1/8 of the parent's cumulative evidence so the
            // confidence ramp survives the split. Interval sums stay zero
            // (data is pre-zeroed; the winner is the only non-atomic writer
            // and shaders only atomicAdd the disjoint interval fields).
            for (int f = 0; f < PG_NUM_LOBES * PG_LOBE_STRIDE; f++) {
                d[f] = c[f];
            }
            for (int lk = 0; lk < PG_NUM_LOBES; lk++) {
                const float* pc = c + PG_CUMS_BASE + lk * PG_SUM_STRIDE;
                float* dc = d + PG_CUMS_BASE + lk * PG_SUM_STRIDE;
                dc[0] = pc[0] * 0.125f;
                dc[1] = pc[1] * 0.125f;
                dc[2] = pc[2] * 0.125f;
                dc[3] = pc[3] * 0.125f;
                dc[PG_S_DIST] = pc[PG_S_DIST] * 0.125f;  // scale distance sum with the rest
            }
            d[PG_CUM_COUNT] = c[PG_CUM_COUNT] * 0.125f;
            d[PG_LAST_HIT_FRAME] = currentFrame;
        }
    }
}

//------------------------------------------------------------------------------
// Lobe (re)initialization: tetrahedral starting mixture for every allocated
// cell. Used after clear() — a zeroed mixture would collapse the hard
// assignment onto lobe 0 (all scores equal).
//------------------------------------------------------------------------------
__global__ void initCellsKernel(float* data,
                                const uint32_t* cellCounter, uint32_t cellCapacity)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t totalCells = *cellCounter;
    if (totalCells > cellCapacity) totalCells = cellCapacity;
    if (idx >= totalCells) return;

    pgInitCellLobes(data + (size_t)idx * PG_ENTRY_STRIDE);
}

} // anonymous namespace

void launchRefitCells(float* data,
                      const uint32_t* cellCounter, uint32_t cellCapacity,
                      float emaDecay, uint32_t currentFrame,
                      cudaStream_t stream)
{
    if (!data || !cellCounter || cellCapacity == 0) return;
    uint32_t blocks = (cellCapacity + BLOCK_SIZE - 1) / BLOCK_SIZE;
    refitCellsKernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
        data, cellCounter, cellCapacity, emaDecay, static_cast<float>(currentFrame));
}

void launchSubdivideCells(uint64_t* hashKeys, uint32_t* hashValues,
                          uint32_t hashTableSize, uint32_t hashShift,
                          uint64_t* cellKeys, uint32_t* cellCounter, uint32_t cellCapacity,
                          const uint32_t* countSnapshot,
                          float* data, uint32_t entryStride,
                          uint32_t maxLevel, float countThreshold,
                          uint32_t currentFrame,
                          cudaStream_t stream)
{
    if (!data || !cellCounter || !countSnapshot || !hashKeys || cellCapacity == 0) return;
    PathGuideTableDevice table = {};
    table.hash_keys = reinterpret_cast<unsigned long long*>(hashKeys);
    table.hash_values = hashValues;
    table.hash_table_size = hashTableSize;
    table.hash_shift = hashShift;
    table.cell_keys = reinterpret_cast<unsigned long long*>(cellKeys);
    table.cell_counter = cellCounter;
    table.cell_capacity = cellCapacity;
    table.data = data;
    table.entry_stride = entryStride;

    uint32_t blocks = (cellCapacity + BLOCK_SIZE - 1) / BLOCK_SIZE;
    subdivideCellsKernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
        table, countSnapshot, maxLevel, countThreshold, static_cast<float>(currentFrame));
}

void launchInitCells(float* data,
                     const uint32_t* cellCounter, uint32_t cellCapacity,
                     cudaStream_t stream)
{
    if (!data || !cellCounter || cellCapacity == 0) return;
    uint32_t blocks = (cellCapacity + BLOCK_SIZE - 1) / BLOCK_SIZE;
    initCellsKernel<<<blocks, BLOCK_SIZE, 0, stream>>>(data, cellCounter, cellCapacity);
}

} // namespace spectra
