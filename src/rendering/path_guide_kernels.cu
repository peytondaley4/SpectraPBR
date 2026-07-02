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
                                 const unsigned long long* cellKeys,
                                 const uint32_t* cellCounter, uint32_t cellCapacity,
                                 float emaDecay, float baseCellSize,
                                 float currentFrame)
{
    // Maturity decays much slower than the fit window: evidence accumulates
    // over ~1/(1-0.98) = 50 refits, so trickle-fed cells (dim regions, hard
    // lights) eventually activate guiding instead of deadlocking at zero
    // confidence. The distribution itself still tracks the fast 0.85 window.
    constexpr float MATURITY_DECAY = 0.98f;

    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t totalCells = *cellCounter;
    if (totalCells > cellCapacity) totalCells = cellCapacity;
    if (idx >= totalCells) return;

    float* c = data + (size_t)idx * PG_ENTRY_STRIDE;

    // This cell's edge length: base size halves per level (per_level_scale=2).
    uint32_t cellLevel = (uint32_t)(cellKeys[idx] >> 48);
    float cellSize = baseCellSize * exp2f(-(float)cellLevel);

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
    float maturity = MATURITY_DECAY * c[PG_MATURITY] + c[PG_INT_COUNT];
    c[PG_CUM_COUNT] = cumN;
    c[PG_MATURITY] = maturity;
    c[PG_INT_COUNT] = 0.0f;

    // Fold the interval spatial first moments into their EMA totals on the same
    // decay footing as the weight sums, so the centroid the subdivision kernel
    // reads (c_a = S_a / W) stays consistent. Sum(w^2) folds with decay^2: an
    // old deposit's weight is decay^k * w, so its SQUARE is decay^(2k) * w^2 —
    // this makes nEff = W^2/Sum(w^2) exactly the Kish effective sample size of
    // the same exponentially-decayed population the centroid is estimated from.
    c[PG_CUM_SR_X] = emaDecay * c[PG_CUM_SR_X] + c[PG_INT_SR_X];
    c[PG_CUM_SR_Y] = emaDecay * c[PG_CUM_SR_Y] + c[PG_INT_SR_Y];
    c[PG_CUM_SR_Z] = emaDecay * c[PG_CUM_SR_Z] + c[PG_INT_SR_Z];
    c[PG_CUM_SW2]  = emaDecay * emaDecay * c[PG_CUM_SW2] + c[PG_INT_SW2];
    c[PG_INT_SR_X] = 0.0f;
    c[PG_INT_SR_Y] = 0.0f;
    c[PG_INT_SR_Z] = 0.0f;
    c[PG_INT_SW2]  = 0.0f;

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
                    // Weighted mean distance to the incident radiance, for
                    // parallax-aware reprojection at lookup AND the geometric
                    // kappa cap below. cumS[PG_S_DIST] is the EMA sum of
                    // weight*dist; dividing by the weight sum gives the mean —
                    // decay-invariant like rbar.
                    float meanDist = cumS[PG_S_DIST] / lobeW[k];

                    // GEOMETRY-AWARE sharpness ceiling. A lobe is consumed up
                    // to ~1 cell away from where it was fit (the +-0.5-cell
                    // box-filter jitter), and the parallax reprojection can be
                    // off by a fraction of a cell (cell-center pivot vs true
                    // deposit centroid). Both put an angular error of order
                    // cellSize/meanDist on the borrowed lobe — so its width
                    // must not be narrower than that error, i.e.
                    // kappa <= (2*meanDist/cellSize)^2 (vMF std ~ 1/sqrt(k)).
                    // A NEAR light over fine cells caps at a few tens — a
                    // tight lobe there would miss the source entirely when
                    // borrowed, wasting the guide's sample share and leaving
                    // firefly residue for the clamp to eat (the cell-shaped
                    // dark/noisy checkerboard around a close small light). A
                    // DISTANT source (meanDist >> cellSize, incl. env at
                    // sceneFar) still earns up to 2000 (~2 deg), which is the
                    // point of the raised ceiling. Evidence gate uses the
                    // slow-decayed MATURITY, not the rate-based fast count.
                    // Evidence gates only the 300 -> 2000 unlock; the geometric
                    // cap is a SAFETY and applies whenever meanDist is known
                    // (a near-field lobe is fragile at 300 too).
                    float kappaMax = (maturity >= 64.0f) ? 2000.0f : 300.0f;
                    if (meanDist > 1e-4f) {
                        float ratio = 2.0f * meanDist / fmaxf(cellSize, 1e-6f);
                        kappaMax = fminf(kappaMax, fmaxf(ratio * ratio, 8.0f));
                    }
                    kappa = fminf(kappa, kappaMax);
                    l[PG_L_MU_X] = cumS[0] * invLen;
                    l[PG_L_MU_Y] = cumS[1] * invLen;
                    l[PG_L_MU_Z] = cumS[2] * invLen;
                    l[PG_L_KAPPA] = kappa;
                    // Cache the vMF normalization term: halves the expf count
                    // in the shader's PDF/sampler hot path.
                    l[PG_L_EXP_NEG2K] = expf(-2.0f * kappa);
                    l[PG_L_MEAN_DIST] = meanDist;
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
// Subdivision: insert the 8 children of any cell that contains a spatial
// BARRIER. A pure sample-count criterion refines wherever the most samples
// land — but under uniform primary visibility every floor cell gets equal
// traffic, so the grid subdivides uniformly and gives almost no benefit over
// no guiding, while a high-VARIANCE feature like a caustic (same count, very
// different radiance) is ignored. Instead split only where the radiance varies
// SPATIALLY across the cell: with the weighted centroid c_a = S_a / W of the
// deposit positions (S_a = EMA Sum(w*rel_a), W = EMA weight sum, rel in
// [-1,1]), |centroid|^2 = sum_a S_a^2 / W^2 measures how off-center the
// radiance is. It is scale-INVARIANT (W cancels), so a uniform cell — bright or
// dark — has centroid ~0 and is never split, while the boundary of a difference
// (a caustic edge, a shadow line) has a large centroid and is refined. Count is
// kept only as a min-sample gate so the centroid estimate is meaningful.
//
// Once children exist, the top-down lookup targets them, the parent stops
// receiving deposits, and its EMA evidence decays within a few refits — so
// re-examining subdivided parents costs only hash probes (pgTableInsert is
// idempotent).
//------------------------------------------------------------------------------
__global__ void subdivideCellsKernel(PathGuideTableDevice table,
                                     const uint32_t* countSnapshot,
                                     uint32_t maxLevel, float minCount,
                                     float contrastThreshold,
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

    // Noise-floor multiplier for the contrast test. Under the null hypothesis
    // (spatially uniform radiance, rel ~ U[-1,1] per axis) the centroid is pure
    // estimation noise with E[|centroid|^2] = 3*Var(rel)/nEff ~= 1/nEff, where
    // nEff = W^2/Sum(w^2) is the Kish effective sample size. Requiring
    // contrast > LAMBDA/nEff (~4x the null mean, several sigma) means a weight
    // spike (one firefly collapses nEff to a handful while the deposit COUNT
    // stays large) can never masquerade as spatial structure. This is the
    // guard the deposit-count gate cannot provide: it counts deposits, not
    // effective weight mass, and Li/pdf weights are heavy-tailed.
    constexpr float CONTRAST_NOISE_LAMBDA = 4.0f;
    // Escape hatch: a first-moment centroid is blind to EVEN-SYMMETRIC spatial
    // variation (a light pool centered in the cell, a stripe through the
    // middle) — such cells would never split under a hard contrast gate. A
    // cell that keeps absorbing traffic long past maturity splits anyway. The
    // waste is bounded: an ultra-hot but genuinely uniform cell splits ONCE,
    // its children each inherit ~1/8 of the traffic and never reach the hatch
    // again — at most one surplus level, in exchange for never deadlocking
    // real-but-symmetric structure at coarse resolution.
    constexpr float COUNT_HATCH_MULT = 8.0f;

    float* c = table.data + (size_t)idx * table.entry_stride;
    // Min-sample gate: affordability + maturity — children inherit 1/8 of the
    // parent's evidence, so a split below this leaves them under-trained; and
    // the centroid is only meaningful with enough deposits.
    if (c[PG_CUM_COUNT] < minCount) return;

    // (minCount <= 0 disables the hatch rather than the contrast test: with a
    // zero gate, `cumN < 8*0` would be false for every cell and the whole grid
    // would split unconditionally each pass.)
    if (minCount <= 0.0f || c[PG_CUM_COUNT] < COUNT_HATCH_MULT * minCount) {
        // Scale-invariant spatial-contrast criterion: |centroid|^2 = sum S_a^2 / W^2,
        // accepted only if it clears BOTH the absolute threshold and the
        // nEff-based noise floor.
        float sumW = 0.0f;
        for (uint32_t lk = 0; lk < PG_NUM_LOBES; lk++)
            sumW += c[PG_CUMS_BASE + lk * PG_SUM_STRIDE + 3];   // per-lobe weight sum W
        if (!(sumW > 1e-8f)) return;
        float sx = c[PG_CUM_SR_X], sy = c[PG_CUM_SR_Y], sz = c[PG_CUM_SR_Z];
        float contrast = (sx * sx + sy * sy + sz * sz) / (sumW * sumW);   // |centroid|^2
        float nEff = (sumW * sumW) / fmaxf(c[PG_CUM_SW2], 1e-12f);
        float gate = fmaxf(contrastThreshold, CONTRAST_NOISE_LAMBDA / fmaxf(nEff, 1.0f));
        if (!(contrast > gate)) return;
    }

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
            // Maturity inherits like the count: a split parent had plenty of
            // evidence, and children starting at zero would drop the guide
            // confidence to 0 exactly where the grid just refined (guiding
            // pops off, then slowly re-ramps over the maturity window).
            d[PG_MATURITY] = c[PG_MATURITY] * 0.125f;
            // Spatial moments are measured relative to the PARENT's center and
            // half-size, so they don't carry into a child's own frame — reset
            // them. A zero centroid also blocks an immediate cascade re-split
            // before the child gathers its own evidence. Sum(w^2) resets with
            // them so the noise-floor nEff is computed from the same fresh
            // population as the centroid (inherited W makes the child's early
            // contrast an UNDERestimate — conservative, converges as the
            // inherited mass decays out of the EMA).
            d[PG_CUM_SR_X] = 0.0f; d[PG_CUM_SR_Y] = 0.0f; d[PG_CUM_SR_Z] = 0.0f;
            d[PG_INT_SR_X] = 0.0f; d[PG_INT_SR_Y] = 0.0f; d[PG_INT_SR_Z] = 0.0f;
            d[PG_CUM_SW2] = 0.0f;  d[PG_INT_SW2] = 0.0f;
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

void launchRefitCells(float* data, const uint64_t* cellKeys,
                      const uint32_t* cellCounter, uint32_t cellCapacity,
                      float emaDecay, float baseCellSize, uint32_t currentFrame,
                      cudaStream_t stream)
{
    if (!data || !cellKeys || !cellCounter || cellCapacity == 0) return;
    uint32_t blocks = (cellCapacity + BLOCK_SIZE - 1) / BLOCK_SIZE;
    refitCellsKernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
        data, reinterpret_cast<const unsigned long long*>(cellKeys),
        cellCounter, cellCapacity, emaDecay, baseCellSize,
        static_cast<float>(currentFrame));
}

void launchSubdivideCells(uint64_t* hashKeys, uint32_t* hashValues,
                          uint32_t hashTableSize, uint32_t hashShift,
                          uint64_t* cellKeys, uint32_t* cellCounter, uint32_t cellCapacity,
                          const uint32_t* countSnapshot,
                          float* data, uint32_t entryStride,
                          uint32_t maxLevel, float minCount, float contrastThreshold,
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
        table, countSnapshot, maxLevel, minCount, contrastThreshold,
        static_cast<float>(currentFrame));
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
