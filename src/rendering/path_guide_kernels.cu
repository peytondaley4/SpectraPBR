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

    // Fold the deposit-position moments (centroid + spread, jittered basis)
    // on the same decay footing as their SMW normalizer, and the visit /
    // half-cell radiance stats (exact basis) on the same footing as each
    // other — every ratio the consumers form is a pair of identically
    // decayed populations.
    c[PG_CUM_SR_X] = emaDecay * c[PG_CUM_SR_X] + c[PG_INT_SR_X];
    c[PG_CUM_SR_Y] = emaDecay * c[PG_CUM_SR_Y] + c[PG_INT_SR_Y];
    c[PG_CUM_SR_Z] = emaDecay * c[PG_CUM_SR_Z] + c[PG_INT_SR_Z];
    c[PG_CUM_SRR]  = emaDecay * c[PG_CUM_SRR] + c[PG_INT_SRR];
    c[PG_CUM_SMW]  = emaDecay * c[PG_CUM_SMW] + c[PG_INT_SMW];
    c[PG_CUM_VISITS] = emaDecay * c[PG_CUM_VISITS] + c[PG_INT_VISITS];
    c[PG_CUM_SL]   = emaDecay * c[PG_CUM_SL] + c[PG_INT_SL];
    c[PG_CUM_HC_X] = emaDecay * c[PG_CUM_HC_X] + c[PG_INT_HC_X];
    c[PG_CUM_HC_Y] = emaDecay * c[PG_CUM_HC_Y] + c[PG_INT_HC_Y];
    c[PG_CUM_HC_Z] = emaDecay * c[PG_CUM_HC_Z] + c[PG_INT_HC_Z];
    c[PG_CUM_HL_X] = emaDecay * c[PG_CUM_HL_X] + c[PG_INT_HL_X];
    c[PG_CUM_HL_Y] = emaDecay * c[PG_CUM_HL_Y] + c[PG_INT_HL_Y];
    c[PG_CUM_HL_Z] = emaDecay * c[PG_CUM_HL_Z] + c[PG_INT_HL_Z];
    c[PG_INT_SR_X] = 0.0f;
    c[PG_INT_SR_Y] = 0.0f;
    c[PG_INT_SR_Z] = 0.0f;
    c[PG_INT_SRR]  = 0.0f;
    c[PG_INT_SMW]  = 0.0f;
    c[PG_INT_VISITS] = 0.0f;
    c[PG_INT_SL]   = 0.0f;
    c[PG_INT_HC_X] = 0.0f;
    c[PG_INT_HC_Y] = 0.0f;
    c[PG_INT_HC_Z] = 0.0f;
    c[PG_INT_HL_X] = 0.0f;
    c[PG_INT_HL_Y] = 0.0f;
    c[PG_INT_HL_Z] = 0.0f;

    // Deposit spread about the mw-weighted centroid (rel units, [0..~1.7]):
    // the lobe's true positional uncertainty for the geometric kappa cap.
    // With the parallax pivot at the measured centroid (raygen), the
    // borrowing error is the spread of the deposits around that pivot — NOT
    // the whole cell size — so a compact light pool inside a coarse cell can
    // still earn a sharp, correctly-aimed lobe (this is what decouples guide
    // quality from refinement level; the old cellSize-denominated cap forced
    // coarse cells near lights to floodlight lobes, which rendered every
    // refinement-level boundary as a convergence cliff). Floored at 0.25:
    // an ultra-tight measured spread is more likely undersampling than a
    // point source, and the pivot itself carries estimation error.
    float spreadRel = 0.25f;
    float cellSmw = c[PG_CUM_SMW];   // log-tamed weight mass; resets on split
    if (cellSmw > 1e-6f) {
        float invSmw = 1.0f / cellSmw;
        float cwx = c[PG_CUM_SR_X] * invSmw;
        float cwy = c[PG_CUM_SR_Y] * invSmw;
        float cwz = c[PG_CUM_SR_Z] * invSmw;
        float spread2 = c[PG_CUM_SRR] * invSmw - (cwx * cwx + cwy * cwy + cwz * cwz);
        spreadRel = fmaxf(sqrtf(fmaxf(spread2, 0.0f)), 0.25f);
    }

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

                    // GEOMETRY-AWARE sharpness ceiling, denominated in the
                    // measured DEPOSIT SPREAD about the parallax pivot — not
                    // the cell size. The lobe is consumed up to ~1 cell from
                    // where it was fit (box-filter jitter), but the parallax
                    // reprojection pivots at the measured centroid, so the
                    // residual positional error is the spread of the deposits
                    // around that pivot: sigma_pos ~ spreadRel * halfCell.
                    // Angular error ~ sigma_pos/meanDist, and vMF std ~
                    // 1/sqrt(kappa), so kappa <= (meanDist/sigma_pos)^2.
                    // The old cellSize denominator assumed the worst pivot
                    // error (half a cell); with the centroid pivot that
                    // over-penalized every coarse cell near a light by up to
                    // 16x, hard-capping them to floodlight lobes and turning
                    // refinement-level boundaries into convergence cliffs.
                    // Evidence gate unchanged: maturity unlocks 300 -> 2000.
                    float kappaMax = (maturity >= 64.0f) ? 2000.0f : 300.0f;
                    // Fresh-evidence damp on the 2000 unlock ONLY: maturity
                    // is inherited IN FULL on split (the mixture is
                    // verbatim), but the cell's own fitted mass starts at
                    // zero — 1-2 fresh aligned deposits give rbar ~= 1 and a
                    // raw kappa in the tens of thousands, which inherited
                    // maturity would otherwise wave straight through to
                    // 2000. Denominate the EXTRA sharpness in the cell's
                    // LOG-TAMED weight mass (CUM_SMW: <= ~6.9 per deposit,
                    // resets on split): ~40 units (~6+ deposits) for the
                    // full 2000. FLOORED AT 300 — the pre-unlock trust
                    // ceiling — because a fresh child re-fitting its
                    // inherited (verbatim, trusted) mixture with SMW = 0
                    // must not have its kappa crushed to zero: that
                    // de-eligibles every lobe and switches guiding OFF in
                    // exactly the cells the grid just refined, stamping a
                    // noisy rectangle into the accumulation for every late
                    // split (the visible "notches").
                    kappaMax = fminf(kappaMax, fmaxf(50.0f * cellSmw, 300.0f));
                    if (meanDist > 1e-4f) {
                        float sigmaPos = fmaxf(spreadRel * 0.5f * cellSize, 1e-6f);
                        float ratio = meanDist / sigmaPos;
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
// Subdivision: insert the 8 children of a cell when BOTH hold:
//  1. VISIT sufficiency (level-normalized): enough guided-vertex traffic —
//     counted with NO radiance gate — that the statistics below are
//     trustworthy. Traffic, not brightness, buys eligibility: the previous
//     radiance-gated deposit count handed every split decision to bright
//     cells before any criterion ran, which was the persistent root cause
//     of brightness-correlated refinement across four criterion iterations.
//  2. RADIANCE STRUCTURE (per-axis half-cell log-radiance ratio): the
//     conditional mean log1p(radiance) differs between the two halves of
//     the cell along some axis. Density-invariant, geometry-invariant,
//     importance-sampling-invariant, measured at exact (unjittered)
//     positions — see the criterion comment in the kernel body and the
//     layout header for why every weighted-centroid contrast variant
//     failed structurally.
//
// Once children exist, the top-down lookup targets them and the parent's
// EXACT-cell visit stats stop accumulating (its EMA evidence decays within
// a few refits); the box-filter jitter may still deposit lobe training into
// it, which is intentional cross-face splatting. Re-examining subdivided
// parents costs only hash probes (pgTableInsert is idempotent).
//------------------------------------------------------------------------------
__global__ void subdivideCellsKernel(PathGuideTableDevice table,
                                     const uint32_t* countSnapshot,
                                     uint32_t maxLevel, uint32_t startLevel,
                                     float minVisits, float hlrThreshold,
                                     float baseCellSize,
                                     float currentFrame, uint32_t* stats)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Bound by the pre-launch snapshot, NOT the live counter: children
    // inserted by this pass must not be processed in the same pass (their
    // warm-started evidence could cascade-subdivide, and their payload may
    // still be mid-write by the inserting thread).
    uint32_t totalCells = *countSnapshot;
    if (totalCells > table.cell_capacity) totalCells = table.cell_capacity;
    if (idx >= totalCells) return;

    unsigned long long key = table.cell_keys[idx];
    uint32_t level = (uint32_t)(key >> 48);
    if (stats) {
        uint32_t l = level < 15u ? level : 15u;
        atomicAdd(&stats[PG_SUBDIV_STAT_LEVEL0 + l], 1u);  // live-cell level histogram
    }
    if (level >= maxLevel) return;

    float* c = table.data + (size_t)idx * table.entry_stride;

    // ── Gate 1: statistical sufficiency, in VISITS (traffic), level-
    // normalized. Visits are counted for every guided vertex with NO
    // radiance gate, so eligibility follows where the camera's paths
    // actually are — a dim-but-visible cell is exactly as eligible as a
    // bright one (the old radiance-gated deposit count handed every split
    // decision to bright cells before any criterion even ran; that was the
    // saga-long driver of brightness-correlated refinement). The threshold
    // halves per axis per level (visits per cell fall ~4x per level for
    // 2D surface traffic). NOTE the semantics: CUM_VISITS is a fast EMA
    // (steady state = 6.67x the per-refit-interval rate), so these are
    // sustained-RATE thresholds, not cumulative counts — a cell below the
    // rate never qualifies no matter how long the camera stares. The floor
    // is min(256, minVisits) so a small configured threshold cannot make
    // deep levels HARDER to reach than the start level; minVisits <= 0
    // disables subdivision entirely.
    float gateVisits = minVisits;
    if (level > startLevel) {
        gateVisits = fmaxf(minVisits * exp2f(-2.0f * (float)(level - startLevel)),
                           fminf(256.0f, minVisits));
    }
    float visits = c[PG_CUM_VISITS];
    if (minVisits <= 0.0f || visits < gateVisits) return;
    if (stats) atomicAdd(&stats[PG_SUBDIV_STAT_ELIGIBLE], 1u);   // gate-passed

    // ── Gate 2: radiance structure — per-axis half-cell statistics on the
    // conditional MEAN log1p(radiance) of the two half-cells (negative half
    // derived by subtraction from the totals). Why this statistic survives
    // where every weighted-centroid variant failed (established
    // numerically, 2026-07):
    //  - conditional MEANS are invariant to deposit DENSITY, so the
    //    radiance-correlated visit density that poisoned the centroid
    //    criteria (bright halves get more samples) cancels by construction;
    //  - both halves are measured directly — a surface cutting the cell
    //    off-center shifts both means identically (geometry-invariant);
    //  - the statistic is built on RAW radiance, not Li/pdf, so the guide's
    //    own convergence does not flatten it away;
    //  - measured at EXACT vertex positions (raygen packs the half-cell
    //    signs before jittering), so the box filter cannot smear the edge.
    // TWO forms are tested, because log1p changes character with exposure:
    //  - RATIO |log((posMean+eps)/(negMean+eps))|: for radiance <~ 1,
    //    log1p(L) ~= L, so this is the scale-invariant log radiance ratio —
    //    the calibrated regime (edges 1.7+, smooth 4x falloff ~0.26). The
    //    eps floor bounds the dark-half ratio; it also means scenes whose
    //    mean incident luminance sits below ~0.02 cannot split on this form.
    //  - DIFFERENCE |posMean - negMean| vs 1.7x the threshold: for radiance
    //    >> 1, log1p(L) ~= log(L), so the DIFFERENCE of means is the log of
    //    the geometric-mean ratio — exactly scale-invariant where the ratio
    //    form's double-log compression would go blind (a 20:1 edge measures
    //    ~3.0 at ANY bright exposure; a smooth 4x gradient <= ~0.7 < 1.19).
    // Split when either form fires on any axis.
    {
        const float EPS = 0.02f;       // mean floor: bounds the dark-half ratio
        const float HALF_FLOOR = 32.0f; // min visits per half for a valid mean
        float sl = c[PG_CUM_SL];
        bool structure = false;
        #pragma unroll
        for (int a = 0; a < 3; a++) {
            float posC = c[PG_CUM_HC_X + a];
            float negC = visits - posC;
            if (posC < HALF_FLOOR || negC < HALF_FLOOR) continue;
            float posL = c[PG_CUM_HL_X + a];
            float negL = fmaxf(sl - posL, 0.0f);
            float posMean = posL / posC;
            float negMean = negL / negC;
            float ratio = fabsf(logf((posMean + EPS) / (negMean + EPS)));
            float diff  = fabsf(posMean - negMean);
            if (ratio > hlrThreshold || diff > 1.7f * hlrThreshold) {
                structure = true;
                break;
            }
        }
        // ── Gate 2b: RESOLUTION-LIMITED test (Ruppert-style adaptivity).
        // The half-cell test above only sees scalar radiance EDGES; a
        // smooth-but-NEAR illumination field (the falloff band around a
        // light pool) never trips it, yet its directional distribution
        // varies too fast for a coarse cell: the spread-denominated kappa
        // cap pins the lobes to floodlights (observed: kappa ~9 in a base
        // cell beside a pool whose fine neighbors earn 2000), and the
        // visible result is "convergence tracks grid density". Split when a
        // trained lobe's FITTED concentration demand (Banerjee/Sra from the
        // cum sums, same formula as the refit) exceeds the cell's
        // achievable cap severalfold — the guide is provably limited by
        // spatial resolution, not by evidence. Brightness-neutral (rbar is
        // a normalized direction statistic) and self-limiting: each split
        // quadruples the cap, so refinement stops as soon as the cap clears
        // the demand (bounded by the physical source size, max_level, and
        // the visit gate above).
        if (!structure) {
            float smw = c[PG_CUM_SMW];
            if (smw > 1e-6f) {
                float invSmw = 1.0f / smw;
                float cwx = c[PG_CUM_SR_X] * invSmw;
                float cwy = c[PG_CUM_SR_Y] * invSmw;
                float cwz = c[PG_CUM_SR_Z] * invSmw;
                float spread2 = c[PG_CUM_SRR] * invSmw - (cwx * cwx + cwy * cwy + cwz * cwz);
                float spreadRel = fmaxf(sqrtf(fmaxf(spread2, 0.0f)), 0.25f);
                float cellSize = baseCellSize * exp2f(-(float)level);
                float sigmaPos = fmaxf(spreadRel * 0.5f * cellSize, 1e-6f);
                for (int k = 0; k < PG_NUM_LOBES; k++) {
                    const float* cumS = c + PG_CUMS_BASE + k * PG_SUM_STRIDE;
                    float w = cumS[3];
                    if (w < 32.0f) continue;   // lobe must be well-evidenced
                    float len = sqrtf(cumS[0] * cumS[0] + cumS[1] * cumS[1] + cumS[2] * cumS[2]);
                    float rbar = fminf(len / w, 0.99999f);
                    float implied = rbar * (3.0f - rbar * rbar) / fmaxf(1.0f - rbar * rbar, 1e-4f);
                    float meanDist = cumS[PG_S_DIST] / w;
                    if (meanDist < 1e-4f) continue;
                    float ratio = meanDist / sigmaPos;
                    // Achievable ceiling, floored at 8 like the refit's cap:
                    // below the floor the refit grants min(implied, 8)
                    // regardless of ratio, so the guide is not resolution-
                    // limited there. (The inspector heuristic in
                    // application.cpp mirrors this formula.)
                    float cap = fmaxf(ratio * ratio, 8.0f);
                    // cap >= 500: already sharp enough that refinement buys
                    // little; demand > 4x cap: real headroom, with hysteresis.
                    if (cap < 500.0f && implied > 4.0f * cap) {
                        structure = true;
                        break;
                    }
                }
            }
        }
        if (!structure) {
            if (stats) atomicAdd(&stats[PG_SUBDIV_STAT_NOSTRUCT], 1u);   // structure test failed
            return;
        }
    }

    unsigned long long morton = key & ((1ull << 48) - 1);
    if (stats) atomicAdd(&stats[PG_SUBDIV_STAT_SPLIT], 1u);   // split

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
            // Maturity inherits IN FULL, not 1/8: the child's mixture is the
            // parent's verbatim (copied above), so the guide's *confidence*
            // in it did not drop — only the deposit-rate semantics did. The
            // old 1/8 inheritance cut the confidence ramp (and the kappa
            // evidence unlock) 8x at exactly the cells the grid just deemed
            // important, drawing cell-shaped confidence rings around every
            // split.
            d[PG_MATURITY] = c[PG_MATURITY];
            // Spatial moments and the visit/half-cell radiance stats are
            // measured relative to the PARENT's frame — they don't carry
            // into a child's own frame. Reset; children earn their own
            // (fresh-population ratios stay consistent because every
            // numerator/denominator pair resets together).
            d[PG_CUM_SR_X] = 0.0f; d[PG_CUM_SR_Y] = 0.0f; d[PG_CUM_SR_Z] = 0.0f;
            d[PG_INT_SR_X] = 0.0f; d[PG_INT_SR_Y] = 0.0f; d[PG_INT_SR_Z] = 0.0f;
            d[PG_CUM_SRR] = 0.0f;  d[PG_INT_SRR] = 0.0f;
            d[PG_CUM_SMW] = 0.0f;  d[PG_INT_SMW] = 0.0f;
            d[PG_CUM_VISITS] = 0.0f; d[PG_INT_VISITS] = 0.0f;
            d[PG_CUM_SL] = 0.0f;   d[PG_INT_SL] = 0.0f;
            d[PG_CUM_HC_X] = 0.0f; d[PG_CUM_HC_Y] = 0.0f; d[PG_CUM_HC_Z] = 0.0f;
            d[PG_INT_HC_X] = 0.0f; d[PG_INT_HC_Y] = 0.0f; d[PG_INT_HC_Z] = 0.0f;
            d[PG_CUM_HL_X] = 0.0f; d[PG_CUM_HL_Y] = 0.0f; d[PG_CUM_HL_Z] = 0.0f;
            d[PG_INT_HL_X] = 0.0f; d[PG_INT_HL_Y] = 0.0f; d[PG_INT_HL_Z] = 0.0f;
            d[PG_LAST_HIT_FRAME] = currentFrame;
            if (stats) atomicAdd(&stats[PG_SUBDIV_STAT_CHILDREN], 1u);
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
                          uint32_t maxLevel, uint32_t startLevel,
                          float minVisits, float hlrThreshold,
                          float baseCellSize,
                          uint32_t currentFrame, uint32_t* stats,
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
        table, countSnapshot, maxLevel, startLevel, minVisits, hlrThreshold,
        baseCellSize, static_cast<float>(currentFrame), stats);
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
