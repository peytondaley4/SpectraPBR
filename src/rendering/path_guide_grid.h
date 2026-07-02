#pragma once

//------------------------------------------------------------------------------
// Sparse Multi-Resolution Path Guide Grid — device-resident allocation
//
// === Architecture ===
//
// Spatial: sparse voxel grid keyed by (level << 48 | morton) in ONE persistent
//   device hash table (path_guide_hash_device.h). Shaders allocate cells on
//   first touch with atomicCAS + a bump allocator — there is no staging
//   buffer, no async readback, no CPU merge, and no double-buffered swap.
//   Cell indices are stable for the lifetime of the grid.
// Directional: a 4-lobe von Mises–Fisher MIXTURE per cell (layout shared via
//   optix_programs/path_guide_cell_layout.h). Deposits hard-assign to the
//   most responsible lobe (stepwise EM, Ruppert et al. 2020 flavored).
// Training: online via GPU atomicAdd of importance-weighted (Li/p) deposits
//   into per-lobe INTERVAL sums (raygen, backward pass over the path).
// Fitting: ON DEVICE — the refit kernel periodically folds interval sums into
//   EMA cumulative sums and refits each lobe in place (path_guide_kernels.cu).
// Refinement: ON DEVICE — the subdivision kernel inserts the 8 children of
//   any cell whose EMA deposit count crossed a threshold (PPG-style
//   sample-count criterion). Children warm-start from the parent. There is no
//   coarsening: indices are stable and memory is bounded by cell_capacity.
// Lookup: top-down — probe the start level, descend while a finer cell
//   containing the position exists (per_level_scale is fixed at 2, so finer
//   cells are always proper octree children).
//
// The CPU keeps only debug/UI mirrors: a non-blocking 4-byte poll of the
// allocation counter for the stats panel, and an on-demand synchronous
// readback of the cell-key list for the wireframe visualization and the
// cell inspector.
//
// References:
//   - Müller et al., "Practical Path Guiding for Efficient Light-Transport
//     Simulation", EGSR 2017 (Computer Graphics Forum).
//   - Banerjee et al. / Sra: vMF kappa approximation from mean resultant length.
//   - von Mises–Fisher (vMF): C3(κ) exp(κ μ·ω); Wood/Ulrich sampling.
//
// Cell data layout: see optix_programs/path_guide_cell_layout.h (the single
// source of truth, shared with the device code).
//------------------------------------------------------------------------------

#include <cstdint>
#include <cuda_runtime.h>
#include <unordered_map>
#include <vector>
#include "path_guide_cell_layout.h"

namespace spectra {

// Alias used by application config setup
constexpr uint32_t PATH_GUIDE_ENTRY_STRIDE_DEFAULT = PG_ENTRY_STRIDE;

//------------------------------------------------------------------------------
// Configuration
//------------------------------------------------------------------------------
struct PathGuideGridConfig {
    uint32_t num_levels = 8;          // bounds the level-resolution table
    uint32_t base_resolution = 16;
    float per_level_scale = 2.0f;     // must stay 2.0 (octree child derivation)
    uint32_t entry_stride = PATH_GUIDE_ENTRY_STRIDE_DEFAULT;
    float bounds_min[3] = { -10.0f, -10.0f, -10.0f };
    float bounds_max[3] = {  10.0f,  10.0f,  10.0f };

    // Max live cells. Data memory = cell_capacity * entry_stride * 4 bytes;
    // the hash table is sized to 2x capacity (<= 50% load).
    uint32_t cell_capacity = 1u << 19;

    uint32_t start_level = 2;              // Base level allocated on first touch
    uint32_t min_level = 1;                // Retired from the hot path (UI compat)
    uint32_t max_level = 6;                // Finest allowed level (subdivision cap)
    uint32_t refine_interval_frames = 30;  // Subdivision kernel cadence

    // Subdivision is spatial-contrast driven, not raw sample count: a cell
    // splits only when the radiance is spatially OFF-CENTER inside it, i.e. it
    // straddles a barrier (caustic edge, shadow line). The trigger is the
    // scale-invariant |centroid|^2 = sum_a S_a^2 / W^2 (S_a = EMA Sum(w*rel_a),
    // W = EMA weight sum, rel in [-1,1]) exceeding subdivide_contrast_threshold,
    // gated by at least subdivide_count_threshold deposits so the centroid is
    // meaningful. Because |centroid|^2 is independent of brightness, a uniform
    // cell (bright OR dark) is never split and the grid no longer subdivides
    // uniformly under flat primary visibility. Counts are in DEPOSIT units —
    // deposits are subsampled 1/4 in raygen (PG_TRAIN_PROB). The contrast
    // threshold is dimensionless in [0,3]; ~0.1 catches sharp edges. The
    // kernel ALSO requires contrast to clear a per-cell noise floor ~4/nEff,
    // nEff = W^2/Sum(w^2) (Kish effective sample size): the centroid's noise
    // is governed by nEff, not deposit count — one heavy Li/pdf firefly can
    // collapse nEff to a handful while the count stays in the thousands, and
    // would otherwise fake spatial structure in a uniform cell. Cells past 8x
    // the count gate split regardless of contrast (first-moment centroids are
    // blind to even-symmetric variation; costs at most one surplus level on
    // hot uniform cells). Both thresholds are empirical — tune.
    //
    // The count gate is also a WELL-FED gate: a cell must be well-sampled
    // before it splits, because (a) the centroid is only trustworthy with
    // enough deposits and (b) each of the 8 children inherits just 1/8 of the
    // parent's cumulative evidence (incl. the slow-decayed PG_MATURITY that
    // drives the guide confidence ramp) — split too early and the children
    // are born under-trained and stay noisy until they re-accumulate. At
    // 2048, children start at ~256, comfortably trained. (Fast-EMA window is
    // ~1/(1-decay) ≈ 6.7 refits, and cells reached the old 8192 rule, so 2048
    // is well within reach.)
    float subdivide_count_threshold = 2048.0f;      // min-sample / maturity gate
    float subdivide_contrast_threshold = 0.12f;     // |centroid|^2 trigger, [0,3]

    // Device refit: EMA decay applied to cumulative sums per refit. The
    // effective averaging window is refit_interval / (1 - decay) frames.
    float refit_ema_decay = 0.85f;
};

//------------------------------------------------------------------------------
// GPU descriptor (copied into launch params each frame)
//------------------------------------------------------------------------------
struct SparsePathGuideDescriptor {
    float* data = nullptr;                  // entry_stride floats per cell
    uint64_t* hash_keys = nullptr;          // (level<<48 | morton), empty = ~0 (device CAS target)
    uint32_t* hash_values = nullptr;        // cell index or sentinel
    uint32_t hash_table_size = 0;           // power of 2
    uint32_t hash_shift = 64;               // 64 - log2(hash_table_size)
    uint64_t* cell_keys = nullptr;          // packed key per allocated cell
    uint32_t* cell_counter = nullptr;       // bump allocator (1 element)
    uint32_t cell_capacity = 0;
    uint32_t entry_stride = 0;
    uint32_t num_levels = 0;
    uint32_t base_resolution = 0;
    float per_level_scale = 2.0f;
    float bounds_min[3] = {};
    float bounds_max[3] = {};
};

//------------------------------------------------------------------------------
// PathGuideGrid: device-resident cell table; CPU keeps debug/UI mirrors only
//------------------------------------------------------------------------------
class PathGuideGrid {
public:
    PathGuideGrid() = default;
    ~PathGuideGrid();

    PathGuideGrid(const PathGuideGrid&) = delete;
    PathGuideGrid& operator=(const PathGuideGrid&) = delete;

    bool init(const PathGuideGridConfig& config);
    void shutdown();

    // Zero all cell data (lobes + stats). The cell SET persists — allocated
    // cells keep their indices and retrain from scratch.
    void clear(cudaStream_t stream = nullptr);

    SparsePathGuideDescriptor getDescriptor() const;

    // Fold interval sums into EMA cumulative sums and refit mu/kappa in
    // place, on the render stream. Cheap — call every few frames.
    void refitLobes(uint32_t currentFrame, cudaStream_t stream);

    // Insert children of well-fed cells (EMA count >= threshold). Render
    // stream for ordering against optixLaunch. Cheap and idempotent.
    void runSubdivisionPass(uint32_t currentFrame, cudaStream_t stream);

    // Non-blocking cell-count poll for the UI: each call first harvests a
    // previously completed copy, then kicks off a fresh 4-byte async copy.
    void requestCellCountAsync(cudaStream_t stream);
    uint32_t lastCellCount() const { return m_lastCellCount; }

    // Synchronous readback of the allocation counter + cell key list.
    // Debug paths only (wireframe viz refresh, inspector click).
    // Returns true when at least one cell exists.
    bool refreshHostMirror();

    uint32_t getNumLevels() const { return m_config.num_levels; }
    uint32_t getEntryStride() const { return m_entryStride; }
    uint32_t getTotalCells() const { return m_lastCellCount; }
    bool isInitialized() const { return m_data != nullptr; }
    bool hasSparseData() const { return m_lastCellCount > 0; }

    const PathGuideGridConfig& getConfig() const { return m_config; }

    // Generate wireframe edge vertices from the host mirror (call
    // refreshHostMirror() first to update it).
    std::vector<float> generateEdgeVertices(uint32_t level) const;
    std::vector<float> generateEdgeVerticesAllLevels() const;

    // Cell inspection: look up the deepest cell containing a world-space
    // position. Refreshes the host mirror and reads back just that cell
    // (synchronous — UI click only).
    struct CellInspectionResult {
        bool found = false;
        uint32_t level = 0;
        uint32_t ix = 0, iy = 0, iz = 0;
        float data[PG_ENTRY_STRIDE] = {};
        float aabbMin[3] = {};
        float aabbMax[3] = {};
    };
    CellInspectionResult inspectCellAtPosition(float px, float py, float pz);

    uint32_t getStartLevel() const { return m_config.start_level; }
    uint32_t getMinLevel() const { return m_config.min_level; }
    uint32_t getMaxLevel() const { return m_config.max_level; }

private:
    static void mortonDecode(uint64_t morton, uint32_t& ix, uint32_t& iy, uint32_t& iz);
    void appendCellEdges(std::vector<float>& vertices, uint32_t level, uint64_t morton) const;

    PathGuideGridConfig m_config;
    uint32_t m_entryStride = 0;

    // Precomputed level resolutions: floor(base_res * scale^level)
    static constexpr uint32_t MAX_LEVELS = 16;
    uint32_t m_levelResolutions[MAX_LEVELS] = {};

    // ── Device cell table ──
    uint64_t* m_hashKeys = nullptr;
    uint32_t* m_hashValues = nullptr;
    uint32_t  m_hashTableSize = 0;
    uint32_t  m_hashShift = 64;
    uint64_t* m_cellKeys = nullptr;     // [capacity] packed key per cell
    uint32_t* m_cellCounter = nullptr;  // bump allocator
    uint32_t* m_counterSnapshot = nullptr;  // pre-subdivision copy (cascade guard)
    float*    m_data = nullptr;         // [capacity * entry_stride]

    // ── UI count poll ──
    uint32_t* m_pinnedCount = nullptr;  // pinned host memory (4 bytes)
    cudaEvent_t m_countEvent = nullptr;
    bool m_countInFlight = false;
    uint32_t m_lastCellCount = 0;

    // ── Host debug mirror (refreshHostMirror) ──
    std::vector<uint64_t> m_hostKeys;                    // [count] packed keys
    std::unordered_map<uint64_t, uint32_t> m_hostKeyToIndex;
};

} // namespace spectra
