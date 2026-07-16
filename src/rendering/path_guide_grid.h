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
//   any cell with sufficient guided-vertex VISITS (traffic, level-normalized
//   — Mueller-style eligibility, brightness-neutral) AND genuine radiance
//   structure (per-axis half-cell log-radiance ratio). Children warm-start
//   from the parent. There is no coarsening: indices are stable and memory
//   is bounded by cell_capacity.
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
#include "path_guide_kernels.h"   // PG_SUBDIV_STAT_* layout for pollSubdivStats

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

    // Subdivision = VISIT sufficiency x RADIANCE STRUCTURE (both required):
    //
    //  - subdivide_count_threshold is the VISIT gate at start_level, in
    //    guided-vertex visits (fast-EMA units; visits are subsampled by
    //    PG_TRAIN_PROB in raygen, currently 1/2). Visits are counted with NO
    //    radiance gate, so eligibility follows camera/path traffic, not
    //    brightness — the old radiance-gated deposit count only ever
    //    admitted bright cells, which was the persistent root cause of
    //    brightness-correlated refinement no trigger criterion could fix.
    //    The kernel halves the gate per axis per level below start_level
    //    (visits/cell fall ~4x per level for 2D surface traffic), floored
    //    at 256 so the half-cell statistics stay trustworthy.
    //
    //  - subdivide_contrast_threshold is the STRUCTURE trigger: the largest
    //    per-axis |log ratio| of conditional mean log1p(radiance) between
    //    the cell's two halves (exact positions, eps-floored means, minimum
    //    32 visits per half). ~0.7 separates real edges (hard lit/unlit
    //    boundaries measure 1.7+) from smooth inverse-square falloff
    //    (~0.26 at 4x across a cell) with a wide margin. Density-,
    //    geometry-, and importance-sampling-invariant — the reasons every
    //    weighted-centroid contrast variant failed are documented in
    //    path_guide_cell_layout.h.
    float subdivide_count_threshold = 2048.0f;      // visit gate at start_level
    float subdivide_contrast_threshold = 0.7f;      // half-cell log-radiance ratio

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

    // Insert children of eligible cells (visit gate x structure test).
    // Render stream for ordering against optixLaunch. Cheap and idempotent.
    // Each pass also fills the subdivision stats buffer (split counts +
    // level histogram) and kicks an async readback tagged with passId —
    // unless a previous readback is still in flight, in which case the pass
    // runs without stats (at most one readback outstanding).
    void runSubdivisionPass(uint32_t currentFrame, uint32_t passId,
                            cudaStream_t stream);

    // Harvest the stats of the most recently COMPLETED subdivision pass.
    // Non-blocking; returns true at most once per pass (layout:
    // PG_SUBDIV_STAT_* in path_guide_kernels.h). outPassId receives the
    // passId given to the pass the stats belong to.
    bool pollSubdivStats(uint32_t out[/*PG_SUBDIV_STATS_SIZE*/],
                         uint32_t* outPassId = nullptr);

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

    // ── Subdivision-pass statistics (async readback, console diagnostics) ──
    uint32_t* m_subdivStats = nullptr;        // device, PG_SUBDIV_STATS_SIZE
    uint32_t* m_pinnedSubdivStats = nullptr;  // pinned host copy
    cudaEvent_t m_subdivStatsEvent = nullptr;
    bool m_subdivStatsInFlight = false;
    uint32_t m_subdivStatsPassId = 0;         // passId of the in-flight readback

    // ── Host debug mirror (refreshHostMirror) ──
    std::vector<uint64_t> m_hostKeys;                    // [count] packed keys
    std::unordered_map<uint64_t, uint32_t> m_hostKeyToIndex;
};

} // namespace spectra
