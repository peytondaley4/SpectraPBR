#pragma once

//------------------------------------------------------------------------------
// Sparse Multi-Resolution Path Guide Grid (collision-free, GPU-friendly)
//
// === Architecture ===
//
// Spatial: sparse voxel grid with Morton-coded cells and O(1) hash lookup.
// Directional: one von Mises–Fisher (vMF) lobe per cell (mu vector + kappa).
// Training: online via GPU atomicAdd of importance-weighted (Li/p) deposits
//   into per-cell INTERVAL sums (raygen, backward pass over the path).
// Fitting: ON DEVICE — a refit kernel periodically folds interval sums into
//   EMA cumulative sums and refits mu/kappa in place (path_guide_kernels.cu).
//   No readback, no CPU fit, no upload, no training-loss window.
// Structure: the CPU only manages the CELL SET. Staging entries (appended by
//   raygen on cell-lookup miss) are read back asynchronously; when new cells
//   appear, the CPU builds the new morton/hash arrays, uploads them to the
//   inactive grid, and a device gather kernel re-layouts the LIVE cell data
//   at swap time.
// Refinement: adaptive subdivision (poor fit) / coarsening (stale cells),
//   driven by the cumulative statistics; children warm-start with the
//   parent's lobe.
//
// === Why one lobe per cell ===
//
// The aggregate sums (sumX/Y/Z/W) are sufficient statistics for a SINGLE vMF
// lobe; they cannot separate the modes of a bimodal distribution, so the old
// second lobe (fit from interval-vs-cumulative divergence) never represented
// real bimodality — it only reacted to temporal lighting changes. Spatial
// subdivision plus the BSDF MIS leg covers multimodal regions; per-lobe
// sufficient statistics with streaming soft assignment can reintroduce
// mixtures later.
//
// References:
//   - Müller et al., "Practical Path Guiding for Efficient Light-Transport
//     Simulation", EGSR 2017 (Computer Graphics Forum).
//   - Banerjee et al. / Sra: vMF kappa approximation from mean resultant length.
//   - von Mises–Fisher (vMF): C3(κ) exp(κ μ·ω); Wood/Ulrich sampling.
//
// === Cell Data Layout (16 floats = 64 bytes) ===
//
//   [0-2]:  mu (unit mean direction)        — written by refit kernel
//   [3]:    kappa (<= 0: no valid lobe)     — written by refit kernel
//   [4-7]:  interval sums (x, y, z, w)      — atomicAdd by shaders
//   [8-11]: cumulative sums (EMA lifetime)  — owned by refit kernel
//   [12]:   lastHitFrame (atomicMax)
//   [13]:   interval deposit count
//   [14]:   cumulative deposit count (EMA)
//   [15]:   reserved
//------------------------------------------------------------------------------

#include <cstdint>
#include <cuda_runtime.h>
#include <vector>

namespace spectra {

// ─── Named cell data offsets (must match path_guide_grid_device.h exactly) ──
constexpr uint32_t PG_MU_X           = 0;
constexpr uint32_t PG_MU_Y           = 1;
constexpr uint32_t PG_MU_Z           = 2;
constexpr uint32_t PG_KAPPA          = 3;
constexpr uint32_t PG_INT_SUM_X      = 4;
constexpr uint32_t PG_INT_SUM_Y      = 5;
constexpr uint32_t PG_INT_SUM_Z      = 6;
constexpr uint32_t PG_INT_SUM_W      = 7;
constexpr uint32_t PG_CUM_SUM_X      = 8;
constexpr uint32_t PG_CUM_SUM_Y      = 9;
constexpr uint32_t PG_CUM_SUM_Z      = 10;
constexpr uint32_t PG_CUM_SUM_W      = 11;
constexpr uint32_t PG_LAST_HIT_FRAME = 12;
constexpr uint32_t PG_INT_COUNT      = 13;
constexpr uint32_t PG_CUM_COUNT      = 14;
constexpr uint32_t PG_RESERVED       = 15;
constexpr uint32_t PG_ENTRY_STRIDE   = 16;

// Alias used by application config setup
constexpr uint32_t PATH_GUIDE_ENTRY_STRIDE_DEFAULT = PG_ENTRY_STRIDE;

//------------------------------------------------------------------------------
// Configuration
//------------------------------------------------------------------------------
struct PathGuideGridConfig {
    uint32_t num_levels = 8;
    uint32_t base_resolution = 16;
    float per_level_scale = 2.0f;
    uint32_t entry_stride = PATH_GUIDE_ENTRY_STRIDE_DEFAULT;
    float bounds_min[3] = { -10.0f, -10.0f, -10.0f };
    float bounds_max[3] = {  10.0f,  10.0f,  10.0f };
    uint32_t staging_capacity = 1u << 19;  // 512K entries for cell seeding

    // Adaptive refinement settings
    uint32_t start_level = 2;              // Initial coarse level for new regions
    uint32_t min_level = 1;                // Minimum level (coarsest allowed)
    uint32_t max_level = 6;                // Maximum level (finest allowed)
    uint32_t refine_interval_frames = 30;  // Run refinement every N frames
    float subdivide_sample_threshold = 50.0f;    // (UI display heuristic)
    float subdivide_variance_threshold = 0.08f;  // (UI display heuristic)
    uint32_t coarsen_frames_threshold = 480;     // Frames without hits before coarsening

    // Device refit: EMA decay applied to cumulative sums per refit. The
    // effective averaging window is refit_interval / (1 - decay) frames.
    float refit_ema_decay = 0.85f;
};

//------------------------------------------------------------------------------
// GPU descriptor for sparse grid (copy to device / launch params)
//------------------------------------------------------------------------------
struct SparsePathGuideDescriptor {
    const uint64_t* morton_codes;   // Sorted per level (device)
    float* data;                    // entry_stride floats per cell (device)
    const uint32_t* level_offsets;  // [0 .. num_levels], level l -> [level_offsets[l], level_offsets[l+1])
    uint32_t num_levels;
    uint32_t entry_stride;
    uint32_t base_resolution;
    float per_level_scale;
    float bounds_min[3];
    float bounds_max[3];

    // Hash table for O(1) cell lookup
    const uint64_t* hash_keys = nullptr;
    const uint32_t* hash_values = nullptr;
    uint32_t hash_table_size = 0;
    uint32_t hash_shift = 0;
};

//------------------------------------------------------------------------------
// Staging for occupancy collection (raygen appends; build reads back)
//------------------------------------------------------------------------------
struct PathGuideStagingDescriptor {
    uint32_t* buffer;    // 4 × capacity: [level, ix, iy, iz] per entry (device)
    uint32_t* count;     // Atomic counter (device), 1 element
    uint32_t capacity;
};

//------------------------------------------------------------------------------
// PathGuideGrid: sparse buffers + staging; structure on CPU, data on device
//------------------------------------------------------------------------------
class PathGuideGrid {
public:
    PathGuideGrid() = default;
    ~PathGuideGrid();

    PathGuideGrid(const PathGuideGrid&) = delete;
    PathGuideGrid& operator=(const PathGuideGrid&) = delete;

    bool init(const PathGuideGridConfig& config);
    void shutdown();

    void clear(cudaStream_t stream = nullptr);  // Zero data (not morton/offsets)

    // Staging: reset the append counter (stream-ordered)
    void resetStagingCount(cudaStream_t stream = nullptr);

    SparsePathGuideDescriptor getDescriptor() const;
    SparsePathGuideDescriptor getRenderDescriptor() const;  // Always returns active render grid
    PathGuideStagingDescriptor getStagingDescriptor() const;

    // ── Device-side lobe refit (replaces the CPU fit round trip) ──
    // Folds interval sums into EMA cumulative sums and refits mu/kappa in
    // place, on the given (render) stream. Cheap — call every few frames.
    void refitLobes(uint32_t currentFrame, cudaStream_t stream);

    // ── Async structure pipeline (staging readback → CPU merge → gather) ──
    bool initAsync();
    void shutdownAsync();
    void beginAsyncReadback(cudaStream_t renderStream, uint32_t currentFrame);
    bool pollAsyncReadback();
    // Background-thread part: merges staging into the cell set and uploads
    // the new structure to the build grid. Returns false when the cell set
    // did not change (common steady-state case — no swap needed).
    bool finishBuildFromReadback();
    // Render-thread part: gathers live cell data into the new layout on the
    // render stream and switches the active grid.
    void swapGrids(cudaStream_t renderStream = nullptr);

    uint32_t getNumLevels() const { return m_numLevels; }
    uint32_t getEntryStride() const { return m_entryStride; }
    uint32_t getTotalCells() const { return m_totalCells; }
    const uint32_t* getLevelOffsets() const { return m_levelOffsets.data(); }
    bool isInitialized() const { return m_stagingBuffer != nullptr && m_levelOffsetsDevice != nullptr; }
    bool hasSparseData() const { return m_totalCells > 0; }

    // Get config for bounds information
    const PathGuideGridConfig& getConfig() const { return m_config; }

    // Generate wireframe edge vertices for visualization
    std::vector<float> generateEdgeVertices(uint32_t level) const;
    std::vector<float> generateEdgeVerticesAllLevels() const;

    // Adaptive refinement: subdivide poorly-fit cells, coarsen stale cells.
    // Does one bulk stats readback (sync on `stream`), then restructures via
    // the same upload + gather path as the async build. Returns true if the
    // grid structure changed.
    bool runRefinementPass(uint32_t currentFrame, cudaStream_t stream = nullptr);

    // Cell inspection: look up the cell containing a world-space position.
    // Reads back just that cell (64 bytes, synchronous — UI click only).
    struct CellInspectionResult {
        bool found = false;
        uint32_t level = 0;
        uint32_t ix = 0, iy = 0, iz = 0;
        float data[PG_ENTRY_STRIDE] = {};
        float aabbMin[3] = {};
        float aabbMax[3] = {};
    };
    CellInspectionResult inspectCellAtPosition(float px, float py, float pz) const;

    uint32_t getStartLevel() const { return m_config.start_level; }
    uint32_t getMinLevel() const { return m_config.min_level; }
    uint32_t getMaxLevel() const { return m_config.max_level; }

private:
    struct CellKey {
        uint32_t level;
        uint64_t morton;
        bool operator<(const CellKey& o) const {
            if (level != o.level) return level < o.level;
            return morton < o.morton;
        }
        bool operator==(const CellKey& o) const {
            return level == o.level && morton == o.morton;
        }
    };

    // Morton decode helper
    static void mortonDecode(uint64_t morton, uint32_t& ix, uint32_t& iy, uint32_t& iz);

    // Build per-level offsets + sorted morton array + old→new gather map from
    // a sorted, deduplicated cell-key list. Updates m_levelOffsets,
    // m_mortonCodesHost, m_totalCells. gatherMap entries reference the OLD
    // layout (PG_GATHER_* flags from path_guide_kernels.h).
    void buildStructureArrays(const std::vector<CellKey>& newCells,
                              const std::vector<uint64_t>& oldMorton,
                              const std::vector<uint32_t>& oldOffsets,
                              std::vector<uint64_t>& outMorton,
                              std::vector<uint32_t>& outGatherMap);

    // Upload structure (morton/offsets/hash/gather map) to the given grid's
    // buffers on `stream`, growing them as needed. Returns false on OOM.
    struct GridBuffers;
    bool uploadStructure(GridBuffers& grid,
                         const std::vector<uint64_t>& mortonHost,
                         const std::vector<uint32_t>& gatherMap,
                         cudaStream_t stream);

    // Build hash table from morton codes + level offsets, upload to GPU
    bool buildAndUploadHashTable(uint64_t*& outKeys, uint32_t*& outValues,
                                 uint32_t& outSize, uint32_t& outShift, uint32_t& outAllocated,
                                 const std::vector<uint64_t>& mortonHost,
                                 uint32_t totalCells,
                                 cudaStream_t stream = nullptr);

    PathGuideGridConfig m_config;
    uint64_t* m_mortonCodes = nullptr;         // alias of render grid
    float* m_data = nullptr;                   // alias of render grid
    uint32_t* m_levelOffsetsDevice = nullptr;  // alias of render grid
    std::vector<uint32_t> m_levelOffsets;      // host structure mirror
    std::vector<uint64_t> m_mortonCodesHost;   // host structure mirror
    uint32_t m_numLevels = 0;
    uint32_t m_entryStride = 0;
    uint32_t m_totalCells = 0;
    uint32_t m_allocatedCells = 0;

    uint32_t* m_stagingBuffer = nullptr;
    uint32_t* m_stagingCount = nullptr;
    uint32_t m_stagingCapacity = 0;

    // Precomputed level resolutions: avoids std::pow in hot loops
    static constexpr uint32_t MAX_LEVELS = 16;
    uint32_t m_levelResolutions[MAX_LEVELS] = {};  // floor(base_res * scale^level)

    // Hash table aliases (point into the active render grid)
    uint64_t* m_hashKeys = nullptr;
    uint32_t* m_hashValues = nullptr;
    uint32_t  m_hashTableSize = 0;
    uint32_t  m_hashShift = 0;
    uint32_t  m_hashAllocated = 0;

    // ── Double-buffered GPU grid (render vs build) ──
    struct GridBuffers {
        uint64_t* mortonCodes = nullptr;
        float*    data = nullptr;
        uint32_t* levelOffsetsDevice = nullptr;
        uint32_t  totalCells = 0;
        uint32_t  allocatedCells = 0;

        // Hash table (device memory)
        uint64_t* hashKeys = nullptr;
        uint32_t* hashValues = nullptr;
        uint32_t  hashTableSize = 0;
        uint32_t  hashShift = 0;
        uint32_t  hashAllocated = 0;
    };
    GridBuffers m_grids[2];
    int m_renderGridIdx = 0;
    int m_buildGridIdx = 1;

    // Gather map (device, grow-only): old→new cell index mapping for the
    // re-layout kernel
    uint32_t* m_d_gatherMap = nullptr;
    uint32_t  m_gatherMapCapacity = 0;
    uint32_t  m_pendingGatherCount = 0;   // cells in the pending build grid

    // Async readback infrastructure
    cudaStream_t m_readbackStream = nullptr;
    cudaEvent_t  m_renderDoneEvent = nullptr;
    cudaEvent_t  m_readbackDoneEvent = nullptr;

    // Pinned host buffers (required for non-blocking async D2H)
    uint32_t* m_pinnedStagingCount = nullptr;
    uint32_t* m_pinnedStagingBuffer = nullptr;
    size_t    m_pinnedStagingBufferCapacity = 0;

    // Two-phase readback: count first (4 bytes), buffer only when count > 0.
    // Steady state (no new cells) costs a 4-byte copy per build interval.
    enum class AsyncState { Idle, CountInFlight, BufferInFlight, ReadbackReady };
    AsyncState m_asyncState = AsyncState::Idle;
    uint32_t m_pendingCurrentFrame = 0;
    uint32_t m_pendingStagingCount = 0;
    bool m_asyncInitialized = false;
};

} // namespace spectra
