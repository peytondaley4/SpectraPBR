#pragma once

//------------------------------------------------------------------------------
// Sparse Multi-Resolution Path Guide Grid (collision-free, GPU-friendly)
//
// === Architecture ===
//
// Spatial: sparse voxel grid with Morton-coded cells and O(1) hash lookup.
// Directional: 2-lobe von Mises–Fisher (vMF) mixture per cell.
// Training: online via GPU atomicAdd of importance-weighted (Li/p) deposits.
// Fitting: CPU-side single-lobe fitFromSums + kappa-aware bimodal detection.
// Refinement: adaptive subdivision (high variance) / coarsening (unused cells).
//
// === Training Pipeline ===
//
// GPU (per frame):
//   1. Closest-hit (primary + bounce) atomically accumulates (dir*weight, weight)
//      into cell interval stats [6-9] via pathGuideTrainCell().
//   2. New-cell seeding appends (level, ix, iy, iz) to staging buffer.
//
// CPU (periodic, async):
//   1. Read back staging + cell data from GPU
//   2. Merge new cells from staging into sparse grid (cumulative)
//   3. EMA-decay lifetime sums, add interval sums → fit vMF lobes
//   4. Zero interval stats, upload fitted lobes + hash table to GPU
//   5. Adaptive refinement: subdivide high-variance, coarsen stale cells
//
// === Two-lobe Limitation ===
//
// Online aggregate sums (sumX/Y/Z/W) are sufficient statistics for a SINGLE
// vMF lobe, but NOT for a mixture. Proper EM mixture fitting (see
// vmf_fitting::fitTwoLobes) requires per-sample data, which is not available
// in the online setting. The current heuristic uses kappa-aware divergence
// between interval and cumulative mean directions to detect bimodality.
//
// References:
//   - Müller et al., "Practical Path Guiding for Efficient Light-Transport
//     Simulation", EGSR 2017 (Computer Graphics Forum).
//   - Yalçıner & Akyüz, "Path Guiding for Wavefront Path Tracing with Sparse
//     Voxel Octree", arXiv 2405.06997.
//   - Heitz, "Sampling the GGX Distribution of Visible Normals", JCGT 2018.
//   - von Mises–Fisher (vMF): C3(κ) exp(κ μ·ω); Wood/Ulrich sampling.
//
// === Cell Data Layout (12 floats) ===
//
//   [0-2]: lobe 0 (theta, phi, kappa)   — primary fitted lobe
//   [3-5]: lobe 1 (theta, phi, kappa)   — secondary (kappa=0 if inactive)
//   [6-9]: interval stats (sumX, sumY, sumZ, sumW)  — zeroed after each build
//   [10]:  pi_0 (mixture weight for lobe 0, 1.0 if single-lobe)
//   [11]:  lastHitFrame (for coarsening decisions)
//------------------------------------------------------------------------------

#include <cstdint>
#include <cuda_runtime.h>
#include <vector>

namespace spectra {

// ─── Named cell data offsets (must match path_guide_grid_device.h exactly) ──
constexpr uint32_t PG_FLOATS_PER_LOBE   = 3;
constexpr uint32_t PG_NUM_LOBES         = 2;
constexpr uint32_t PG_LOBE0_THETA       = 0;
constexpr uint32_t PG_LOBE0_PHI         = 1;
constexpr uint32_t PG_LOBE0_KAPPA       = 2;
constexpr uint32_t PG_LOBE1_THETA       = 3;
constexpr uint32_t PG_LOBE1_PHI         = 4;
constexpr uint32_t PG_LOBE1_KAPPA       = 5;
constexpr uint32_t PG_STAT_SUM_X        = 6;
constexpr uint32_t PG_STAT_SUM_Y        = 7;
constexpr uint32_t PG_STAT_SUM_Z        = 8;
constexpr uint32_t PG_STAT_SUM_W        = 9;
constexpr uint32_t PG_MIX_WEIGHT        = 10;
constexpr uint32_t PG_LAST_HIT_FRAME    = 11;
constexpr uint32_t PG_ENTRY_STRIDE      = 12;

// Legacy aliases for existing code
constexpr uint32_t PATH_GUIDE_VMF_FLOATS_PER_LOBE = PG_FLOATS_PER_LOBE;
constexpr uint32_t PATH_GUIDE_LOBES_PER_SLOT = PG_NUM_LOBES;
constexpr uint32_t PATH_GUIDE_VMF_FLOATS = PG_NUM_LOBES * PG_FLOATS_PER_LOBE;
constexpr uint32_t PATH_GUIDE_MIX_WEIGHT_OFFSET = PG_MIX_WEIGHT;
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
    uint32_t staging_capacity = 1u << 19;  // 512K entries for cell seeding (no debug staging)

    // Adaptive refinement settings
    uint32_t start_level = 2;              // Initial coarse level for new regions
    uint32_t min_level = 1;                // Minimum level (coarsest allowed)
    uint32_t max_level = 6;                // Maximum level (finest allowed)
    uint32_t refine_interval_frames = 30;  // Run refinement every N frames
    float subdivide_sample_threshold = 50.0f;    // Min sumW (luminance-weighted) before considering subdivision
    float subdivide_variance_threshold = 0.08f;  // Directional variance (1 - R̄) threshold for subdivision
    uint32_t coarsen_frames_threshold = 480;    // Frames without hits before coarsening (high to let grid settle)
};

//------------------------------------------------------------------------------
// GPU descriptor for sparse grid (copy to device / launch params)
//------------------------------------------------------------------------------
struct SparsePathGuideDescriptor {
    const uint64_t* morton_codes;   // Sorted per level (device)
    float* data;                    // 6 floats per cell (device)
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
// Staging for occupancy collection (closest-hit appends; build reads back)
//------------------------------------------------------------------------------
struct PathGuideStagingDescriptor {
    uint32_t* buffer;    // 4 × capacity: [level, ix, iy, iz] per entry (device)
    uint32_t* count;     // Atomic counter (device), 1 element
    uint32_t capacity;
};

//------------------------------------------------------------------------------
// PathGuideGrid: sparse buffers + staging, build from staging
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

    // Staging: call before each frame to let closest-hit append occupancy
    void resetStagingCount(cudaStream_t stream = nullptr);

    // Build sparse grid from staging (readback, sort/unique, upload). Call after trace.
    // currentFrame: used to initialize lastHitFrame for newly-created cells so they
    // get a full coarsening grace period before refinement considers removing them.
    bool buildFromStaging(cudaStream_t stream = nullptr, uint32_t currentFrame = 0);

    SparsePathGuideDescriptor getDescriptor() const;
    SparsePathGuideDescriptor getRenderDescriptor() const;  // Always returns active render grid
    PathGuideStagingDescriptor getStagingDescriptor() const;

    // ── Async readback pipeline (double-buffered, non-blocking) ──
    bool initAsync();
    void shutdownAsync();
    void beginAsyncReadback(cudaStream_t renderStream, uint32_t currentFrame);
    bool pollAsyncReadback();
    bool finishBuildFromReadback();
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
    // Returns flat array of [x,y,z] triplets, 24 vertices per cell (12 edges * 2 vertices)
    std::vector<float> generateEdgeVertices(uint32_t level) const;
    // Generate wireframe for ALL levels (shows actual adaptive subdivision structure)
    std::vector<float> generateEdgeVerticesAllLevels() const;

    // Adaptive refinement: call periodically (every N frames) to subdivide/coarsen
    // Returns true if grid structure changed
    bool runRefinementPass(uint32_t currentFrame, cudaStream_t stream = nullptr);

    // Cell inspection: look up the cell containing a world-space position
    struct CellInspectionResult {
        bool found = false;
        uint32_t level = 0;
        uint32_t ix = 0, iy = 0, iz = 0;
        float data[12] = {};       // All 12 floats (vMF + stats)
        float aabbMin[3] = {};
        float aabbMax[3] = {};
    };
    CellInspectionResult inspectCellAtPosition(float px, float py, float pz) const;

    // Get refinement statistics
    uint32_t getStartLevel() const { return m_config.start_level; }
    uint32_t getMinLevel() const { return m_config.min_level; }
    uint32_t getMaxLevel() const { return m_config.max_level; }

private:
    // Morton decode helper
    static void mortonDecode(uint64_t morton, uint32_t& ix, uint32_t& iy, uint32_t& iz);

    PathGuideGridConfig m_config;
    uint64_t* m_mortonCodes = nullptr;
    float* m_data = nullptr;
    uint32_t* m_levelOffsetsDevice = nullptr;  // num_levels+1 on device
    std::vector<uint32_t> m_levelOffsets;      // host copy
    std::vector<uint64_t> m_mortonCodesHost;   // host copy for edge generation
    std::vector<float> m_dataHost;             // host copy to avoid D2H readback on rebuild
    uint32_t m_numLevels = 0;
    uint32_t m_entryStride = 0;
    uint32_t m_totalCells = 0;
    uint32_t m_allocatedCells = 0;  // Capacity of m_mortonCodes/m_data GPU buffers

    uint32_t* m_stagingBuffer = nullptr;
    uint32_t* m_stagingCount = nullptr;
    uint32_t m_stagingCapacity = 0;

    // Precomputed level resolutions: avoids std::pow in hot loops
    static constexpr uint32_t MAX_LEVELS = 16;
    uint32_t m_levelResolutions[MAX_LEVELS] = {};  // floor(base_res * scale^level)

    // Hash table for single-grid path (device memory)
    uint64_t* m_hashKeys = nullptr;
    uint32_t* m_hashValues = nullptr;
    uint32_t  m_hashTableSize = 0;
    uint32_t  m_hashShift = 0;
    uint32_t  m_hashAllocated = 0;

    // Build hash table from current morton codes + level offsets, upload to GPU
    bool buildAndUploadHashTable(uint64_t*& outKeys, uint32_t*& outValues,
                                 uint32_t& outSize, uint32_t& outShift, uint32_t& outAllocated,
                                 const std::vector<uint64_t>& mortonHost,
                                 uint32_t totalCells,
                                 cudaStream_t stream = nullptr);

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
        uint32_t  hashAllocated = 0;  // current GPU capacity
    };
    GridBuffers m_grids[2];
    int m_renderGridIdx = 0;
    int m_buildGridIdx = 1;

    // Async readback infrastructure
    cudaStream_t m_readbackStream = nullptr;
    cudaEvent_t  m_renderDoneEvent = nullptr;
    cudaEvent_t  m_readbackDoneEvent = nullptr;

    // Pinned host buffers (required for non-blocking async D2H)
    uint32_t* m_pinnedStagingCount = nullptr;
    uint32_t* m_pinnedStagingBuffer = nullptr;
    size_t    m_pinnedStagingBufferCapacity = 0;
    float*    m_pinnedGpuData = nullptr;
    size_t    m_pinnedGpuDataCapacity = 0;

    enum class AsyncState { Idle, ReadbackInFlight, ReadbackReady };
    AsyncState m_asyncState = AsyncState::Idle;
    uint32_t m_pendingCurrentFrame = 0;
    bool m_asyncInitialized = false;
};

} // namespace spectra
