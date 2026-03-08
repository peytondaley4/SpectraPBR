#pragma once

//------------------------------------------------------------------------------
// Sparse Multi-Resolution Path Guide Grid (collision-free, GPU-friendly)
//
// Per-level sorted arrays of (Morton code, data). Only occupied cells exist;
// lookup is binary search by Morton code. No hash, no collisions.
// References:
//   - Müller et al., "Practical Path Guiding for Efficient Light-Transport
//     Simulation", EGSR 2017 (Computer Graphics Forum). SD-tree / directional
//     guiding; we use a sparse voxel grid + vMF mixture per cell.
//   - Yalçıner & Akyüz, "Path Guiding for Wavefront Path Tracing with Sparse
//     Voxel Octree", arXiv 2405.06997.
//   - von Mises–Fisher (vMF) distribution: C3(κ) exp(κ μ·ω); sampling via
//     Wood/Ulrich (w = 1 + ln(ξ+(1-ξ)e^{-2κ})/κ, then orthonormal tangent + circle).
//
// Layout:
//   - level_offsets[0 .. num_levels]: level l's cells at indices [level_offsets[l], level_offsets[l+1])
//   - morton_codes[]: sorted by Morton within each level (device)
//   - data[]: 6 floats per cell (2 vMF lobes × 3), same order (device)
//
// Occupancy is collected via a staging buffer: closest-hit appends (level, ix, iy, iz).
// Build (CPU): read back staging, sort/unique by (level, Morton), upload sparse arrays.
//
// PDF: each cell stores 2 vMF lobes (theta, phi, kappa) × 3 floats = 6 floats.
//------------------------------------------------------------------------------

#include <cstdint>
#include <cuda_runtime.h>
#include <vector>

namespace spectra {

constexpr uint32_t PATH_GUIDE_VMF_FLOATS_PER_LOBE = 3;
constexpr uint32_t PATH_GUIDE_LOBES_PER_SLOT = 2;
constexpr uint32_t PATH_GUIDE_VMF_FLOATS = PATH_GUIDE_VMF_FLOATS_PER_LOBE * PATH_GUIDE_LOBES_PER_SLOT;  // 6 floats

// Per-cell statistics for adaptive refinement
// Layout: [sumX, sumY, sumZ, sumW, pi_0 (mixture weight), lastHitFrame] = 6 floats
constexpr uint32_t PATH_GUIDE_STATS_FLOATS = 6;

// Offset within cell data for the mixture weight pi_0 (lobe 0 weight)
constexpr uint32_t PATH_GUIDE_MIX_WEIGHT_OFFSET = 10;

// Total entry stride: vMF lobes (6) + refinement stats (6) = 12 floats per cell
constexpr uint32_t PATH_GUIDE_ENTRY_STRIDE_DEFAULT =
    PATH_GUIDE_VMF_FLOATS + PATH_GUIDE_STATS_FLOATS;  // 12 floats per cell

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
