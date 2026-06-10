#include "path_guide_grid.h"
#include "path_guide_kernels.h"
#include "vmf_fitting.h"
#include <cstring>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <unordered_set>
#include <unordered_map>

namespace spectra {

namespace {

// 3D Morton encode: (ix, iy, iz) -> 64-bit. Up to 21 bits per axis (0x1fffff).
inline uint64_t spread3(uint64_t x) {
    x &= 0x1fffff;
    x = (x | x << 32) & 0x001f00000000ffffull;
    x = (x | x << 16) & 0x001f0000ff0000ffull;
    x = (x | x << 8)  & 0x010f00f00f00f00full;
    x = (x | x << 4)  & 0x10c30c30c30c30c3ull;
    x = (x | x << 2)  & 0x1249249249249249ull;
    return x;
}

inline uint64_t mortonEncode(uint32_t ix, uint32_t iy, uint32_t iz) {
    return spread3(ix) | (spread3(iy) << 1) | (spread3(iz) << 2);
}

inline uint64_t compact3(uint64_t x) {
    x &= 0x1249249249249249ull;
    x = (x | x >> 2)  & 0x10c30c30c30c30c3ull;
    x = (x | x >> 4)  & 0x010f00f00f00f00full;
    x = (x | x >> 8)  & 0x001f0000ff0000ffull;
    x = (x | x >> 16) & 0x001f00000000ffffull;
    x = (x | x >> 32) & 0x1fffffull;
    return x;
}

// Pack (level, morton) for hash sets/maps. Morton uses < 48 bits for any
// realistic resolution (level 15 caps at 2^16 per axis).
inline uint64_t packKey(uint32_t level, uint64_t morton) {
    return (static_cast<uint64_t>(level) << 48) | morton;
}

} // namespace

void PathGuideGrid::mortonDecode(uint64_t morton, uint32_t& ix, uint32_t& iy, uint32_t& iz) {
    ix = static_cast<uint32_t>(compact3(morton));
    iy = static_cast<uint32_t>(compact3(morton >> 1));
    iz = static_cast<uint32_t>(compact3(morton >> 2));
}

PathGuideGrid::~PathGuideGrid() {
    shutdown();
}

bool PathGuideGrid::init(const PathGuideGridConfig& config) {
    shutdown();
    m_config = config;
    m_numLevels = config.num_levels;
    m_entryStride = config.entry_stride > 0 ? config.entry_stride : PG_ENTRY_STRIDE;
    m_levelOffsets.assign(m_numLevels + 1, 0u);

    // Precompute level resolutions to avoid std::pow in hot loops
    for (uint32_t l = 0; l < MAX_LEVELS; l++) {
        float res = std::floor(static_cast<float>(config.base_resolution) *
            std::pow(config.per_level_scale, static_cast<float>(l)));
        m_levelResolutions[l] = (res < 1.0f) ? 1u : static_cast<uint32_t>(res);
    }

    cudaError_t err = cudaMalloc(&m_stagingBuffer, (size_t)config.staging_capacity * 4 * sizeof(uint32_t));
    if (err != cudaSuccess) {
        std::cerr << "[PathGuideGrid] cudaMalloc staging failed: " << cudaGetErrorString(err) << "\n";
        return false;
    }
    err = cudaMalloc(&m_stagingCount, sizeof(uint32_t));
    if (err != cudaSuccess) {
        cudaFree(m_stagingBuffer);
        m_stagingBuffer = nullptr;
        std::cerr << "[PathGuideGrid] cudaMalloc staging count failed: " << cudaGetErrorString(err) << "\n";
        return false;
    }
    m_stagingCapacity = config.staging_capacity;

    // Sparse arrays: start empty (no cells until the first structure build)
    m_totalCells = 0;
    m_mortonCodes = nullptr;
    m_data = nullptr;
    m_levelOffsetsDevice = nullptr;
    err = cudaMalloc(&m_levelOffsetsDevice, (m_numLevels + 1) * sizeof(uint32_t));
    if (err != cudaSuccess) {
        cudaFree(m_stagingCount);
        cudaFree(m_stagingBuffer);
        m_stagingCount = nullptr;
        m_stagingBuffer = nullptr;
        std::cerr << "[PathGuideGrid] cudaMalloc level_offsets failed: " << cudaGetErrorString(err) << "\n";
        return false;
    }
    cudaMemcpy(m_levelOffsetsDevice, m_levelOffsets.data(),
        (m_numLevels + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemset(m_stagingCount, 0, sizeof(uint32_t));
    return true;
}

void PathGuideGrid::shutdown() {
    shutdownAsync();
    if (m_mortonCodes) { cudaFree(m_mortonCodes); m_mortonCodes = nullptr; }
    if (m_data) { cudaFree(m_data); m_data = nullptr; }
    if (m_levelOffsetsDevice) { cudaFree(m_levelOffsetsDevice); m_levelOffsetsDevice = nullptr; }
    if (m_stagingBuffer) { cudaFree(m_stagingBuffer); m_stagingBuffer = nullptr; }
    if (m_stagingCount) { cudaFree(m_stagingCount); m_stagingCount = nullptr; }
    if (m_hashKeys) { cudaFree(m_hashKeys); m_hashKeys = nullptr; }
    if (m_hashValues) { cudaFree(m_hashValues); m_hashValues = nullptr; }
    if (m_d_gatherMap) { cudaFree(m_d_gatherMap); m_d_gatherMap = nullptr; }
    m_gatherMapCapacity = 0;
    m_hashTableSize = 0;
    m_hashShift = 0;
    m_hashAllocated = 0;
    m_levelOffsets.clear();
    m_mortonCodesHost.clear();
    m_numLevels = 0;
    m_entryStride = 0;
    m_totalCells = 0;
    m_allocatedCells = 0;
    m_stagingCapacity = 0;
}

void PathGuideGrid::clear(cudaStream_t stream) {
    const GridBuffers& g = m_grids[m_renderGridIdx];
    float* data = m_asyncInitialized ? g.data : m_data;
    uint32_t cells = m_asyncInitialized ? g.totalCells : m_totalCells;
    if (!data || cells == 0) return;
    size_t numBytes = static_cast<size_t>(cells) * static_cast<size_t>(m_entryStride) * sizeof(float);
    if (stream)
        cudaMemsetAsync(data, 0, numBytes, stream);
    else
        cudaMemset(data, 0, numBytes);
}

void PathGuideGrid::resetStagingCount(cudaStream_t stream) {
    if (!m_stagingCount) return;
    if (stream)
        cudaMemsetAsync(m_stagingCount, 0, sizeof(uint32_t), stream);
    else
        cudaMemset(m_stagingCount, 0, sizeof(uint32_t));
}

//------------------------------------------------------------------------------
// Device-side lobe refit
//------------------------------------------------------------------------------

void PathGuideGrid::refitLobes(uint32_t currentFrame, cudaStream_t stream) {
    const GridBuffers& g = m_grids[m_renderGridIdx];
    float* data = m_asyncInitialized ? g.data : m_data;
    uint32_t cells = m_asyncInitialized ? g.totalCells : m_totalCells;
    if (!data || cells == 0) return;
    launchRefitCells(data, cells, m_config.refit_ema_decay, currentFrame, stream);
}

//------------------------------------------------------------------------------
// Structure construction (shared by async build and refinement)
//------------------------------------------------------------------------------

void PathGuideGrid::buildStructureArrays(const std::vector<CellKey>& newCells,
                                         const std::vector<uint64_t>& oldMorton,
                                         const std::vector<uint32_t>& oldOffsets,
                                         std::vector<uint64_t>& outMorton,
                                         std::vector<uint32_t>& outGatherMap)
{
    // Per-level counts and prefix sum
    m_levelOffsets.assign(m_numLevels + 1, 0u);
    for (const auto& k : newCells)
        m_levelOffsets[k.level + 1]++;
    for (uint32_t l = 0; l < m_numLevels; l++)
        m_levelOffsets[l + 1] += m_levelOffsets[l];

    m_totalCells = static_cast<uint32_t>(newCells.size());
    outMorton.assign(m_totalCells, 0ull);
    outGatherMap.assign(m_totalCells, PG_GATHER_NEW_CELL);

    std::vector<uint32_t> levelCellCount(m_numLevels, 0u);
    for (const auto& k : newCells) {
        uint32_t base = m_levelOffsets[k.level];
        uint32_t idx = base + levelCellCount[k.level]++;
        outMorton[idx] = k.morton;
    }

    // Old -> new gather map via per-level two-pointer merge (both sorted)
    if (!oldMorton.empty()) {
        for (uint32_t lev = 0; lev < m_numLevels && lev + 1 < static_cast<uint32_t>(oldOffsets.size()); lev++) {
            uint32_t oi = oldOffsets[lev];
            uint32_t oEnd = oldOffsets[lev + 1];
            uint32_t ni = m_levelOffsets[lev];
            uint32_t nEnd = m_levelOffsets[lev + 1];
            while (oi < oEnd && ni < nEnd) {
                if (oldMorton[oi] < outMorton[ni]) {
                    oi++;
                } else if (oldMorton[oi] > outMorton[ni]) {
                    ni++;
                } else {
                    outGatherMap[ni] = oi;
                    oi++;
                    ni++;
                }
            }
        }
    }

    m_mortonCodesHost = outMorton;
}

bool PathGuideGrid::uploadStructure(GridBuffers& grid,
                                    const std::vector<uint64_t>& mortonHost,
                                    const std::vector<uint32_t>& gatherMap,
                                    cudaStream_t stream)
{
    uint32_t total = static_cast<uint32_t>(mortonHost.size());

    // Grow grid buffers with 2x headroom. cudaMalloc/cudaFree imply a
    // device-wide sync on WDDM, so growth must stay rare.
    if (total > grid.allocatedCells) {
        uint32_t newCapacity = std::max(total * 2u, 8192u);

        uint64_t* newMorton = nullptr;
        float* newData = nullptr;
        cudaError_t err = cudaMalloc(&newMorton, (size_t)newCapacity * sizeof(uint64_t));
        if (err != cudaSuccess) {
            std::cerr << "[PathGuideGrid] cudaMalloc morton failed: " << cudaGetErrorString(err) << "\n";
            return false;
        }
        err = cudaMalloc(&newData, (size_t)newCapacity * m_entryStride * sizeof(float));
        if (err != cudaSuccess) {
            cudaFree(newMorton);
            std::cerr << "[PathGuideGrid] cudaMalloc data failed: " << cudaGetErrorString(err) << "\n";
            return false;
        }
        if (grid.mortonCodes) cudaFree(grid.mortonCodes);
        if (grid.data) cudaFree(grid.data);
        grid.mortonCodes = newMorton;
        grid.data = newData;
        grid.allocatedCells = newCapacity;
    }

    // Grow the gather map buffer
    if (total > m_gatherMapCapacity) {
        uint32_t newCapacity = std::max(total * 2u, 8192u);
        uint32_t* newMap = nullptr;
        cudaError_t err = cudaMalloc(&newMap, (size_t)newCapacity * sizeof(uint32_t));
        if (err != cudaSuccess) {
            std::cerr << "[PathGuideGrid] cudaMalloc gather map failed: " << cudaGetErrorString(err) << "\n";
            return false;
        }
        if (m_d_gatherMap) cudaFree(m_d_gatherMap);
        m_d_gatherMap = newMap;
        m_gatherMapCapacity = newCapacity;
    }

    // Pageable H2D async copies briefly block the CPU but do not stall other
    // streams. These run on the readback stream (build) or render stream
    // (refinement) — never against an in-flight launch that reads them.
    cudaMemcpyAsync(grid.mortonCodes, mortonHost.data(),
        (size_t)total * sizeof(uint64_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(grid.levelOffsetsDevice, m_levelOffsets.data(),
        (m_numLevels + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(m_d_gatherMap, gatherMap.data(),
        (size_t)total * sizeof(uint32_t), cudaMemcpyHostToDevice, stream);

    if (!buildAndUploadHashTable(grid.hashKeys, grid.hashValues,
                                 grid.hashTableSize, grid.hashShift,
                                 grid.hashAllocated, mortonHost, total, stream)) {
        return false;
    }

    grid.totalCells = total;
    return true;
}

bool PathGuideGrid::buildAndUploadHashTable(
    uint64_t*& outKeys, uint32_t*& outValues,
    uint32_t& outSize, uint32_t& outShift, uint32_t& outAllocated,
    const std::vector<uint64_t>& mortonHost,
    uint32_t totalCells,
    cudaStream_t stream)
{
    if (totalCells == 0) {
        outSize = 0;
        outShift = 64;
        return true;
    }

    // Table size: next power of 2 >= totalCells * 2 (50% load factor), minimum 64
    uint32_t tableSize = 64;
    while (tableSize < totalCells * 2) tableSize *= 2;
    // Shift for Fibonacci hashing: top bits of 64-bit product
    uint32_t shift = 64;
    for (uint32_t s = tableSize; s > 1; s >>= 1) shift--;

    // Build hash table on CPU (key layout must match sparseCellIndexHash)
    std::vector<uint64_t> keys(tableSize, 0xFFFFFFFFFFFFFFFFULL);
    std::vector<uint32_t> values(tableSize, 0xFFFFFFFFu);
    uint32_t mask = tableSize - 1;

    for (uint32_t level = 0; level < m_numLevels; level++) {
        uint32_t start = m_levelOffsets[level];
        uint32_t end = m_levelOffsets[level + 1];
        for (uint32_t i = start; i < end; i++) {
            uint64_t key = (static_cast<uint64_t>(level) << 48) | mortonHost[i];
            uint32_t slot = static_cast<uint32_t>((key * 0x9E3779B97F4A7C15ULL) >> shift) & mask;
            while (keys[slot] != 0xFFFFFFFFFFFFFFFFULL) {
                slot = (slot + 1) & mask;
            }
            keys[slot] = key;
            values[slot] = i;
        }
    }

    // Allocate/grow GPU buffers if needed
    if (tableSize > outAllocated) {
        if (outKeys) cudaFree(outKeys);
        if (outValues) cudaFree(outValues);
        outKeys = nullptr;
        outValues = nullptr;

        cudaError_t err = cudaMalloc(&outKeys, (size_t)tableSize * sizeof(uint64_t));
        if (err != cudaSuccess) {
            std::cerr << "[PathGuideGrid] Hash table cudaMalloc keys failed: " << cudaGetErrorString(err) << "\n";
            outSize = 0;
            outShift = 64;
            return false;
        }
        err = cudaMalloc(&outValues, (size_t)tableSize * sizeof(uint32_t));
        if (err != cudaSuccess) {
            cudaFree(outKeys);
            outKeys = nullptr;
            std::cerr << "[PathGuideGrid] Hash table cudaMalloc values failed: " << cudaGetErrorString(err) << "\n";
            outSize = 0;
            outShift = 64;
            return false;
        }
        outAllocated = tableSize;
    }

    if (stream) {
        cudaMemcpyAsync(outKeys, keys.data(), (size_t)tableSize * sizeof(uint64_t),
            cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(outValues, values.data(), (size_t)tableSize * sizeof(uint32_t),
            cudaMemcpyHostToDevice, stream);
    } else {
        cudaMemcpy(outKeys, keys.data(), (size_t)tableSize * sizeof(uint64_t), cudaMemcpyHostToDevice);
        cudaMemcpy(outValues, values.data(), (size_t)tableSize * sizeof(uint32_t), cudaMemcpyHostToDevice);
    }

    outSize = tableSize;
    outShift = shift;
    return true;
}

//------------------------------------------------------------------------------
// Descriptors
//------------------------------------------------------------------------------

SparsePathGuideDescriptor PathGuideGrid::getDescriptor() const {
    return getRenderDescriptor();
}

SparsePathGuideDescriptor PathGuideGrid::getRenderDescriptor() const {
    SparsePathGuideDescriptor d = {};
    if (m_asyncInitialized) {
        const GridBuffers& g = m_grids[m_renderGridIdx];
        d.morton_codes = g.mortonCodes;
        d.data = g.data;
        d.level_offsets = g.levelOffsetsDevice;
        d.num_levels = m_numLevels;
        d.entry_stride = m_entryStride;
        d.hash_keys = g.hashKeys;
        d.hash_values = g.hashValues;
        d.hash_table_size = g.hashTableSize;
        d.hash_shift = g.hashShift;
    } else {
        d.morton_codes = m_mortonCodes;
        d.data = m_data;
        d.level_offsets = m_levelOffsetsDevice;
        d.num_levels = m_numLevels;
        d.entry_stride = m_entryStride;
        d.hash_keys = m_hashKeys;
        d.hash_values = m_hashValues;
        d.hash_table_size = m_hashTableSize;
        d.hash_shift = m_hashShift;
    }
    d.base_resolution = m_config.base_resolution;
    d.per_level_scale = m_config.per_level_scale;
    d.bounds_min[0] = m_config.bounds_min[0];
    d.bounds_min[1] = m_config.bounds_min[1];
    d.bounds_min[2] = m_config.bounds_min[2];
    d.bounds_max[0] = m_config.bounds_max[0];
    d.bounds_max[1] = m_config.bounds_max[1];
    d.bounds_max[2] = m_config.bounds_max[2];
    return d;
}

PathGuideStagingDescriptor PathGuideGrid::getStagingDescriptor() const {
    PathGuideStagingDescriptor d = {};
    d.buffer = m_stagingBuffer;
    d.count = m_stagingCount;
    d.capacity = m_stagingCapacity;
    return d;
}

//------------------------------------------------------------------------------
// Wireframe generation (debug visualization, host structure mirror)
//------------------------------------------------------------------------------

std::vector<float> PathGuideGrid::generateEdgeVertices(uint32_t level) const {
    std::vector<float> vertices;

    if (level >= m_numLevels || m_mortonCodesHost.empty()) {
        return vertices;
    }

    uint32_t start = m_levelOffsets[level];
    uint32_t end = m_levelOffsets[level + 1];
    uint32_t cellCount = end - start;
    if (cellCount == 0) {
        return vertices;
    }

    vertices.reserve(static_cast<size_t>(cellCount) * 72);

    float res = static_cast<float>(m_levelResolutions[level]);
    float invRes = 1.0f / res;

    float boundsExtentX = m_config.bounds_max[0] - m_config.bounds_min[0];
    float boundsExtentY = m_config.bounds_max[1] - m_config.bounds_min[1];
    float boundsExtentZ = m_config.bounds_max[2] - m_config.bounds_min[2];

    auto pushVertex = [&](float x, float y, float z) {
        vertices.push_back(x);
        vertices.push_back(y);
        vertices.push_back(z);
    };

    for (uint32_t i = start; i < end; i++) {
        uint64_t morton = m_mortonCodesHost[i];
        uint32_t ix, iy, iz;
        mortonDecode(morton, ix, iy, iz);

        float minX = m_config.bounds_min[0] + static_cast<float>(ix) * invRes * boundsExtentX;
        float maxX = m_config.bounds_min[0] + static_cast<float>(ix + 1) * invRes * boundsExtentX;
        float minY = m_config.bounds_min[1] + static_cast<float>(iy) * invRes * boundsExtentY;
        float maxY = m_config.bounds_min[1] + static_cast<float>(iy + 1) * invRes * boundsExtentY;
        float minZ = m_config.bounds_min[2] + static_cast<float>(iz) * invRes * boundsExtentZ;
        float maxZ = m_config.bounds_min[2] + static_cast<float>(iz + 1) * invRes * boundsExtentZ;

        // 12 edges of the box
        pushVertex(minX, minY, minZ); pushVertex(maxX, minY, minZ);
        pushVertex(maxX, minY, minZ); pushVertex(maxX, maxY, minZ);
        pushVertex(maxX, maxY, minZ); pushVertex(minX, maxY, minZ);
        pushVertex(minX, maxY, minZ); pushVertex(minX, minY, minZ);
        pushVertex(minX, minY, maxZ); pushVertex(maxX, minY, maxZ);
        pushVertex(maxX, minY, maxZ); pushVertex(maxX, maxY, maxZ);
        pushVertex(maxX, maxY, maxZ); pushVertex(minX, maxY, maxZ);
        pushVertex(minX, maxY, maxZ); pushVertex(minX, minY, maxZ);
        pushVertex(minX, minY, minZ); pushVertex(minX, minY, maxZ);
        pushVertex(maxX, minY, minZ); pushVertex(maxX, minY, maxZ);
        pushVertex(maxX, maxY, minZ); pushVertex(maxX, maxY, maxZ);
        pushVertex(minX, maxY, minZ); pushVertex(minX, maxY, maxZ);
    }

    return vertices;
}

std::vector<float> PathGuideGrid::generateEdgeVerticesAllLevels() const {
    std::vector<float> vertices;
    if (m_mortonCodesHost.empty() || m_totalCells == 0) return vertices;

    vertices.reserve(static_cast<size_t>(m_totalCells) * 72);

    for (uint32_t level = 0; level < m_numLevels; level++) {
        auto levelVerts = generateEdgeVertices(level);
        vertices.insert(vertices.end(), levelVerts.begin(), levelVerts.end());
    }

    return vertices;
}

//------------------------------------------------------------------------------
// Cell inspection (UI click): host structure lookup + 64-byte readback
//------------------------------------------------------------------------------

PathGuideGrid::CellInspectionResult PathGuideGrid::inspectCellAtPosition(float px, float py, float pz) const {
    CellInspectionResult result = {};

    const GridBuffers& g = m_grids[m_renderGridIdx];
    const float* deviceData = m_asyncInitialized ? g.data : m_data;
    if (m_totalCells == 0 || m_mortonCodesHost.empty() || !deviceData) {
        return result;
    }

    float extX = m_config.bounds_max[0] - m_config.bounds_min[0];
    float extY = m_config.bounds_max[1] - m_config.bounds_min[1];
    float extZ = m_config.bounds_max[2] - m_config.bounds_min[2];
    if (extX < 1e-6f || extY < 1e-6f || extZ < 1e-6f) return result;

    float nx = (px - m_config.bounds_min[0]) / extX;
    float ny = (py - m_config.bounds_min[1]) / extY;
    float nz = (pz - m_config.bounds_min[2]) / extZ;

    if (nx < 0.0f || nx > 1.0f || ny < 0.0f || ny > 1.0f || nz < 0.0f || nz > 1.0f) {
        return result;  // Out of bounds
    }

    // Search from max_level down to min_level for the finest existing cell
    for (int level = static_cast<int>(m_config.max_level); level >= static_cast<int>(m_config.min_level); level--) {
        if (static_cast<uint32_t>(level) >= m_numLevels) continue;

        uint32_t resU = m_levelResolutions[level];
        float res = static_cast<float>(resU);

        uint32_t ix = static_cast<uint32_t>(nx * res);
        uint32_t iy = static_cast<uint32_t>(ny * res);
        uint32_t iz = static_cast<uint32_t>(nz * res);
        if (ix >= resU) ix = resU - 1;
        if (iy >= resU) iy = resU - 1;
        if (iz >= resU) iz = resU - 1;

        uint64_t morton = mortonEncode(ix, iy, iz);
        uint32_t start = m_levelOffsets[level];
        uint32_t end = m_levelOffsets[level + 1];

        auto it = std::lower_bound(m_mortonCodesHost.begin() + start, m_mortonCodesHost.begin() + end, morton);
        if (it != m_mortonCodesHost.begin() + end && *it == morton) {
            size_t globalIdx = static_cast<size_t>(it - m_mortonCodesHost.begin());

            result.found = true;
            result.level = static_cast<uint32_t>(level);
            result.ix = ix;
            result.iy = iy;
            result.iz = iz;

            // Read back just this cell (synchronous, 64 bytes — UI click only)
            cudaMemcpy(result.data, deviceData + globalIdx * m_entryStride,
                       m_entryStride * sizeof(float), cudaMemcpyDeviceToHost);

            float invRes = 1.0f / res;
            result.aabbMin[0] = m_config.bounds_min[0] + static_cast<float>(ix) * invRes * extX;
            result.aabbMin[1] = m_config.bounds_min[1] + static_cast<float>(iy) * invRes * extY;
            result.aabbMin[2] = m_config.bounds_min[2] + static_cast<float>(iz) * invRes * extZ;
            result.aabbMax[0] = m_config.bounds_min[0] + static_cast<float>(ix + 1) * invRes * extX;
            result.aabbMax[1] = m_config.bounds_min[1] + static_cast<float>(iy + 1) * invRes * extY;
            result.aabbMax[2] = m_config.bounds_min[2] + static_cast<float>(iz + 1) * invRes * extZ;

            return result;
        }
    }

    return result;
}

//------------------------------------------------------------------------------
// Async structure pipeline
//------------------------------------------------------------------------------

bool PathGuideGrid::initAsync() {
    if (m_asyncInitialized) return true;
    if (!m_stagingBuffer || !m_levelOffsetsDevice) return false;

    // Create low-priority non-blocking readback stream
    int leastPriority = 0, greatestPriority = 0;
    cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority);
    cudaError_t err = cudaStreamCreateWithPriority(&m_readbackStream, cudaStreamNonBlocking, leastPriority);
    if (err != cudaSuccess) {
        std::cerr << "[PathGuideGrid] Failed to create readback stream: " << cudaGetErrorString(err) << "\n";
        return false;
    }

    err = cudaEventCreateWithFlags(&m_renderDoneEvent, cudaEventDisableTiming);
    if (err != cudaSuccess) {
        std::cerr << "[PathGuideGrid] Failed to create render done event: " << cudaGetErrorString(err) << "\n";
        shutdownAsync();
        return false;
    }
    err = cudaEventCreateWithFlags(&m_readbackDoneEvent, cudaEventDisableTiming);
    if (err != cudaSuccess) {
        std::cerr << "[PathGuideGrid] Failed to create readback done event: " << cudaGetErrorString(err) << "\n";
        shutdownAsync();
        return false;
    }

    // Pinned staging count (4 bytes)
    err = cudaMallocHost(&m_pinnedStagingCount, sizeof(uint32_t));
    if (err != cudaSuccess) {
        std::cerr << "[PathGuideGrid] Failed to alloc pinned staging count: " << cudaGetErrorString(err) << "\n";
        shutdownAsync();
        return false;
    }

    // Pinned staging buffer
    m_pinnedStagingBufferCapacity = static_cast<size_t>(m_stagingCapacity) * 4;
    err = cudaMallocHost(&m_pinnedStagingBuffer, m_pinnedStagingBufferCapacity * sizeof(uint32_t));
    if (err != cudaSuccess) {
        std::cerr << "[PathGuideGrid] Failed to alloc pinned staging buffer: " << cudaGetErrorString(err) << "\n";
        shutdownAsync();
        return false;
    }

    // Initialize both grid buffers with their own levelOffsetsDevice
    for (int i = 0; i < 2; i++) {
        err = cudaMalloc(&m_grids[i].levelOffsetsDevice, (m_numLevels + 1) * sizeof(uint32_t));
        if (err != cudaSuccess) {
            std::cerr << "[PathGuideGrid] Failed to alloc grid[" << i << "] level_offsets: " << cudaGetErrorString(err) << "\n";
            shutdownAsync();
            return false;
        }
        cudaMemcpy(m_grids[i].levelOffsetsDevice, m_levelOffsets.data(),
            (m_numLevels + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice);
    }

    // Free the original m_levelOffsetsDevice from init() — both grids now have
    // their own copies. Redirect the alias to grid[0]'s copy.
    if (m_levelOffsetsDevice &&
        m_levelOffsetsDevice != m_grids[0].levelOffsetsDevice &&
        m_levelOffsetsDevice != m_grids[1].levelOffsetsDevice) {
        cudaFree(m_levelOffsetsDevice);
    }
    m_levelOffsetsDevice = m_grids[0].levelOffsetsDevice;

    // Copy existing single-buffer state into grid[0] (render grid).
    // Grid[1] (build) keeps only its levelOffsetsDevice from the loop above;
    // its morton/data/hash buffers are allocated by the first uploadStructure.
    m_grids[0].mortonCodes = m_mortonCodes;
    m_grids[0].data = m_data;
    m_grids[0].totalCells = m_totalCells;
    m_grids[0].allocatedCells = m_allocatedCells;
    m_grids[0].hashKeys = m_hashKeys;
    m_grids[0].hashValues = m_hashValues;
    m_grids[0].hashTableSize = m_hashTableSize;
    m_grids[0].hashShift = m_hashShift;
    m_grids[0].hashAllocated = m_hashAllocated;

    m_renderGridIdx = 0;
    m_buildGridIdx = 1;
    m_asyncState = AsyncState::Idle;
    m_asyncInitialized = true;

    std::cout << "[PathGuideGrid] Async pipeline initialized (readback stream priority=" << leastPriority << ")\n";
    return true;
}

void PathGuideGrid::shutdownAsync() {
    if (m_readbackStream) {
        cudaStreamSynchronize(m_readbackStream);
        cudaStreamDestroy(m_readbackStream);
        m_readbackStream = nullptr;
    }
    if (m_renderDoneEvent) { cudaEventDestroy(m_renderDoneEvent); m_renderDoneEvent = nullptr; }
    if (m_readbackDoneEvent) { cudaEventDestroy(m_readbackDoneEvent); m_readbackDoneEvent = nullptr; }
    if (m_pinnedStagingCount) { cudaFreeHost(m_pinnedStagingCount); m_pinnedStagingCount = nullptr; }
    if (m_pinnedStagingBuffer) { cudaFreeHost(m_pinnedStagingBuffer); m_pinnedStagingBuffer = nullptr; }
    m_pinnedStagingBufferCapacity = 0;

    // Free grid buffers. The single-buffer aliases point into one of the
    // grids; collect unique pointers to avoid double-frees.
    {
        void* toFree[12] = {};
        int nFree = 0;
        auto addUnique = [&](void* p) {
            if (!p) return;
            for (int j = 0; j < nFree; j++) {
                if (toFree[j] == p) return;
            }
            toFree[nFree++] = p;
        };
        for (int i = 0; i < 2; i++) {
            addUnique(m_grids[i].mortonCodes);
            addUnique(m_grids[i].data);
            addUnique(m_grids[i].levelOffsetsDevice);
            addUnique(m_grids[i].hashKeys);
            addUnique(m_grids[i].hashValues);
        }
        for (int j = 0; j < nFree; j++) {
            cudaFree(toFree[j]);
        }
    }

    for (int i = 0; i < 2; i++) {
        m_grids[i] = GridBuffers{};
    }

    // Null out the single-buffer aliases so shutdown() doesn't double-free
    m_mortonCodes = nullptr;
    m_data = nullptr;
    m_levelOffsetsDevice = nullptr;
    m_totalCells = 0;
    m_allocatedCells = 0;
    m_hashKeys = nullptr;
    m_hashValues = nullptr;
    m_hashTableSize = 0;
    m_hashShift = 0;
    m_hashAllocated = 0;

    m_asyncState = AsyncState::Idle;
    m_asyncInitialized = false;
}

void PathGuideGrid::beginAsyncReadback(cudaStream_t renderStream, uint32_t currentFrame) {
    if (!m_asyncInitialized || m_asyncState != AsyncState::Idle) return;

    // Mark render + staging writes complete on renderStream
    cudaEventRecord(m_renderDoneEvent, renderStream);
    cudaStreamWaitEvent(m_readbackStream, m_renderDoneEvent, 0);

    // Phase 1: read back ONLY the count (4 bytes). The staging buffer itself
    // is fetched in phase 2 and only when count > 0 — steady state (full
    // coverage, no new cells) costs almost nothing.
    cudaMemcpyAsync(m_pinnedStagingCount, m_stagingCount, sizeof(uint32_t),
        cudaMemcpyDeviceToHost, m_readbackStream);

    // Snapshot the staging buffer BEFORE resetting the count: the buffer copy
    // happens in phase 2 from the same region, and appends after the reset
    // start at index 0 again. To avoid phase-2 reading entries overwritten by
    // new appends, the reset is deferred to phase 2 (after the buffer copy).
    cudaEventRecord(m_readbackDoneEvent, m_readbackStream);

    m_pendingCurrentFrame = currentFrame;
    m_asyncState = AsyncState::CountInFlight;
}

bool PathGuideGrid::pollAsyncReadback() {
    if (m_asyncState == AsyncState::ReadbackReady) return true;

    if (m_asyncState == AsyncState::CountInFlight) {
        cudaError_t err = cudaEventQuery(m_readbackDoneEvent);
        if (err == cudaErrorNotReady) return false;
        if (err != cudaSuccess) {
            std::cerr << "[PathGuideGrid] pollAsyncReadback error: " << cudaGetErrorString(err) << "\n";
            m_asyncState = AsyncState::Idle;
            return false;
        }

        uint32_t count = *m_pinnedStagingCount;
        if (count > m_stagingCapacity) {
            std::cerr << "[PathGuideGrid] WARNING: staging overflow (" << count
                      << " entries, capacity " << m_stagingCapacity << ")\n";
            count = m_stagingCapacity;
        }
        m_pendingStagingCount = count;

        if (count == 0) {
            // Nothing staged — no structure change. finishBuildFromReadback
            // will no-op and the caller clears its in-flight flag.
            m_asyncState = AsyncState::ReadbackReady;
            return true;
        }

        // Phase 2: fetch the staged entries, then reset the counter so the
        // next window starts fresh.
        cudaMemcpyAsync(m_pinnedStagingBuffer, m_stagingBuffer,
            (size_t)count * 4 * sizeof(uint32_t), cudaMemcpyDeviceToHost, m_readbackStream);
        cudaMemsetAsync(m_stagingCount, 0, sizeof(uint32_t), m_readbackStream);
        cudaEventRecord(m_readbackDoneEvent, m_readbackStream);
        m_asyncState = AsyncState::BufferInFlight;
        return false;
    }

    if (m_asyncState == AsyncState::BufferInFlight) {
        cudaError_t err = cudaEventQuery(m_readbackDoneEvent);
        if (err == cudaSuccess) {
            m_asyncState = AsyncState::ReadbackReady;
            return true;
        }
        if (err != cudaErrorNotReady) {
            std::cerr << "[PathGuideGrid] pollAsyncReadback error: " << cudaGetErrorString(err) << "\n";
            m_asyncState = AsyncState::Idle;
        }
        return false;
    }

    return false;
}

bool PathGuideGrid::finishBuildFromReadback() {
    if (m_asyncState != AsyncState::ReadbackReady) return false;
    m_asyncState = AsyncState::Idle;

    uint32_t numEntries = m_pendingStagingCount;
    m_pendingStagingCount = 0;
    if (numEntries == 0) return false;

    // Deduplicate staging entries and merge with the existing cell set.
    std::unordered_set<uint64_t> uniqueSet;
    uniqueSet.reserve(std::min(numEntries, 32768u) + m_totalCells);

    for (uint32_t i = 0; i < numEntries; i++) {
        uint32_t level = m_pinnedStagingBuffer[i * 4 + 0];
        uint32_t ix    = m_pinnedStagingBuffer[i * 4 + 1];
        uint32_t iy    = m_pinnedStagingBuffer[i * 4 + 2];
        uint32_t iz    = m_pinnedStagingBuffer[i * 4 + 3];
        if (level >= m_numLevels) continue;
        uint32_t resU = m_levelResolutions[level];
        if (ix >= resU) ix = resU - 1;
        if (iy >= resU) iy = resU - 1;
        if (iz >= resU) iz = resU - 1;
        uniqueSet.insert(packKey(level, mortonEncode(ix, iy, iz)));
    }

    // Merge existing cells (cumulative grid)
    uint32_t prevTotalCells = m_totalCells;
    for (uint32_t lev = 0; lev < m_numLevels; lev++) {
        uint32_t start = m_levelOffsets[lev];
        uint32_t end = m_levelOffsets[lev + 1];
        for (uint32_t i = start; i < end; i++) {
            uniqueSet.insert(packKey(lev, m_mortonCodesHost[i]));
        }
    }

    if (static_cast<uint32_t>(uniqueSet.size()) == prevTotalCells) {
        return false;  // No new cells — nothing to do
    }

    std::vector<CellKey> keys;
    keys.reserve(uniqueSet.size());
    for (uint64_t packed : uniqueSet) {
        keys.push_back({ static_cast<uint32_t>(packed >> 48),
                         packed & ((1ull << 48) - 1) });
    }
    std::sort(keys.begin(), keys.end());

    std::vector<uint64_t> prevMorton = m_mortonCodesHost;
    std::vector<uint32_t> prevOffsets = m_levelOffsets;

    std::vector<uint64_t> mortonHost;
    std::vector<uint32_t> gatherMap;
    buildStructureArrays(keys, prevMorton, prevOffsets, mortonHost, gatherMap);

    GridBuffers& buildGrid = m_grids[m_buildGridIdx];
    if (!uploadStructure(buildGrid, mortonHost, gatherMap, m_readbackStream)) {
        // Restore the host structure mirror — the device still has the old grid.
        m_mortonCodesHost = prevMorton;
        m_levelOffsets = prevOffsets;
        m_totalCells = prevTotalCells;
        return false;
    }

    // Mark uploads complete so swapGrids can order the gather after them.
    cudaEventRecord(m_readbackDoneEvent, m_readbackStream);
    m_pendingGatherCount = static_cast<uint32_t>(mortonHost.size());

    std::cout << "[PathGuide] Structure build: " << prevTotalCells << " -> "
              << m_totalCells << " cells\n";
    return true;
}

void PathGuideGrid::swapGrids(cudaStream_t renderStream) {
    if (!m_asyncInitialized || m_pendingGatherCount == 0) return;

    // Order: structure uploads (readback stream) -> gather (render stream) ->
    // next launch. The gather reads the LIVE old data at this point in the
    // render stream, so every training deposit made since the staging
    // snapshot is carried over — the old pipeline dropped them.
    if (renderStream && m_readbackDoneEvent) {
        cudaStreamWaitEvent(renderStream, m_readbackDoneEvent, 0);
    }

    GridBuffers& buildGrid = m_grids[m_buildGridIdx];
    const GridBuffers& renderGrid = m_grids[m_renderGridIdx];
    launchGatherCells(buildGrid.data, renderGrid.data, m_d_gatherMap,
                      m_pendingGatherCount, m_pendingCurrentFrame, renderStream);
    m_pendingGatherCount = 0;

    std::swap(m_renderGridIdx, m_buildGridIdx);

    // Update the single-buffer aliases to the new render grid
    const GridBuffers& newRender = m_grids[m_renderGridIdx];
    m_mortonCodes = newRender.mortonCodes;
    m_data = newRender.data;
    m_levelOffsetsDevice = newRender.levelOffsetsDevice;
    m_allocatedCells = newRender.allocatedCells;
    m_hashKeys = newRender.hashKeys;
    m_hashValues = newRender.hashValues;
    m_hashTableSize = newRender.hashTableSize;
    m_hashShift = newRender.hashShift;
    m_hashAllocated = newRender.hashAllocated;
}

//------------------------------------------------------------------------------
// Adaptive refinement (subdivide poorly-fit cells, coarsen stale cells)
//------------------------------------------------------------------------------

bool PathGuideGrid::runRefinementPass(uint32_t currentFrame, cudaStream_t stream) {
    const GridBuffers& renderGrid = m_grids[m_renderGridIdx];
    float* deviceData = m_asyncInitialized ? renderGrid.data : m_data;
    uint32_t totalCells = m_asyncInitialized ? renderGrid.totalCells : m_totalCells;
    if (!deviceData || totalCells == 0 || m_mortonCodesHost.empty()) {
        return false;
    }
    // Don't restructure while an async build holds the build grid
    if (m_pendingGatherCount > 0 || m_asyncState != AsyncState::Idle) {
        return false;
    }

    // Bulk stats readback (refinement is rare — every N builds — so one
    // synchronous copy here is fine; it replaced a per-build full readback).
    std::vector<float> stats(static_cast<size_t>(totalCells) * m_entryStride);
    cudaMemcpyAsync(stats.data(), deviceData, stats.size() * sizeof(float),
        cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    const uint32_t maxLevel = m_config.max_level;
    const uint32_t minLevel = m_config.min_level;

    std::vector<CellKey> newCells;
    newCells.reserve(totalCells);
    // Subdivided children -> parent old index (for lobe warm start)
    std::unordered_map<uint64_t, uint32_t> childParent;

    uint32_t numSubdivided = 0, numCoarsened = 0;

    for (uint32_t lev = 0; lev < m_numLevels; lev++) {
        uint32_t start = m_levelOffsets[lev];
        uint32_t end = m_levelOffsets[lev + 1];
        for (uint32_t g = start; g < end; g++) {
            const float* cell = stats.data() + (size_t)g * m_entryStride;
            float sumX = cell[PG_CUM_SUM_X];
            float sumY = cell[PG_CUM_SUM_Y];
            float sumZ = cell[PG_CUM_SUM_Z];
            float sumW = cell[PG_CUM_SUM_W];
            float lastHitFrame = cell[PG_LAST_HIT_FRAME];

            // Coarsening: drop cells that haven't been hit for a long,
            // data-scaled grace period and carry little training weight.
            float framesSinceHit = (currentFrame > static_cast<uint32_t>(lastHitFrame))
                ? static_cast<float>(currentFrame - static_cast<uint32_t>(lastHitFrame)) : 0.0f;
            float gracePeriod = std::max(120.0f, sumW * 10.0f);
            if (lev > minLevel && framesSinceHit > gracePeriod && sumW < 10.0f) {
                numCoarsened++;
                continue;  // omit from the new cell set
            }

            // Subdivision: poor single-lobe fit at this spatial resolution.
            // Log-likelihood per unit weight below the "moderately
            // concentrated" baseline, or a near-uniform fitted kappa, means
            // the directional distribution is too complex for one lobe here —
            // refine spatially. (BIC-flavored: the heuristic scales with the
            // evidence sumW, so sparse cells don't split prematurely.)
            bool subdivide = false;
            if (lev < maxLevel && sumW >= 2.0f) {
                float LL = vmf_fitting::logLikelihoodSingleLobe(sumX, sumY, sumZ, sumW);
                bool poorFit = (LL / sumW < 0.0f);
                float kappa = cell[PG_KAPPA];
                bool lobeIsWide = (kappa > 0.0f && kappa < 2.0f);
                subdivide = poorFit || lobeIsWide;
            }

            if (subdivide) {
                uint32_t pIx, pIy, pIz;
                mortonDecode(m_mortonCodesHost[g], pIx, pIy, pIz);
                for (int dz = 0; dz < 2; dz++) {
                    for (int dy = 0; dy < 2; dy++) {
                        for (int dx = 0; dx < 2; dx++) {
                            uint64_t childMorton = mortonEncode(pIx * 2 + dx, pIy * 2 + dy, pIz * 2 + dz);
                            newCells.push_back({ lev + 1, childMorton });
                            childParent[packKey(lev + 1, childMorton)] = g;
                        }
                    }
                }
                numSubdivided++;
            } else {
                newCells.push_back({ lev, m_mortonCodesHost[g] });
            }
        }
    }

    if (numSubdivided == 0 && numCoarsened == 0) {
        return false;
    }

    std::sort(newCells.begin(), newCells.end());
    newCells.erase(std::unique(newCells.begin(), newCells.end()), newCells.end());

    std::cout << "[PathGuide] Refine: +" << (numSubdivided * 8)
              << " -" << (numSubdivided + numCoarsened) << " cells\n";

    std::vector<uint64_t> prevMorton = m_mortonCodesHost;
    std::vector<uint32_t> prevOffsets = m_levelOffsets;

    std::vector<uint64_t> mortonHost;
    std::vector<uint32_t> gatherMap;
    buildStructureArrays(newCells, prevMorton, prevOffsets, mortonHost, gatherMap);

    // Patch in the lobe-warm-start mapping for subdivided children
    for (uint32_t lev = 0; lev < m_numLevels; lev++) {
        uint32_t start = m_levelOffsets[lev];
        uint32_t end = m_levelOffsets[lev + 1];
        for (uint32_t i = start; i < end; i++) {
            if (gatherMap[i] == PG_GATHER_NEW_CELL) {
                auto it = childParent.find(packKey(lev, mortonHost[i]));
                if (it != childParent.end()) {
                    gatherMap[i] = it->second | PG_GATHER_LOBE_ONLY;
                }
            }
        }
    }

    if (!m_asyncInitialized) {
        // Without the async double buffer there is no second grid to build
        // into; refinement requires initAsync (the app always initializes it).
        std::cerr << "[PathGuideGrid] Refinement requires the async pipeline\n";
        return false;
    }

    // Upload structure + gather + swap, all on the render stream (ordering is
    // implicit; no events needed).
    GridBuffers& buildGrid = m_grids[m_buildGridIdx];
    if (!uploadStructure(buildGrid, mortonHost, gatherMap, stream)) {
        m_mortonCodesHost = prevMorton;
        m_levelOffsets = prevOffsets;
        m_totalCells = static_cast<uint32_t>(prevMorton.size());
        return false;
    }
    launchGatherCells(buildGrid.data, renderGrid.data, m_d_gatherMap,
                      static_cast<uint32_t>(mortonHost.size()), currentFrame, stream);

    std::swap(m_renderGridIdx, m_buildGridIdx);
    const GridBuffers& newRender = m_grids[m_renderGridIdx];
    m_mortonCodes = newRender.mortonCodes;
    m_data = newRender.data;
    m_levelOffsetsDevice = newRender.levelOffsetsDevice;
    m_allocatedCells = newRender.allocatedCells;
    m_hashKeys = newRender.hashKeys;
    m_hashValues = newRender.hashValues;
    m_hashTableSize = newRender.hashTableSize;
    m_hashShift = newRender.hashShift;
    m_hashAllocated = newRender.hashAllocated;

    return true;
}

} // namespace spectra
