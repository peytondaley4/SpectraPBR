#include "path_guide_grid.h"
#include "path_guide_kernels.h"
#include "log.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>

namespace spectra {

namespace {

// 3D Morton encode: (ix, iy, iz) -> 64-bit. Up to 21 bits per axis (0x1fffff).
// Must match pgMortonEncode64 in path_guide_hash_device.h.
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

// Pack (level, morton) — must match pgPackKey on device.
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

    // The kernels and the deposit path address the cell layout via the
    // PG_* compile-time offsets — a different runtime stride would silently
    // corrupt. Lock the invariant instead of trusting the config.
    if (config.entry_stride != PG_ENTRY_STRIDE) {
        std::cerr << "[PathGuideGrid] entry_stride " << config.entry_stride
                  << " unsupported (layout requires " << PG_ENTRY_STRIDE << ")\n";
        return false;
    }
    m_entryStride = PG_ENTRY_STRIDE;

    // Octree child derivation (pgChildMorton) and the top-down lookup both
    // require resolutions to exactly double per level.
    if (config.per_level_scale != 2.0f) {
        std::cerr << "[PathGuideGrid] per_level_scale must be 2.0 (got "
                  << config.per_level_scale << ")\n";
        return false;
    }
    // Hash sizing below doubles to 2x capacity in a uint32 — keep headroom.
    if (config.cell_capacity == 0 || config.cell_capacity > (1u << 28)) {
        std::cerr << "[PathGuideGrid] cell_capacity " << config.cell_capacity
                  << " out of range (1 .. 2^28)\n";
        return false;
    }

    // Precompute level resolutions to avoid std::pow in hot loops
    for (uint32_t l = 0; l < MAX_LEVELS; l++) {
        float res = std::floor(static_cast<float>(config.base_resolution) *
            std::pow(config.per_level_scale, static_cast<float>(l)));
        m_levelResolutions[l] = (res < 1.0f) ? 1u : static_cast<uint32_t>(res);
    }

    // Hash table: next power of 2 >= 2x capacity (<= 50% load), minimum 64
    uint32_t tableSize = 64;
    while (tableSize < config.cell_capacity * 2u) tableSize *= 2;
    uint32_t shift = 64;
    for (uint32_t s = tableSize; s > 1; s >>= 1) shift--;

    auto fail = [this](const char* what, cudaError_t err) {
        std::cerr << "[PathGuideGrid] cudaMalloc " << what << " failed: "
                  << cudaGetErrorString(err) << "\n";
        shutdown();
        return false;
    };

    cudaError_t err = cudaMalloc(&m_hashKeys, (size_t)tableSize * sizeof(uint64_t));
    if (err != cudaSuccess) return fail("hash keys", err);
    err = cudaMalloc(&m_hashValues, (size_t)tableSize * sizeof(uint32_t));
    if (err != cudaSuccess) return fail("hash values", err);
    err = cudaMalloc(&m_cellKeys, (size_t)config.cell_capacity * sizeof(uint64_t));
    if (err != cudaSuccess) return fail("cell keys", err);
    err = cudaMalloc(&m_cellCounter, sizeof(uint32_t));
    if (err != cudaSuccess) return fail("cell counter", err);
    err = cudaMalloc(&m_counterSnapshot, sizeof(uint32_t));
    if (err != cudaSuccess) return fail("counter snapshot", err);
    err = cudaMalloc(&m_data, (size_t)config.cell_capacity * m_entryStride * sizeof(float));
    if (err != cudaSuccess) return fail("cell data", err);

    // 0xFF bytes = PG_KEY_EMPTY for keys and PG_VALUE_PENDING for values
    cudaMemset(m_hashKeys, 0xFF, (size_t)tableSize * sizeof(uint64_t));
    cudaMemset(m_hashValues, 0xFF, (size_t)tableSize * sizeof(uint32_t));
    cudaMemset(m_cellKeys, 0, (size_t)config.cell_capacity * sizeof(uint64_t));
    cudaMemset(m_cellCounter, 0, sizeof(uint32_t));
    cudaMemset(m_counterSnapshot, 0, sizeof(uint32_t));
    cudaMemset(m_data, 0, (size_t)config.cell_capacity * m_entryStride * sizeof(float));

    m_hashTableSize = tableSize;
    m_hashShift = shift;

    err = cudaMallocHost(&m_pinnedCount, sizeof(uint32_t));
    if (err != cudaSuccess) return fail("pinned count", err);
    *m_pinnedCount = 0;
    err = cudaEventCreateWithFlags(&m_countEvent, cudaEventDisableTiming);
    if (err != cudaSuccess) {
        std::cerr << "[PathGuideGrid] count event create failed: "
                  << cudaGetErrorString(err) << "\n";
        shutdown();
        return false;
    }

    m_countInFlight = false;
    m_lastCellCount = 0;
    m_hostKeys.clear();
    m_hostKeyToIndex.clear();

    std::cout << "[PathGuideGrid] Device-resident table: capacity "
              << config.cell_capacity << " cells, hash " << tableSize
              << " slots, " << ((size_t)config.cell_capacity * m_entryStride * sizeof(float)
                  + (size_t)tableSize * 12 + (size_t)config.cell_capacity * 8) / (1024 * 1024)
              << " MB\n";
    return true;
}

void PathGuideGrid::shutdown() {
    if (m_countEvent) {
        cudaEventSynchronize(m_countEvent);
        cudaEventDestroy(m_countEvent);
        m_countEvent = nullptr;
    }
    if (m_pinnedCount) { cudaFreeHost(m_pinnedCount); m_pinnedCount = nullptr; }
    if (m_hashKeys) { cudaFree(m_hashKeys); m_hashKeys = nullptr; }
    if (m_hashValues) { cudaFree(m_hashValues); m_hashValues = nullptr; }
    if (m_cellKeys) { cudaFree(m_cellKeys); m_cellKeys = nullptr; }
    if (m_cellCounter) { cudaFree(m_cellCounter); m_cellCounter = nullptr; }
    if (m_counterSnapshot) { cudaFree(m_counterSnapshot); m_counterSnapshot = nullptr; }
    if (m_data) { cudaFree(m_data); m_data = nullptr; }
    m_hashTableSize = 0;
    m_hashShift = 64;
    m_entryStride = 0;
    m_countInFlight = false;
    m_lastCellCount = 0;
    m_hostKeys.clear();
    m_hostKeyToIndex.clear();
}

void PathGuideGrid::clear(cudaStream_t stream) {
    if (!m_data) return;
    size_t numBytes = (size_t)m_config.cell_capacity * m_entryStride * sizeof(float);
    if (stream)
        cudaMemsetAsync(m_data, 0, numBytes, stream);
    else
        cudaMemset(m_data, 0, numBytes);
    // Restore the tetrahedral lobe init for already-allocated cells — an
    // all-zero mixture would collapse the deposit hard-assignment onto
    // lobe 0. Stream-ordered after the memset.
    launchInitCells(m_data, m_cellCounter, m_config.cell_capacity, stream);
}

//------------------------------------------------------------------------------
// Kernel launchers
//------------------------------------------------------------------------------

void PathGuideGrid::refitLobes(uint32_t currentFrame, cudaStream_t stream) {
    if (!m_data || !m_cellCounter) return;
    // Mean cell edge at level 0 (bounds may be anisotropic; the geometric
    // kappa cap is a heuristic, the average is plenty). Halves per level on
    // the device side.
    float res0 = static_cast<float>(m_levelResolutions[0]);
    float baseCellSize =
        ((m_config.bounds_max[0] - m_config.bounds_min[0]) +
         (m_config.bounds_max[1] - m_config.bounds_min[1]) +
         (m_config.bounds_max[2] - m_config.bounds_min[2])) / (3.0f * res0);
    launchRefitCells(m_data, m_cellKeys, m_cellCounter, m_config.cell_capacity,
                     m_config.refit_ema_decay, baseCellSize, currentFrame, stream);
}

void PathGuideGrid::runSubdivisionPass(uint32_t currentFrame, cudaStream_t stream) {
    if (!m_data || !m_cellCounter || !m_counterSnapshot) return;
    // Snapshot the allocation counter (stream-ordered) so the kernel only
    // processes cells that existed before the pass — children it inserts
    // must not cascade-subdivide within the same pass.
    cudaMemcpyAsync(m_counterSnapshot, m_cellCounter, sizeof(uint32_t),
                    cudaMemcpyDeviceToDevice, stream);
    launchSubdivideCells(m_hashKeys, m_hashValues, m_hashTableSize, m_hashShift,
                         m_cellKeys, m_cellCounter, m_config.cell_capacity,
                         m_counterSnapshot,
                         m_data, m_entryStride,
                         m_config.max_level, m_config.subdivide_count_threshold,
                         m_config.subdivide_contrast_threshold,
                         currentFrame, stream);
}

//------------------------------------------------------------------------------
// UI count poll (non-blocking)
//------------------------------------------------------------------------------

void PathGuideGrid::requestCellCountAsync(cudaStream_t stream) {
    if (!m_cellCounter || !m_pinnedCount || !m_countEvent) return;

    if (m_countInFlight) {
        cudaError_t err = cudaEventQuery(m_countEvent);
        if (err == cudaErrorNotReady) return;  // previous poll still in flight
        m_countInFlight = false;
        if (err == cudaSuccess) {
            m_lastCellCount = std::min(*m_pinnedCount, m_config.cell_capacity);
        }
    }

    cudaMemcpyAsync(m_pinnedCount, m_cellCounter, sizeof(uint32_t),
                    cudaMemcpyDeviceToHost, stream);
    cudaEventRecord(m_countEvent, stream);
    m_countInFlight = true;
}

//------------------------------------------------------------------------------
// Host debug mirror (synchronous; viz toggle / inspector click only)
//------------------------------------------------------------------------------

bool PathGuideGrid::refreshHostMirror() {
    if (!m_cellCounter || !m_cellKeys) return false;

    uint32_t count = 0;
    cudaMemcpy(&count, m_cellCounter, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    count = std::min(count, m_config.cell_capacity);
    m_lastCellCount = count;

    m_hostKeys.resize(count);
    m_hostKeyToIndex.clear();
    if (count == 0) return false;

    cudaMemcpy(m_hostKeys.data(), m_cellKeys, (size_t)count * sizeof(uint64_t),
               cudaMemcpyDeviceToHost);

    m_hostKeyToIndex.reserve(count * 2);
    for (uint32_t i = 0; i < count; i++) {
        m_hostKeyToIndex.emplace(m_hostKeys[i], i);
    }
    return true;
}

//------------------------------------------------------------------------------
// Wireframe generation (debug visualization, host mirror)
//------------------------------------------------------------------------------

void PathGuideGrid::appendCellEdges(std::vector<float>& vertices,
                                    uint32_t level, uint64_t morton) const {
    uint32_t ix, iy, iz;
    mortonDecode(morton, ix, iy, iz);

    float res = static_cast<float>(m_levelResolutions[level < MAX_LEVELS ? level : MAX_LEVELS - 1]);
    float invRes = 1.0f / res;

    float extX = m_config.bounds_max[0] - m_config.bounds_min[0];
    float extY = m_config.bounds_max[1] - m_config.bounds_min[1];
    float extZ = m_config.bounds_max[2] - m_config.bounds_min[2];

    float minX = m_config.bounds_min[0] + static_cast<float>(ix) * invRes * extX;
    float maxX = m_config.bounds_min[0] + static_cast<float>(ix + 1) * invRes * extX;
    float minY = m_config.bounds_min[1] + static_cast<float>(iy) * invRes * extY;
    float maxY = m_config.bounds_min[1] + static_cast<float>(iy + 1) * invRes * extY;
    float minZ = m_config.bounds_min[2] + static_cast<float>(iz) * invRes * extZ;
    float maxZ = m_config.bounds_min[2] + static_cast<float>(iz + 1) * invRes * extZ;

    auto pushVertex = [&vertices](float x, float y, float z) {
        vertices.push_back(x);
        vertices.push_back(y);
        vertices.push_back(z);
    };

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

std::vector<float> PathGuideGrid::generateEdgeVertices(uint32_t level) const {
    std::vector<float> vertices;
    for (uint64_t key : m_hostKeys) {
        uint32_t cellLevel = static_cast<uint32_t>(key >> 48);
        if (cellLevel != level) continue;
        appendCellEdges(vertices, cellLevel, key & ((1ull << 48) - 1));
    }
    return vertices;
}

std::vector<float> PathGuideGrid::generateEdgeVerticesAllLevels() const {
    std::vector<float> vertices;
    vertices.reserve(m_hostKeys.size() * 72);
    for (uint64_t key : m_hostKeys) {
        appendCellEdges(vertices, static_cast<uint32_t>(key >> 48), key & ((1ull << 48) - 1));
    }
    return vertices;
}

//------------------------------------------------------------------------------
// Cell inspection (UI click): host mirror lookup + 64-byte readback
//------------------------------------------------------------------------------

PathGuideGrid::CellInspectionResult PathGuideGrid::inspectCellAtPosition(
    float px, float py, float pz)
{
    CellInspectionResult result = {};
    if (!m_data) return result;
    if (!refreshHostMirror()) return result;

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

    // Deepest existing cell containing the position (max_level down to start)
    for (int level = static_cast<int>(m_config.max_level);
         level >= static_cast<int>(m_config.start_level); level--) {
        if (level >= static_cast<int>(MAX_LEVELS)) continue;

        uint32_t resU = m_levelResolutions[level];
        float res = static_cast<float>(resU);

        uint32_t ix = static_cast<uint32_t>(nx * res);
        uint32_t iy = static_cast<uint32_t>(ny * res);
        uint32_t iz = static_cast<uint32_t>(nz * res);
        if (ix >= resU) ix = resU - 1;
        if (iy >= resU) iy = resU - 1;
        if (iz >= resU) iz = resU - 1;

        auto it = m_hostKeyToIndex.find(packKey(static_cast<uint32_t>(level),
                                                mortonEncode(ix, iy, iz)));
        if (it == m_hostKeyToIndex.end()) continue;

        result.found = true;
        result.level = static_cast<uint32_t>(level);
        result.ix = ix;
        result.iy = iy;
        result.iz = iz;

        // Read back just this cell (synchronous, 64 bytes — UI click only)
        cudaMemcpy(result.data, m_data + (size_t)it->second * m_entryStride,
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

    return result;
}

//------------------------------------------------------------------------------
// Descriptor
//------------------------------------------------------------------------------

SparsePathGuideDescriptor PathGuideGrid::getDescriptor() const {
    SparsePathGuideDescriptor d = {};
    d.data = m_data;
    d.hash_keys = m_hashKeys;
    d.hash_values = m_hashValues;
    d.hash_table_size = m_hashTableSize;
    d.hash_shift = m_hashShift;
    d.cell_keys = m_cellKeys;
    d.cell_counter = m_cellCounter;
    d.cell_capacity = m_config.cell_capacity;
    d.entry_stride = m_entryStride;
    d.num_levels = m_config.num_levels;
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

} // namespace spectra
