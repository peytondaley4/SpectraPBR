#include "path_guide_grid.h"
#include "vmf_fitting.h"
#include <cstring>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <unordered_set>

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

// NOTE: resolutionAtLevel is replaced by precomputed m_levelResolutions[]
// to avoid millions of std::pow calls in hot loops (buildFromStaging, runRefinementPass).

} // namespace

PathGuideGrid::~PathGuideGrid() {
    shutdown();
}

bool PathGuideGrid::init(const PathGuideGridConfig& config) {
    shutdown();
    m_config = config;
    m_numLevels = config.num_levels;
    m_entryStride = config.entry_stride > 0 ? config.entry_stride : 1;
    m_levelOffsets.assign(m_numLevels + 1, 0u);

    // Precompute level resolutions to avoid std::pow in hot loops
    for (uint32_t l = 0; l < MAX_LEVELS; l++) {
        float res = std::floor(static_cast<float>(config.base_resolution) *
            std::pow(config.per_level_scale, static_cast<float>(l)));
        m_levelResolutions[l] = (res < 1.0f) ? 1u : static_cast<uint32_t>(res);
    }

    // Staging: 4 uints per entry (level, ix, iy, iz), plus 1 for atomic count
    size_t stagingBytes = static_cast<size_t>(config.staging_capacity) * 4 * sizeof(uint32_t)
        + sizeof(uint32_t);
    cudaError_t err = cudaMalloc(&m_stagingBuffer, config.staging_capacity * 4 * sizeof(uint32_t));
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

    // Sparse arrays: start empty (no cells until build)
    m_totalCells = 0;
    m_mortonCodes = nullptr;
    m_data = nullptr;
    m_levelOffsetsDevice = nullptr;
    // Allocate minimal level_offsets on device (num_levels+1)
    err = cudaMalloc(&m_levelOffsetsDevice, (m_numLevels + 1) * sizeof(uint32_t));
    if (err != cudaSuccess) {
        cudaFree(m_stagingCount);
        cudaFree(m_stagingBuffer);
        m_stagingCount = nullptr;
        m_stagingBuffer = nullptr;
        std::cerr << "[PathGuideGrid] cudaMalloc level_offsets failed: " << cudaGetErrorString(err) << "\n";
        return false;
    }
    err = cudaMemcpy(m_levelOffsetsDevice, m_levelOffsets.data(),
        (m_numLevels + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(m_levelOffsetsDevice);
        cudaFree(m_stagingCount);
        cudaFree(m_stagingBuffer);
        m_levelOffsetsDevice = nullptr;
        m_stagingCount = nullptr;
        m_stagingBuffer = nullptr;
        std::cerr << "[PathGuideGrid] cudaMemcpy level_offsets failed: " << cudaGetErrorString(err) << "\n";
        return false;
    }
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
    m_hashTableSize = 0;
    m_hashShift = 0;
    m_hashAllocated = 0;
    m_levelOffsets.clear();
    m_numLevels = 0;
    m_entryStride = 0;
    m_totalCells = 0;
    m_allocatedCells = 0;
    m_stagingCapacity = 0;
}

void PathGuideGrid::clear(cudaStream_t stream) {
    if (!m_data || m_totalCells == 0) return;
    size_t numBytes = static_cast<size_t>(m_totalCells) * static_cast<size_t>(m_entryStride) * sizeof(float);
    if (stream)
        cudaMemsetAsync(m_data, 0, numBytes, stream);
    else
        cudaMemset(m_data, 0, numBytes);
    // Also zero host copy to reset lifetime totals
    std::fill(m_dataHost.begin(), m_dataHost.end(), 0.0f);
}

void PathGuideGrid::resetStagingCount(cudaStream_t stream) {
    if (!m_stagingCount) return;
    // Use cudaMemsetAsync instead of cudaMemcpyAsync with pageable host memory.
    // cudaMemcpyAsync from pageable (non-pinned) memory implicitly synchronizes
    // the stream first, which breaks the triple-buffer pipeline on every frame.
    if (stream)
        cudaMemsetAsync(m_stagingCount, 0, sizeof(uint32_t), stream);
    else
        cudaMemset(m_stagingCount, 0, sizeof(uint32_t));
}

bool PathGuideGrid::buildFromStaging(cudaStream_t stream, uint32_t currentFrame) {
    if (!m_stagingBuffer || !m_stagingCount || m_stagingCapacity == 0) {
        return false;
    }
    // Ensure any prior trace has completed so staging count is valid
    if (stream)
        cudaStreamSynchronize(stream);
    uint32_t count = 0;
    cudaError_t err = cudaMemcpy(&count, m_stagingCount, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "[PathGuideGrid] read staging count failed: " << cudaGetErrorString(err) << std::endl;
        return false;
    }
    // When count == 0 (no new staging entries), the grid structure doesn't change
    // but training data still needs processing. Don't return early — let the function
    // flow through with zero staging entries so the existing-cell merge and training
    // processing still run.

    // count = number of entries (each entry = 4 uints: level, ix, iy, iz)
    uint32_t rawCount = count;  // Keep original for overflow detection
    if (count > m_stagingCapacity) {
        std::cerr << "[PathGuideGrid] WARNING: staging overflow! " << rawCount
                  << " entries attempted, capacity is " << m_stagingCapacity
                  << " (" << (100.0f * m_stagingCapacity / rawCount) << "% captured)\n";
        count = m_stagingCapacity;
    }
    uint32_t numEntries = count;
    uint32_t numUints = count * 4;

    std::vector<uint32_t> staging(numUints);
    if (numUints > 0) {
        err = cudaMemcpy(staging.data(), m_stagingBuffer, numUints * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            std::cerr << "[PathGuideGrid] read staging buffer failed: " << cudaGetErrorString(err) << "\n";
            return false;
        }
    }

    // Build unique (level, morton) and sort. Same staging data produces the same
    // cell set; variation between builds comes from different hit positions (e.g. ray jitter).
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
    std::vector<CellKey> keys;
    keys.reserve(numEntries);
    for (uint32_t i = 0; i < numEntries; i++) {
        uint32_t level = staging[i * 4 + 0];
        uint32_t ix   = staging[i * 4 + 1];
        uint32_t iy   = staging[i * 4 + 2];
        uint32_t iz   = staging[i * 4 + 3];
        if (level >= m_numLevels) continue;
        uint32_t resU = m_levelResolutions[level];
        if (ix >= resU) ix = resU - 1;
        if (iy >= resU) iy = resU - 1;
        if (iz >= resU) iz = resU - 1;
        keys.push_back({ level, mortonEncode(ix, iy, iz) });
    }
    // Merge existing cells so the grid is cumulative (not replaced each build).
    // Without this, cells discovered in previous builds would be lost when
    // the current frame's staging doesn't re-discover them.
    if (!m_mortonCodesHost.empty() && m_totalCells > 0) {
        for (uint32_t lev = 0; lev < m_numLevels; lev++) {
            uint32_t start = m_levelOffsets[lev];
            uint32_t end = m_levelOffsets[lev + 1];
            for (uint32_t i = start; i < end; i++) {
                keys.push_back({ lev, m_mortonCodesHost[i] });
            }
        }
    }

    std::sort(keys.begin(), keys.end());
    keys.erase(std::unique(keys.begin(), keys.end()), keys.end());

    // Save previous grid BEFORE overwriting m_levelOffsets (needed for GPU readback carry-forward)
    std::vector<uint64_t> prevMorton = m_mortonCodesHost;
    std::vector<uint32_t> prevOffsets = m_levelOffsets;  // OLD offsets
    uint32_t prevTotalCells = m_mortonCodesHost.empty() ? 0 : static_cast<uint32_t>(m_mortonCodesHost.size());

    // Per-level counts and prefix sum
    m_levelOffsets.assign(m_numLevels + 1, 0u);
    for (const auto& k : keys)
        m_levelOffsets[k.level + 1]++;
    for (uint32_t l = 0; l < m_numLevels; l++)
        m_levelOffsets[l + 1] += m_levelOffsets[l];

    m_totalCells = static_cast<uint32_t>(keys.size());
    if (m_totalCells == 0) {
        m_mortonCodesHost.clear();
        cudaMemcpy(m_levelOffsetsDevice, m_levelOffsets.data(),
            (m_numLevels + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice);
        return true;
    }

    std::vector<uint64_t> mortonHost(m_totalCells);
    std::vector<float> dataHost(static_cast<size_t>(m_totalCells) * m_entryStride, 0.0f);
    std::vector<uint32_t> levelCellCount(m_numLevels, 0u);
    for (const auto& k : keys) {
        uint32_t base = m_levelOffsets[k.level];
        uint32_t idx = base + levelCellCount[k.level]++;
        mortonHost[idx] = k.morton;
    }

    // Keep CPU copy for edge generation
    m_mortonCodesHost = mortonHost;

    // Read back GPU data containing atomic-accumulated interval sums.
    // The GPU m_data has current vMF params (offsets 0-5, 10) AND the interval
    // sums (offsets 6-9) accumulated via atomicAdd in pathGuideTrainAtomic(),
    // plus lastHitFrame (offset 11) via atomicMax.
    // For persistent cells, use GPU readback; for new cells, dataHost is zeroed.
    std::vector<float> gpuReadback;
    if (m_data && prevTotalCells > 0) {
        size_t gpuFloats = static_cast<size_t>(prevTotalCells) * m_entryStride;
        gpuReadback.resize(gpuFloats);
        err = cudaMemcpy(gpuReadback.data(), m_data, gpuFloats * sizeof(float), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            std::cerr << "[PathGuideGrid] Failed to read GPU data: " << cudaGetErrorString(err) << std::endl;
            gpuReadback.clear();
        }
    }

    // Carry forward from GPU readback: persistent cells get their live GPU data
    // (which includes atomically-accumulated interval sums)
    if (!gpuReadback.empty() && prevTotalCells > 0) {
        for (uint32_t lev = 0; lev < m_numLevels && lev < static_cast<uint32_t>(prevOffsets.size()) - 1; lev++) {
            uint32_t pStart = prevOffsets[lev];
            uint32_t pEnd = prevOffsets[lev + 1];
            uint32_t nStart = m_levelOffsets[lev];
            uint32_t nEnd = m_levelOffsets[lev + 1];
            uint32_t pi2 = pStart, ni2 = nStart;
            while (pi2 < pEnd && ni2 < nEnd) {
                if (prevMorton[pi2] < mortonHost[ni2]) {
                    pi2++;
                } else if (prevMorton[pi2] > mortonHost[ni2]) {
                    ni2++;
                } else {
                    // Persistent cell: overwrite with GPU data (has live interval sums)
                    size_t srcOff = static_cast<size_t>(pi2) * m_entryStride;
                    size_t dstOff = static_cast<size_t>(ni2) * m_entryStride;
                    std::memcpy(&dataHost[dstOff], &gpuReadback[srcOff], m_entryStride * sizeof(float));
                    pi2++;
                    ni2++;
                }
            }
        }
    }

    // Resize m_dataHost for new cell count (preserves lifetime totals for persistent cells
    // via the carry-forward above; new cells start at zero)
    {
        std::vector<float> newDataHost(static_cast<size_t>(m_totalCells) * m_entryStride, 0.0f);
        // Carry forward lifetime totals from old m_dataHost for persistent cells
        if (!m_dataHost.empty() && prevTotalCells > 0) {
            for (uint32_t lev = 0; lev < m_numLevels && lev < static_cast<uint32_t>(prevOffsets.size()) - 1; lev++) {
                uint32_t pStart = prevOffsets[lev];
                uint32_t pEnd = prevOffsets[lev + 1];
                uint32_t nStart = m_levelOffsets[lev];
                uint32_t nEnd = m_levelOffsets[lev + 1];
                uint32_t pi3 = pStart, ni3 = nStart;
                while (pi3 < pEnd && ni3 < nEnd) {
                    if (prevMorton[pi3] < mortonHost[ni3]) { pi3++; }
                    else if (prevMorton[pi3] > mortonHost[ni3]) { ni3++; }
                    else {
                        size_t srcOff = static_cast<size_t>(pi3) * m_entryStride;
                        size_t dstOff = static_cast<size_t>(ni3) * m_entryStride;
                        std::memcpy(&newDataHost[dstOff], &m_dataHost[srcOff], m_entryStride * sizeof(float));
                        pi3++; ni3++;
                    }
                }
            }
        }
        m_dataHost = std::move(newDataHost);
    }

    // Accumulate interval sums into lifetime totals with EMA decay, then fit
    // vMF from cumulative sums. Müller 2017 rebuilds from scratch each iteration;
    // in our online setting, EMA decay is the substitute. A lower decay (0.7)
    // gives an effective window of ~3 builds, preventing cells from being
    // permanently committed to early modes while retaining enough history for
    // stable fitting. Combined with importance-weighted training (Li/p on GPU),
    // this allows cells to adapt as the guide improves.
    constexpr float EMA_DECAY = 0.7f;
    uint32_t cellsWithData = 0;
    if (m_totalCells > 0 && m_entryStride >= 12) {
        for (size_t g = 0; g < m_totalCells; g++) {
            size_t base = g * m_entryStride;
            if (base + 11 >= dataHost.size()) break;

            // Extract interval sums from carried-forward GPU data
            float iSumX = dataHost[base + 6];
            float iSumY = dataHost[base + 7];
            float iSumZ = dataHost[base + 8];
            float iSumW = dataHost[base + 9];

            // Accumulate into lifetime totals with EMA decay
            size_t hostBase = g * m_entryStride;
            if (hostBase + 11 < m_dataHost.size()) {
                m_dataHost[hostBase + 6] = EMA_DECAY * m_dataHost[hostBase + 6] + iSumX;
                m_dataHost[hostBase + 7] = EMA_DECAY * m_dataHost[hostBase + 7] + iSumY;
                m_dataHost[hostBase + 8] = EMA_DECAY * m_dataHost[hostBase + 8] + iSumZ;
                m_dataHost[hostBase + 9] = EMA_DECAY * m_dataHost[hostBase + 9] + iSumW;
                m_dataHost[hostBase + 11] = dataHost[base + 11];
            }

            // Fit from cumulative sums (much more stable than interval-only)
            float cumSumX = (hostBase + 9 < m_dataHost.size()) ? m_dataHost[hostBase + 6] : iSumX;
            float cumSumY = (hostBase + 9 < m_dataHost.size()) ? m_dataHost[hostBase + 7] : iSumY;
            float cumSumZ = (hostBase + 9 < m_dataHost.size()) ? m_dataHost[hostBase + 8] : iSumZ;
            float cumSumW = (hostBase + 9 < m_dataHost.size()) ? m_dataHost[hostBase + 9] : iSumW;

            if (cumSumW >= 1.0f) {
                float theta0, phi0, kappa0;
                if (vmf_fitting::fitFromSums(cumSumX, cumSumY, cumSumZ, cumSumW, theta0, phi0, kappa0)) {
                    dataHost[base + 0] = theta0;
                    dataHost[base + 1] = phi0;
                    dataHost[base + 2] = kappa0;
                    dataHost[base + 3] = 0.0f;  // lobe 1 inactive by default
                    dataHost[base + 4] = 0.0f;
                    dataHost[base + 5] = 0.0f;
                    dataHost[base + PATH_GUIDE_MIX_WEIGHT_OFFSET] = 1.0f;
                    cellsWithData++;

                    // Two-lobe fitting: if interval sums point in a significantly
                    // different direction from cumulative, fit a second lobe.
                    // This captures bimodal distributions where indirect light
                    // arrives from multiple paths (e.g., two windows in a room).
                    float iLen = std::sqrt(iSumX*iSumX + iSumY*iSumY + iSumZ*iSumZ);
                    if (iSumW >= 2.0f && iLen > 1e-6f) {
                        float iNx = iSumX / iLen, iNy = iSumY / iLen, iNz = iSumZ / iLen;
                        // Cumulative mean direction
                        float cumLen = std::sqrt(cumSumX*cumSumX + cumSumY*cumSumY + cumSumZ*cumSumZ);
                        if (cumLen > 1e-6f) {
                            float cNx = cumSumX / cumLen, cNy = cumSumY / cumLen, cNz = cumSumZ / cumLen;
                            float cosAngle = iNx*cNx + iNy*cNy + iNz*cNz;
                            // If interval direction diverges >45° from cumulative, fit lobe 1
                            if (cosAngle < 0.707f) {
                                float theta1, phi1, kappa1;
                                if (vmf_fitting::fitFromSums(iSumX, iSumY, iSumZ, iSumW, theta1, phi1, kappa1)) {
                                    dataHost[base + 3] = theta1;
                                    dataHost[base + 4] = phi1;
                                    dataHost[base + 5] = kappa1;
                                    // Mixture weight: ratio of cumulative to total,
                                    // scaled by effective sample count
                                    float effCum = cumSumW;
                                    float effInt = iSumW / (1.0f - EMA_DECAY);  // Scale interval to match cumulative timescale
                                    float pi0 = effCum / (effCum + effInt);
                                    pi0 = std::max(0.1f, std::min(0.9f, pi0));
                                    dataHost[base + PATH_GUIDE_MIX_WEIGHT_OFFSET] = pi0;
                                }
                            }
                        }
                    }
                }
            }

            // Zero interval stats for next accumulation window
            dataHost[base + 6] = 0.0f;
            dataHost[base + 7] = 0.0f;
            dataHost[base + 8] = 0.0f;
            dataHost[base + 9] = 0.0f;
        }
        std::cout << "[PathGuide] Build: " << m_totalCells << " cells, "
                  << cellsWithData << " fitted\n";
    }

    // Initialize lastHitFrame for newly-created cells (those with lastHitFrame == 0)
    // so they get a full coarsening grace period before refinement considers them.
    if (currentFrame > 0) {
        float currentFrameF = static_cast<float>(currentFrame);
        for (size_t g = 0; g < m_totalCells; g++) {
            size_t base = g * m_entryStride;
            if (base + 11 < dataHost.size() && dataHost[base + 11] == 0.0f) {
                dataHost[base + 11] = currentFrameF;
            }
        }
    }

    // Save host-side copy: dataHost has fitted vMF params + zeroed interval stats.
    // m_dataHost retains lifetime cumulative stats for refinement decisions.
    // We need m_dataHost to have the fitted vMF params too for inspectCellAtPosition.
    for (size_t g = 0; g < m_totalCells; g++) {
        size_t base = g * m_entryStride;
        if (base + 11 >= dataHost.size()) break;
        // Copy vMF params and pi_0 from dataHost to m_dataHost
        for (uint32_t k = 0; k < 6; k++)
            m_dataHost[base + k] = dataHost[base + k];
        m_dataHost[base + PATH_GUIDE_MIX_WEIGHT_OFFSET] = dataHost[base + PATH_GUIDE_MIX_WEIGHT_OFFSET];
    }

    // Reuse existing GPU buffers if they're large enough; only reallocate when growing
    if (m_totalCells > m_allocatedCells) {
        if (m_mortonCodes) { cudaFree(m_mortonCodes); m_mortonCodes = nullptr; }
        if (m_data) { cudaFree(m_data); m_data = nullptr; }
        m_allocatedCells = 0;

        err = cudaMalloc(&m_mortonCodes, m_totalCells * sizeof(uint64_t));
        if (err != cudaSuccess) {
            std::cerr << "[PathGuideGrid] cudaMalloc morton_codes failed: " << cudaGetErrorString(err) << "\n";
            return false;
        }
        err = cudaMalloc(&m_data, static_cast<size_t>(m_totalCells) * m_entryStride * sizeof(float));
        if (err != cudaSuccess) {
            cudaFree(m_mortonCodes);
            m_mortonCodes = nullptr;
            std::cerr << "[PathGuideGrid] cudaMalloc data failed: " << cudaGetErrorString(err) << "\n";
            return false;
        }
        m_allocatedCells = m_totalCells;
    }
    err = cudaMemcpy(m_mortonCodes, mortonHost.data(), m_totalCells * sizeof(uint64_t), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(m_data);
        cudaFree(m_mortonCodes);
        m_data = nullptr;
        m_mortonCodes = nullptr;
        std::cerr << "[PathGuideGrid] cudaMemcpy morton_codes failed: " << cudaGetErrorString(err) << "\n";
        return false;
    }
    // Upload dataHost to GPU: has fitted vMF params + zeroed interval stats (offsets 6-9),
    // ready for next accumulation window via atomicAdd.
    err = cudaMemcpy(m_data, dataHost.data(),
        static_cast<size_t>(m_totalCells) * m_entryStride * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(m_data);
        cudaFree(m_mortonCodes);
        m_data = nullptr;
        m_mortonCodes = nullptr;
        std::cerr << "[PathGuideGrid] cudaMemcpy data failed: " << cudaGetErrorString(err) << "\n";
        return false;
    }
    err = cudaMemcpy(m_levelOffsetsDevice, m_levelOffsets.data(),
        (m_numLevels + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(m_data);
        cudaFree(m_mortonCodes);
        m_data = nullptr;
        m_mortonCodes = nullptr;
        std::cerr << "[PathGuideGrid] cudaMemcpy level_offsets failed: " << cudaGetErrorString(err) << "\n";
        return false;
    }
    // Build hash table for O(1) lookups on GPU
    buildAndUploadHashTable(m_hashKeys, m_hashValues, m_hashTableSize, m_hashShift,
                            m_hashAllocated, m_mortonCodesHost, m_totalCells, stream);

    // Sync hash table to render grid if async is active
    if (m_asyncInitialized) {
        GridBuffers& renderGrid = m_grids[m_renderGridIdx];
        renderGrid.hashKeys = m_hashKeys;
        renderGrid.hashValues = m_hashValues;
        renderGrid.hashTableSize = m_hashTableSize;
        renderGrid.hashShift = m_hashShift;
        renderGrid.hashAllocated = m_hashAllocated;
    }

    // Reset staging count so next accumulation starts fresh
    resetStagingCount(stream);
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

    // Build hash table on CPU
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

        cudaError_t err = cudaMalloc(&outKeys, tableSize * sizeof(uint64_t));
        if (err != cudaSuccess) {
            std::cerr << "[PathGuideGrid] Hash table cudaMalloc keys failed: " << cudaGetErrorString(err) << "\n";
            outSize = 0;
            outShift = 64;
            return false;
        }
        err = cudaMalloc(&outValues, tableSize * sizeof(uint32_t));
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

    // Upload to GPU
    if (stream) {
        cudaMemcpyAsync(outKeys, keys.data(), tableSize * sizeof(uint64_t),
            cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(outValues, values.data(), tableSize * sizeof(uint32_t),
            cudaMemcpyHostToDevice, stream);
    } else {
        cudaMemcpy(outKeys, keys.data(), tableSize * sizeof(uint64_t), cudaMemcpyHostToDevice);
        cudaMemcpy(outValues, values.data(), tableSize * sizeof(uint32_t), cudaMemcpyHostToDevice);
    }

    outSize = tableSize;
    outShift = shift;
    return true;
}

SparsePathGuideDescriptor PathGuideGrid::getDescriptor() const {
    SparsePathGuideDescriptor d = {};
    d.morton_codes = m_mortonCodes;
    d.data = m_data;
    d.level_offsets = m_levelOffsetsDevice;
    d.num_levels = m_numLevels;
    d.entry_stride = m_entryStride;
    d.base_resolution = m_config.base_resolution;
    d.per_level_scale = m_config.per_level_scale;
    d.bounds_min[0] = m_config.bounds_min[0];
    d.bounds_min[1] = m_config.bounds_min[1];
    d.bounds_min[2] = m_config.bounds_min[2];
    d.bounds_max[0] = m_config.bounds_max[0];
    d.bounds_max[1] = m_config.bounds_max[1];
    d.bounds_max[2] = m_config.bounds_max[2];
    d.hash_keys = m_hashKeys;
    d.hash_values = m_hashValues;
    d.hash_table_size = m_hashTableSize;
    d.hash_shift = m_hashShift;
    return d;
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

// Morton decode: reverse of spread3/mortonEncode
// Compact bits from interleaved position back to 21-bit value
namespace {
inline uint64_t compact3(uint64_t x) {
    x &= 0x1249249249249249ull;
    x = (x | x >> 2)  & 0x10c30c30c30c30c3ull;
    x = (x | x >> 4)  & 0x010f00f00f00f00full;
    x = (x | x >> 8)  & 0x001f0000ff0000ffull;
    x = (x | x >> 16) & 0x001f00000000ffffull;
    x = (x | x >> 32) & 0x1fffffull;
    return x;
}
} // anonymous namespace

void PathGuideGrid::mortonDecode(uint64_t morton, uint32_t& ix, uint32_t& iy, uint32_t& iz) {
    ix = static_cast<uint32_t>(compact3(morton));
    iy = static_cast<uint32_t>(compact3(morton >> 1));
    iz = static_cast<uint32_t>(compact3(morton >> 2));
}

std::vector<float> PathGuideGrid::generateEdgeVertices(uint32_t level) const {
    std::vector<float> vertices;

    if (level >= m_numLevels || m_mortonCodesHost.empty()) {
        return vertices;
    }

    // Get cell range for this level
    uint32_t start = m_levelOffsets[level];
    uint32_t end = m_levelOffsets[level + 1];
    uint32_t cellCount = end - start;

    if (cellCount == 0) {
        return vertices;
    }

    // Reserve space: 12 edges * 2 vertices * 3 floats = 72 floats per cell
    vertices.reserve(static_cast<size_t>(cellCount) * 72);

    // Use same floored resolution as build so wireframe boxes exactly bound cell regions
    float res = static_cast<float>(m_levelResolutions[level]);
    float invRes = 1.0f / res;

    float boundsExtentX = m_config.bounds_max[0] - m_config.bounds_min[0];
    float boundsExtentY = m_config.bounds_max[1] - m_config.bounds_min[1];
    float boundsExtentZ = m_config.bounds_max[2] - m_config.bounds_min[2];

    for (uint32_t i = start; i < end; i++) {
        uint64_t morton = m_mortonCodesHost[i];
        uint32_t ix, iy, iz;
        mortonDecode(morton, ix, iy, iz);

        // Compute cell AABB in world space
        float minX = m_config.bounds_min[0] + static_cast<float>(ix) * invRes * boundsExtentX;
        float maxX = m_config.bounds_min[0] + static_cast<float>(ix + 1) * invRes * boundsExtentX;
        float minY = m_config.bounds_min[1] + static_cast<float>(iy) * invRes * boundsExtentY;
        float maxY = m_config.bounds_min[1] + static_cast<float>(iy + 1) * invRes * boundsExtentY;
        float minZ = m_config.bounds_min[2] + static_cast<float>(iz) * invRes * boundsExtentZ;
        float maxZ = m_config.bounds_min[2] + static_cast<float>(iz + 1) * invRes * boundsExtentZ;

        // 8 corners of the box
        // 0: (minX, minY, minZ)  1: (maxX, minY, minZ)
        // 2: (maxX, maxY, minZ)  3: (minX, maxY, minZ)
        // 4: (minX, minY, maxZ)  5: (maxX, minY, maxZ)
        // 6: (maxX, maxY, maxZ)  7: (minX, maxY, maxZ)

        // 12 edges (each as 2 vertices):
        // Bottom face: 0-1, 1-2, 2-3, 3-0
        // Top face: 4-5, 5-6, 6-7, 7-4
        // Vertical edges: 0-4, 1-5, 2-6, 3-7

        // Helper to push a vertex
        auto pushVertex = [&](float x, float y, float z) {
            vertices.push_back(x);
            vertices.push_back(y);
            vertices.push_back(z);
        };

        // Bottom face edges
        pushVertex(minX, minY, minZ); pushVertex(maxX, minY, minZ);  // 0-1
        pushVertex(maxX, minY, minZ); pushVertex(maxX, maxY, minZ);  // 1-2
        pushVertex(maxX, maxY, minZ); pushVertex(minX, maxY, minZ);  // 2-3
        pushVertex(minX, maxY, minZ); pushVertex(minX, minY, minZ);  // 3-0

        // Top face edges
        pushVertex(minX, minY, maxZ); pushVertex(maxX, minY, maxZ);  // 4-5
        pushVertex(maxX, minY, maxZ); pushVertex(maxX, maxY, maxZ);  // 5-6
        pushVertex(maxX, maxY, maxZ); pushVertex(minX, maxY, maxZ);  // 6-7
        pushVertex(minX, maxY, maxZ); pushVertex(minX, minY, maxZ);  // 7-4

        // Vertical edges
        pushVertex(minX, minY, minZ); pushVertex(minX, minY, maxZ);  // 0-4
        pushVertex(maxX, minY, minZ); pushVertex(maxX, minY, maxZ);  // 1-5
        pushVertex(maxX, maxY, minZ); pushVertex(maxX, maxY, maxZ);  // 2-6
        pushVertex(minX, maxY, minZ); pushVertex(minX, maxY, maxZ);  // 3-7
    }

    return vertices;
}

std::vector<float> PathGuideGrid::generateEdgeVerticesAllLevels() const {
    std::vector<float> vertices;
    if (m_mortonCodesHost.empty() || m_totalCells == 0) return vertices;

    // Reserve for all cells across all levels
    vertices.reserve(static_cast<size_t>(m_totalCells) * 72);

    float boundsExtentX = m_config.bounds_max[0] - m_config.bounds_min[0];
    float boundsExtentY = m_config.bounds_max[1] - m_config.bounds_min[1];
    float boundsExtentZ = m_config.bounds_max[2] - m_config.bounds_min[2];

    for (uint32_t level = 0; level < m_numLevels; level++) {
        uint32_t start = m_levelOffsets[level];
        uint32_t end = m_levelOffsets[level + 1];
        if (start >= end) continue;

        float res = static_cast<float>(m_levelResolutions[level]);
        float invRes = 1.0f / res;

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

            auto pushVertex = [&](float x, float y, float z) {
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
    }

    return vertices;
}

bool PathGuideGrid::runRefinementPass(uint32_t currentFrame, cudaStream_t stream) {
    // Adaptive refinement: subdivide high-variance cells, coarsen unused cells
    // Reference: Müller et al., "Practical Path Guiding", EGSR 2017 (SD-tree concept)
    //
    // Cell data layout (12 floats per cell):
    //   [0-5]: vMF lobes (theta0, phi0, kappa0, theta1, phi1, kappa1)
    //   [6-11]: stats (sumX, sumY, sumZ, sumW, pi_0 (mixture weight), lastHitFrame)

    if (!m_data || m_totalCells == 0 || !m_mortonCodes) {
        return false;  // No data to refine
    }

    // Use persistent host copy instead of blocking D2H readback
    if (m_dataHost.size() != static_cast<size_t>(m_totalCells) * m_entryStride) {
        // Fallback: host copy is stale or missing, must sync and read back
        if (stream) cudaStreamSynchronize(stream);
        m_dataHost.resize(static_cast<size_t>(m_totalCells) * m_entryStride);
        cudaError_t err = cudaMemcpy(m_dataHost.data(), m_data,
            m_dataHost.size() * sizeof(float), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            std::cerr << "[PathGuideGrid] Refinement: failed to read cell data: "
                      << cudaGetErrorString(err) << "\n";
            return false;
        }
    }
    std::vector<float> dataHost = m_dataHost;

    // Track cells to add (subdivisions) and remove (coarsening)
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

    std::vector<CellKey> cellsToAdd;
    std::vector<size_t> cellsToRemove;  // indices into current cell array

    const float subdivideThreshold = m_config.subdivide_sample_threshold;
    const float varianceThreshold = m_config.subdivide_variance_threshold;
    const uint32_t coarsenThreshold = m_config.coarsen_frames_threshold;
    const uint32_t maxLevel = m_config.max_level;
    const uint32_t minLevel = m_config.min_level;

    // Diagnostic accumulators for refinement stats
    float diagMinSumW = 1e30f, diagMaxSumW = 0.0f, diagTotalSumW = 0.0f;
    float diagMinVar = 1e30f, diagMaxVar = 0.0f, diagTotalVar = 0.0f;
    uint32_t diagCellsWithData = 0;

    for (size_t g = 0; g < m_totalCells; g++) {
        // Determine which level this cell is in
        uint32_t cellLevel = 0;
        for (uint32_t l = 0; l < m_numLevels; l++) {
            if (g >= m_levelOffsets[l] && g < m_levelOffsets[l + 1]) {
                cellLevel = l;
                break;
            }
        }

        const float* stats = dataHost.data() + g * m_entryStride + PATH_GUIDE_VMF_FLOATS;
        float sumX = stats[0];
        float sumY = stats[1];
        float sumZ = stats[2];
        float sumW = stats[3];
        // stats[4] is pi_0 (mixture weight), not used for refinement
        float lastHitFrame = stats[5];

        // Coarsening: remove cells that haven't been hit for many frames AND
        // have negligible accumulated data. Cells with good fits (high sumW) are
        // retained — their guiding data is still valid even without fresh samples.
        if (cellLevel > minLevel && currentFrame > static_cast<uint32_t>(lastHitFrame) + coarsenThreshold
            && sumW < subdivideThreshold) {
            cellsToRemove.push_back(g);
            continue;
        }

        // Track stats for diagnostics
        if (sumW > 1e-9f) {
            float meanX = sumX / sumW;
            float meanY = sumY / sumW;
            float meanZ = sumZ / sumW;
            float meanLen = sqrtf(meanX * meanX + meanY * meanY + meanZ * meanZ);
            float variance = fmaxf(0.0f, 1.0f - meanLen);

            diagMinSumW = fminf(diagMinSumW, sumW);
            diagMaxSumW = fmaxf(diagMaxSumW, sumW);
            diagTotalSumW += sumW;
            diagMinVar = fminf(diagMinVar, variance);
            diagMaxVar = fmaxf(diagMaxVar, variance);
            diagTotalVar += variance;
            diagCellsWithData++;
        }

        // Subdivision: need sufficient samples and high directional variance
        if (cellLevel < maxLevel && sumW >= subdivideThreshold) {
            // Compute mean direction
            float meanX = sumX / sumW;
            float meanY = sumY / sumW;
            float meanZ = sumZ / sumW;
            float meanLen = sqrtf(meanX * meanX + meanY * meanY + meanZ * meanZ);

            // Directional variance: 1 - R̄ (where R̄ is mean resultant length)
            // Low R̄ = high variance (directions spread out), High R̄ = low variance (directions clustered)
            float variance = fmaxf(0.0f, 1.0f - meanLen);

            if (variance > varianceThreshold) {
                // Subdivide: create 8 children at next finer level
                uint64_t parentMorton = m_mortonCodesHost[g];
                uint32_t parentIx, parentIy, parentIz;
                mortonDecode(parentMorton, parentIx, parentIy, parentIz);

                // Child cells at next level have 2x the resolution per axis
                for (int dz = 0; dz < 2; dz++) {
                    for (int dy = 0; dy < 2; dy++) {
                        for (int dx = 0; dx < 2; dx++) {
                            uint32_t childIx = parentIx * 2 + dx;
                            uint32_t childIy = parentIy * 2 + dy;
                            uint32_t childIz = parentIz * 2 + dz;
                            uint64_t childMorton = mortonEncode(childIx, childIy, childIz);
                            cellsToAdd.push_back({cellLevel + 1, childMorton});
                        }
                    }
                }

                // Mark parent for removal after subdivision
                cellsToRemove.push_back(g);
            }
        }
    }

    // Print refinement diagnostics
    if (diagCellsWithData > 0) {
        float avgSumW = diagTotalSumW / diagCellsWithData;
        float avgVar = diagTotalVar / diagCellsWithData;
        std::cout << "[PathGuide] Refine check: " << diagCellsWithData << "/" << m_totalCells
                  << " cells with data | sumW: [" << diagMinSumW << ", " << diagMaxSumW
                  << "] avg=" << avgSumW << " (thresh=" << subdivideThreshold
                  << ") | var: [" << diagMinVar << ", " << diagMaxVar
                  << "] avg=" << avgVar << " (thresh=" << varianceThreshold << ")\n";
    }

    if (cellsToAdd.empty() && cellsToRemove.empty()) {
        return false;  // No changes needed
    }

    if (!cellsToAdd.empty() || !cellsToRemove.empty()) {
        std::cout << "[PathGuide] Refine: +" << cellsToAdd.size()
                  << " -" << cellsToRemove.size() << " cells\n";
    }

    // Build new cell set: existing cells minus removed plus added
    std::unordered_set<size_t> removeSet(cellsToRemove.begin(), cellsToRemove.end());

    std::vector<CellKey> newCells;
    newCells.reserve(m_totalCells - cellsToRemove.size() + cellsToAdd.size());

    // Keep cells that aren't being removed
    for (size_t g = 0; g < m_totalCells; g++) {
        if (removeSet.count(g) == 0) {
            uint32_t cellLevel = 0;
            for (uint32_t l = 0; l < m_numLevels; l++) {
                if (g >= m_levelOffsets[l] && g < m_levelOffsets[l + 1]) {
                    cellLevel = l;
                    break;
                }
            }
            newCells.push_back({cellLevel, m_mortonCodesHost[g]});
        }
    }

    // Add new cells
    for (const auto& cell : cellsToAdd) {
        newCells.push_back(cell);
    }

    // Sort and deduplicate
    std::sort(newCells.begin(), newCells.end());
    newCells.erase(std::unique(newCells.begin(), newCells.end()), newCells.end());

    // Save old arrays for carry-forward BEFORE overwriting m_levelOffsets/m_mortonCodesHost
    std::vector<uint64_t> oldMorton = m_mortonCodesHost;
    std::vector<uint32_t> oldOffsets = m_levelOffsets;

    // Rebuild level offsets
    m_levelOffsets.assign(m_numLevels + 1, 0u);
    for (const auto& k : newCells)
        m_levelOffsets[k.level + 1]++;
    for (uint32_t l = 0; l < m_numLevels; l++)
        m_levelOffsets[l + 1] += m_levelOffsets[l];

    m_totalCells = static_cast<uint32_t>(newCells.size());
    if (m_totalCells == 0) {
        m_mortonCodesHost.clear();
        m_hashTableSize = 0;
        m_hashShift = 64;
        cudaMemcpyAsync(m_levelOffsetsDevice, m_levelOffsets.data(),
            (m_numLevels + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice, stream);
        return true;
    }

    // Build new arrays
    std::vector<uint64_t> mortonHost(m_totalCells);
    std::vector<float> newDataHost(static_cast<size_t>(m_totalCells) * m_entryStride, 0.0f);
    std::vector<uint32_t> levelCellCount(m_numLevels, 0u);

    for (const auto& k : newCells) {
        uint32_t base = m_levelOffsets[k.level];
        uint32_t idx = base + levelCellCount[k.level]++;
        mortonHost[idx] = k.morton;
    }

    // Carry forward data for surviving cells (two-pointer merge per level)
    // New child cells from subdivision start zeroed; surviving cells retain their stats.
    if (!oldMorton.empty() && !dataHost.empty()) {
        for (uint32_t lev = 0; lev < m_numLevels && lev < static_cast<uint32_t>(oldOffsets.size()) - 1; lev++) {
            uint32_t oldStart = oldOffsets[lev];
            uint32_t oldEnd = oldOffsets[lev + 1];
            uint32_t newStart = m_levelOffsets[lev];
            uint32_t newEnd = m_levelOffsets[lev + 1];
            uint32_t oi = oldStart, ni = newStart;
            while (oi < oldEnd && ni < newEnd) {
                if (oldMorton[oi] < mortonHost[ni]) {
                    oi++;
                } else if (oldMorton[oi] > mortonHost[ni]) {
                    ni++;
                } else {
                    // Same cell in both old and new — copy all data
                    size_t srcOff = static_cast<size_t>(oi) * m_entryStride;
                    size_t dstOff = static_cast<size_t>(ni) * m_entryStride;
                    std::memcpy(&newDataHost[dstOff], &dataHost[srcOff], m_entryStride * sizeof(float));
                    oi++;
                    ni++;
                }
            }
        }
    }

    m_mortonCodesHost = mortonHost;

    // Initialize lastHitFrame = currentFrame for newly-created cells (those with
    // lastHitFrame == 0 after carry-forward). Without this, child cells from subdivision
    // get immediately coarsened on the next refinement because currentFrame > 0 + threshold.
    if (currentFrame > 0) {
        float currentFrameF = static_cast<float>(currentFrame);
        for (size_t g = 0; g < m_totalCells; g++) {
            size_t base = g * m_entryStride;
            if (base + 11 < newDataHost.size() && newDataHost[base + 11] == 0.0f) {
                newDataHost[base + 11] = currentFrameF;
            }
        }
    }

    // Save host-side copy for next build/refinement
    m_dataHost = newDataHost;

    // Reuse existing GPU buffers if large enough; only reallocate when growing
    if (m_totalCells > m_allocatedCells) {
        if (m_mortonCodes) { cudaFree(m_mortonCodes); m_mortonCodes = nullptr; }
        if (m_data) { cudaFree(m_data); m_data = nullptr; }
        m_allocatedCells = 0;

        cudaError_t err = cudaMalloc(&m_mortonCodes, m_totalCells * sizeof(uint64_t));
        if (err != cudaSuccess) {
            std::cerr << "[PathGuideGrid] Refinement: cudaMalloc morton failed: "
                      << cudaGetErrorString(err) << "\n";
            return false;
        }

        err = cudaMalloc(&m_data, static_cast<size_t>(m_totalCells) * m_entryStride * sizeof(float));
        if (err != cudaSuccess) {
            cudaFree(m_mortonCodes);
            m_mortonCodes = nullptr;
            std::cerr << "[PathGuideGrid] Refinement: cudaMalloc data failed: "
                      << cudaGetErrorString(err) << "\n";
            return false;
        }
        m_allocatedCells = m_totalCells;
    }

    // Use async copies on the render stream to avoid blocking ALL streams.
    // Sync cudaMemcpy uses the default stream which synchronizes every stream on the device.
    // These pageable-to-device copies still briefly block the CPU (CUDA runtime behavior),
    // but they don't stall unrelated streams like the readback stream.
    cudaMemcpyAsync(m_mortonCodes, mortonHost.data(), m_totalCells * sizeof(uint64_t),
        cudaMemcpyHostToDevice, stream);

    // Upload with zeroed interval stats (offsets 6-9) so the GPU starts fresh.
    // m_dataHost contains lifetime cumulative sums at [6-9] which must NOT be
    // uploaded — otherwise the GPU atomicAdds on top of them and the next
    // buildFromStaging reads them back as interval data, compounding the totals.
    {
        std::vector<float> gpuUpload = m_dataHost;
        for (size_t g = 0; g < m_totalCells; g++) {
            size_t base = g * m_entryStride;
            gpuUpload[base + 6] = 0.0f;
            gpuUpload[base + 7] = 0.0f;
            gpuUpload[base + 8] = 0.0f;
            gpuUpload[base + 9] = 0.0f;
        }
        cudaMemcpyAsync(m_data, gpuUpload.data(),
            static_cast<size_t>(m_totalCells) * m_entryStride * sizeof(float),
            cudaMemcpyHostToDevice, stream);
    }

    cudaMemcpyAsync(m_levelOffsetsDevice, m_levelOffsets.data(),
        (m_numLevels + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice, stream);

    // Rebuild hash table after refinement changed the grid structure
    buildAndUploadHashTable(m_hashKeys, m_hashValues, m_hashTableSize, m_hashShift,
                            m_hashAllocated, m_mortonCodesHost, m_totalCells, stream);

    // When async is initialized, the m_* pointers may have been reallocated.
    // Sync them back to the render grid's GridBuffers so getRenderDescriptor()
    // returns the updated pointers instead of stale (freed) ones.
    if (m_asyncInitialized) {
        GridBuffers& renderGrid = m_grids[m_renderGridIdx];
        renderGrid.mortonCodes = m_mortonCodes;
        renderGrid.data = m_data;
        renderGrid.totalCells = m_totalCells;
        renderGrid.allocatedCells = m_allocatedCells;
        renderGrid.levelOffsetsDevice = m_levelOffsetsDevice;
        renderGrid.hashKeys = m_hashKeys;
        renderGrid.hashValues = m_hashValues;
        renderGrid.hashTableSize = m_hashTableSize;
        renderGrid.hashShift = m_hashShift;
        renderGrid.hashAllocated = m_hashAllocated;
    }

    return true;
}

PathGuideGrid::CellInspectionResult PathGuideGrid::inspectCellAtPosition(float px, float py, float pz) const {
    CellInspectionResult result = {};

    if (m_totalCells == 0 || m_mortonCodesHost.empty() || !m_data) {
        return result;
    }

    // Normalize world position to [0,1]
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

            // Read cell data from host copy (avoids blocking GPU readback)
            if (m_entryStride <= 12 && !m_dataHost.empty()) {
                size_t offset = globalIdx * m_entryStride;
                if (offset + m_entryStride <= m_dataHost.size()) {
                    std::memcpy(result.data, &m_dataHost[offset], m_entryStride * sizeof(float));
                }
            }

            // Compute AABB
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
// Async readback pipeline
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

    // Create events (timing disabled for lowest overhead)
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
        // Copy current level offsets to both grids
        cudaMemcpy(m_grids[i].levelOffsetsDevice, m_levelOffsets.data(),
            (m_numLevels + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice);
    }

    // Free the original m_levelOffsetsDevice from init() — both grids now have their own copies.
    // Redirect m_levelOffsetsDevice to grid[0]'s copy for compatibility with getDescriptor().
    if (m_levelOffsetsDevice &&
        m_levelOffsetsDevice != m_grids[0].levelOffsetsDevice &&
        m_levelOffsetsDevice != m_grids[1].levelOffsetsDevice) {
        cudaFree(m_levelOffsetsDevice);
    }
    m_levelOffsetsDevice = m_grids[0].levelOffsetsDevice;

    // Copy existing single-buffer state into grid[0] (render grid)
    m_grids[0].mortonCodes = m_mortonCodes;
    m_grids[0].data = m_data;
    m_grids[0].totalCells = m_totalCells;
    m_grids[0].allocatedCells = m_allocatedCells;

    // Build grid[1] starts empty (will be populated on first finishBuildFromReadback)
    m_grids[1].mortonCodes = nullptr;
    m_grids[1].data = nullptr;
    m_grids[1].totalCells = 0;
    m_grids[1].allocatedCells = 0;

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
    if (m_pinnedGpuData) { cudaFreeHost(m_pinnedGpuData); m_pinnedGpuData = nullptr; }
    m_pinnedGpuDataCapacity = 0;

    // Free grid buffers. After swapGrids(), m_mortonCodes/m_data/m_levelOffsetsDevice
    // alias one of the grid buffers. We collect all unique device pointers to free,
    // avoiding double-frees.
    {
        // Collect unique pointers to free
        void* toFree[10] = {};
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
        m_grids[i].mortonCodes = nullptr;
        m_grids[i].data = nullptr;
        m_grids[i].levelOffsetsDevice = nullptr;
        m_grids[i].totalCells = 0;
        m_grids[i].allocatedCells = 0;
        m_grids[i].hashKeys = nullptr;
        m_grids[i].hashValues = nullptr;
        m_grids[i].hashTableSize = 0;
        m_grids[i].hashShift = 0;
        m_grids[i].hashAllocated = 0;
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

    const GridBuffers& renderGrid = m_grids[m_renderGridIdx];

    // Mark render + staging writes complete on renderStream
    cudaEventRecord(m_renderDoneEvent, renderStream);

    // Readback stream waits for render to finish
    cudaStreamWaitEvent(m_readbackStream, m_renderDoneEvent, 0);

    // Async D2H: staging count
    cudaMemcpyAsync(m_pinnedStagingCount, m_stagingCount, sizeof(uint32_t),
        cudaMemcpyDeviceToHost, m_readbackStream);

    // Async D2H: staging buffer
    size_t stagingBytes = m_pinnedStagingBufferCapacity * sizeof(uint32_t);
    cudaMemcpyAsync(m_pinnedStagingBuffer, m_stagingBuffer, stagingBytes,
        cudaMemcpyDeviceToHost, m_readbackStream);

    // Async D2H: render grid's cell data (if any)
    if (renderGrid.data && renderGrid.totalCells > 0) {
        size_t gpuDataFloats = static_cast<size_t>(renderGrid.totalCells) * m_entryStride;
        // Grow pinned buffer if needed
        if (gpuDataFloats > m_pinnedGpuDataCapacity) {
            if (m_pinnedGpuData) cudaFreeHost(m_pinnedGpuData);
            m_pinnedGpuDataCapacity = gpuDataFloats + gpuDataFloats / 4;  // 25% headroom
            cudaError_t err = cudaMallocHost(&m_pinnedGpuData, m_pinnedGpuDataCapacity * sizeof(float));
            if (err != cudaSuccess) {
                std::cerr << "[PathGuideGrid] Failed to grow pinned GPU data buffer: " << cudaGetErrorString(err) << "\n";
                m_pinnedGpuData = nullptr;
                m_pinnedGpuDataCapacity = 0;
            }
        }
        if (m_pinnedGpuData) {
            cudaMemcpyAsync(m_pinnedGpuData, renderGrid.data, gpuDataFloats * sizeof(float),
                cudaMemcpyDeviceToHost, m_readbackStream);
        }
    }

    // Reset staging count on the readback stream AFTER the D2H copies complete,
    // so the count is not zeroed before the readback reads it (WAR hazard fix).
    cudaMemsetAsync(m_stagingCount, 0, sizeof(uint32_t), m_readbackStream);

    // Mark readback + staging reset complete
    cudaEventRecord(m_readbackDoneEvent, m_readbackStream);

    // Render stream must wait for the staging count reset before the next frame's
    // staging writes, otherwise atomicAdd would increment a stale count.
    cudaStreamWaitEvent(renderStream, m_readbackDoneEvent, 0);

    m_pendingCurrentFrame = currentFrame;
    m_asyncState = AsyncState::ReadbackInFlight;
}

bool PathGuideGrid::pollAsyncReadback() {
    if (m_asyncState != AsyncState::ReadbackInFlight) return false;
    cudaError_t err = cudaEventQuery(m_readbackDoneEvent);
    if (err == cudaSuccess) {
        m_asyncState = AsyncState::ReadbackReady;
        return true;
    }
    if (err != cudaErrorNotReady) {
        // Real error (not just "still in flight") — log and abort the build
        std::cerr << "[PathGuideGrid] pollAsyncReadback error: " << cudaGetErrorString(err) << "\n";
        m_asyncState = AsyncState::Idle;
    }
    return false;
}

bool PathGuideGrid::finishBuildFromReadback() {
    if (m_asyncState != AsyncState::ReadbackReady) return false;

    const GridBuffers& renderGrid = m_grids[m_renderGridIdx];
    GridBuffers& buildGrid = m_grids[m_buildGridIdx];
    uint32_t currentFrame = m_pendingCurrentFrame;

    // Read count from pinned memory
    uint32_t count = *m_pinnedStagingCount;
    if (count > m_stagingCapacity) count = m_stagingCapacity;
    uint32_t numEntries = count;

    // Deduplicate staging entries using a hash set — O(n) instead of O(n log n).
    // Most staging entries are duplicates (many rays hit the same cell), so this
    // reduces the sort from 500K+ entries down to typically 2K-10K unique cells.
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

    // Pack (level, morton) into a single uint64_t for fast hashing
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
        uint64_t morton = mortonEncode(ix, iy, iz);
        uniqueSet.insert((uint64_t(level) << 60) | morton);
    }

    // Merge existing cells (cumulative grid)
    if (!m_mortonCodesHost.empty() && m_totalCells > 0) {
        for (uint32_t lev = 0; lev < m_numLevels; lev++) {
            uint32_t start = m_levelOffsets[lev];
            uint32_t end = m_levelOffsets[lev + 1];
            for (uint32_t i = start; i < end; i++) {
                uniqueSet.insert((uint64_t(lev) << 60) | m_mortonCodesHost[i]);
            }
        }
    }

    // Convert to sorted vector (only unique cells, typically 100x fewer than staging entries)
    std::vector<CellKey> keys;
    keys.reserve(uniqueSet.size());
    for (uint64_t packed : uniqueSet) {
        uint32_t level = static_cast<uint32_t>(packed >> 60);
        uint64_t morton = packed & ((1ull << 60) - 1);
        keys.push_back({ level, morton });
    }
    std::sort(keys.begin(), keys.end());

    // Save previous grid state
    std::vector<uint64_t> prevMorton = m_mortonCodesHost;
    std::vector<uint32_t> prevOffsets = m_levelOffsets;
    uint32_t prevTotalCells = renderGrid.totalCells;

    // Per-level counts and prefix sum
    m_levelOffsets.assign(m_numLevels + 1, 0u);
    for (const auto& k : keys)
        m_levelOffsets[k.level + 1]++;
    for (uint32_t l = 0; l < m_numLevels; l++)
        m_levelOffsets[l + 1] += m_levelOffsets[l];

    m_totalCells = static_cast<uint32_t>(keys.size());
    if (m_totalCells == 0) {
        m_mortonCodesHost.clear();
        cudaMemcpyAsync(buildGrid.levelOffsetsDevice, m_levelOffsets.data(),
            (m_numLevels + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice, m_readbackStream);
        buildGrid.totalCells = 0;
        m_asyncState = AsyncState::Idle;
        return true;
    }

    std::vector<uint64_t> mortonHost(m_totalCells);
    std::vector<float> dataHost(static_cast<size_t>(m_totalCells) * m_entryStride, 0.0f);
    std::vector<uint32_t> levelCellCount(m_numLevels, 0u);
    for (const auto& k : keys) {
        uint32_t base = m_levelOffsets[k.level];
        uint32_t idx = base + levelCellCount[k.level]++;
        mortonHost[idx] = k.morton;
    }

    m_mortonCodesHost = mortonHost;

    // Carry forward GPU data from pinned readback (same merge logic as buildFromStaging)
    if (m_pinnedGpuData && prevTotalCells > 0) {
        for (uint32_t lev = 0; lev < m_numLevels && lev < static_cast<uint32_t>(prevOffsets.size()) - 1; lev++) {
            uint32_t pStart = prevOffsets[lev];
            uint32_t pEnd = prevOffsets[lev + 1];
            uint32_t nStart = m_levelOffsets[lev];
            uint32_t nEnd = m_levelOffsets[lev + 1];
            uint32_t pi2 = pStart, ni2 = nStart;
            while (pi2 < pEnd && ni2 < nEnd) {
                if (prevMorton[pi2] < mortonHost[ni2]) { pi2++; }
                else if (prevMorton[pi2] > mortonHost[ni2]) { ni2++; }
                else {
                    size_t srcOff = static_cast<size_t>(pi2) * m_entryStride;
                    size_t dstOff = static_cast<size_t>(ni2) * m_entryStride;
                    std::memcpy(&dataHost[dstOff], &m_pinnedGpuData[srcOff], m_entryStride * sizeof(float));
                    pi2++; ni2++;
                }
            }
        }
    }

    // Carry forward lifetime totals from m_dataHost
    {
        std::vector<float> newDataHost(static_cast<size_t>(m_totalCells) * m_entryStride, 0.0f);
        if (!m_dataHost.empty() && !prevMorton.empty()) {
            for (uint32_t lev = 0; lev < m_numLevels && lev < static_cast<uint32_t>(prevOffsets.size()) - 1; lev++) {
                uint32_t pStart = prevOffsets[lev];
                uint32_t pEnd = prevOffsets[lev + 1];
                uint32_t nStart = m_levelOffsets[lev];
                uint32_t nEnd = m_levelOffsets[lev + 1];
                uint32_t pi3 = pStart, ni3 = nStart;
                while (pi3 < pEnd && ni3 < nEnd) {
                    if (prevMorton[pi3] < mortonHost[ni3]) { pi3++; }
                    else if (prevMorton[pi3] > mortonHost[ni3]) { ni3++; }
                    else {
                        size_t srcOff = static_cast<size_t>(pi3) * m_entryStride;
                        size_t dstOff = static_cast<size_t>(ni3) * m_entryStride;
                        std::memcpy(&newDataHost[dstOff], &m_dataHost[srcOff], m_entryStride * sizeof(float));
                        pi3++; ni3++;
                    }
                }
            }
        }
        m_dataHost = std::move(newDataHost);
    }

    // Accumulate interval sums into lifetime totals with EMA decay, then fit
    // from cumulative sums (same logic as buildFromStaging — see comments there).
    constexpr float EMA_DECAY = 0.7f;
    uint32_t cellsWithData = 0;
    if (m_totalCells > 0 && m_entryStride >= 12) {
        for (size_t g = 0; g < m_totalCells; g++) {
            size_t base = g * m_entryStride;
            if (base + 11 >= dataHost.size()) break;

            float iSumX = dataHost[base + 6];
            float iSumY = dataHost[base + 7];
            float iSumZ = dataHost[base + 8];
            float iSumW = dataHost[base + 9];

            // Accumulate into lifetime totals with EMA decay
            size_t hostBase = g * m_entryStride;
            if (hostBase + 11 < m_dataHost.size()) {
                m_dataHost[hostBase + 6] = EMA_DECAY * m_dataHost[hostBase + 6] + iSumX;
                m_dataHost[hostBase + 7] = EMA_DECAY * m_dataHost[hostBase + 7] + iSumY;
                m_dataHost[hostBase + 8] = EMA_DECAY * m_dataHost[hostBase + 8] + iSumZ;
                m_dataHost[hostBase + 9] = EMA_DECAY * m_dataHost[hostBase + 9] + iSumW;
                m_dataHost[hostBase + 11] = dataHost[base + 11];
            }

            // Fit from cumulative sums
            float cumSumX = (hostBase + 9 < m_dataHost.size()) ? m_dataHost[hostBase + 6] : iSumX;
            float cumSumY = (hostBase + 9 < m_dataHost.size()) ? m_dataHost[hostBase + 7] : iSumY;
            float cumSumZ = (hostBase + 9 < m_dataHost.size()) ? m_dataHost[hostBase + 8] : iSumZ;
            float cumSumW = (hostBase + 9 < m_dataHost.size()) ? m_dataHost[hostBase + 9] : iSumW;

            if (cumSumW >= 1.0f) {
                float theta0, phi0, kappa0;
                if (vmf_fitting::fitFromSums(cumSumX, cumSumY, cumSumZ, cumSumW, theta0, phi0, kappa0)) {
                    dataHost[base + 0] = theta0;
                    dataHost[base + 1] = phi0;
                    dataHost[base + 2] = kappa0;
                    dataHost[base + 3] = 0.0f;
                    dataHost[base + 4] = 0.0f;
                    dataHost[base + 5] = 0.0f;
                    dataHost[base + PATH_GUIDE_MIX_WEIGHT_OFFSET] = 1.0f;
                    cellsWithData++;

                    // Two-lobe fitting (same logic as buildFromStaging)
                    float iLen = std::sqrt(iSumX*iSumX + iSumY*iSumY + iSumZ*iSumZ);
                    if (iSumW >= 2.0f && iLen > 1e-6f) {
                        float iNx = iSumX / iLen, iNy = iSumY / iLen, iNz = iSumZ / iLen;
                        float cumLen = std::sqrt(cumSumX*cumSumX + cumSumY*cumSumY + cumSumZ*cumSumZ);
                        if (cumLen > 1e-6f) {
                            float cNx = cumSumX / cumLen, cNy = cumSumY / cumLen, cNz = cumSumZ / cumLen;
                            float cosAngle = iNx*cNx + iNy*cNy + iNz*cNz;
                            if (cosAngle < 0.707f) {
                                float theta1, phi1, kappa1;
                                if (vmf_fitting::fitFromSums(iSumX, iSumY, iSumZ, iSumW, theta1, phi1, kappa1)) {
                                    dataHost[base + 3] = theta1;
                                    dataHost[base + 4] = phi1;
                                    dataHost[base + 5] = kappa1;
                                    float effCum = cumSumW;
                                    float effInt = iSumW / (1.0f - EMA_DECAY);
                                    float pi0 = effCum / (effCum + effInt);
                                    pi0 = std::max(0.1f, std::min(0.9f, pi0));
                                    dataHost[base + PATH_GUIDE_MIX_WEIGHT_OFFSET] = pi0;
                                }
                            }
                        }
                    }
                }
            }

            // Zero interval stats for upload
            dataHost[base + 6] = 0.0f;
            dataHost[base + 7] = 0.0f;
            dataHost[base + 8] = 0.0f;
            dataHost[base + 9] = 0.0f;
        }
        std::cout << "[PathGuide] Async build: " << m_totalCells << " cells, "
                  << cellsWithData << " fitted\n";
    }

    // Initialize lastHitFrame for newly-created cells
    if (currentFrame > 0) {
        float currentFrameF = static_cast<float>(currentFrame);
        for (size_t g = 0; g < m_totalCells; g++) {
            size_t base = g * m_entryStride;
            if (base + 11 < dataHost.size() && dataHost[base + 11] == 0.0f) {
                dataHost[base + 11] = currentFrameF;
            }
        }
    }

    // Save vMF params to m_dataHost
    for (size_t g = 0; g < m_totalCells; g++) {
        size_t base = g * m_entryStride;
        if (base + 11 >= dataHost.size()) break;
        for (uint32_t k = 0; k < 6; k++)
            m_dataHost[base + k] = dataHost[base + k];
        m_dataHost[base + PATH_GUIDE_MIX_WEIGHT_OFFSET] = dataHost[base + PATH_GUIDE_MIX_WEIGHT_OFFSET];
    }

    // Upload to build grid (async on readback stream)
    // Allocate with 2x headroom to minimize cudaMalloc/cudaFree frequency.
    // On WDDM, cudaMalloc/cudaFree cause implicit device-wide synchronization
    // that stalls the render stream even when called from a background thread.
    if (m_totalCells > buildGrid.allocatedCells) {
        uint32_t newCapacity = std::max(m_totalCells * 2u, 8192u);

        uint64_t* newMorton = nullptr;
        float* newData = nullptr;
        cudaError_t err = cudaMalloc(&newMorton, newCapacity * sizeof(uint64_t));
        if (err != cudaSuccess) {
            std::cerr << "[PathGuideGrid] Async: cudaMalloc morton failed: " << cudaGetErrorString(err) << "\n";
            m_asyncState = AsyncState::Idle;
            return false;
        }
        err = cudaMalloc(&newData, static_cast<size_t>(newCapacity) * m_entryStride * sizeof(float));
        if (err != cudaSuccess) {
            cudaFree(newMorton);
            std::cerr << "[PathGuideGrid] Async: cudaMalloc data failed: " << cudaGetErrorString(err) << "\n";
            m_asyncState = AsyncState::Idle;
            return false;
        }

        // Free old after successful allocation
        if (buildGrid.mortonCodes) cudaFree(buildGrid.mortonCodes);
        if (buildGrid.data) cudaFree(buildGrid.data);
        buildGrid.mortonCodes = newMorton;
        buildGrid.data = newData;
        buildGrid.allocatedCells = newCapacity;
    }

    cudaMemcpyAsync(buildGrid.mortonCodes, mortonHost.data(),
        m_totalCells * sizeof(uint64_t), cudaMemcpyHostToDevice, m_readbackStream);
    cudaMemcpyAsync(buildGrid.data, dataHost.data(),
        static_cast<size_t>(m_totalCells) * m_entryStride * sizeof(float),
        cudaMemcpyHostToDevice, m_readbackStream);
    cudaMemcpyAsync(buildGrid.levelOffsetsDevice, m_levelOffsets.data(),
        (m_numLevels + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice, m_readbackStream);

    // Build hash table for O(1) lookups on GPU (uploaded to build grid)
    buildAndUploadHashTable(buildGrid.hashKeys, buildGrid.hashValues,
                            buildGrid.hashTableSize, buildGrid.hashShift,
                            buildGrid.hashAllocated, mortonHost, m_totalCells, m_readbackStream);

    buildGrid.totalCells = m_totalCells;

    // Record event after uploads so swapGrids can make the render stream wait.
    // No CPU-side sync needed — the GPU handles ordering via events.
    cudaEventRecord(m_readbackDoneEvent, m_readbackStream);

    m_asyncState = AsyncState::Idle;
    return true;
}

void PathGuideGrid::swapGrids(cudaStream_t renderStream) {
    if (!m_asyncInitialized) return;

    // Ensure the render stream waits for the build grid's H2D uploads to finish
    // before it reads the new grid. GPU-side ordering only — no CPU stall.
    if (renderStream && m_readbackDoneEvent) {
        cudaStreamWaitEvent(renderStream, m_readbackDoneEvent, 0);
    }

    std::swap(m_renderGridIdx, m_buildGridIdx);

    // Update the single-buffer pointers to reflect the new render grid
    // (for compatibility with getDescriptor(), edge generation, etc.)
    const GridBuffers& newRender = m_grids[m_renderGridIdx];
    m_mortonCodes = newRender.mortonCodes;
    m_data = newRender.data;
    m_levelOffsetsDevice = newRender.levelOffsetsDevice;
    m_totalCells = newRender.totalCells;
    m_allocatedCells = newRender.allocatedCells;
    m_hashKeys = newRender.hashKeys;
    m_hashValues = newRender.hashValues;
    m_hashTableSize = newRender.hashTableSize;
    m_hashShift = newRender.hashShift;
    m_hashAllocated = newRender.hashAllocated;
}

} // namespace spectra
