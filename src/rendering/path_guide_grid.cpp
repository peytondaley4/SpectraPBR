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

    // Training staging: 9 floats per entry (level, ix, iy, iz, dir_x, dir_y, dir_z, weight, frame)
    uint32_t trainCap = config.training_capacity > 0 ? config.training_capacity : config.staging_capacity;
    err = cudaMalloc(&m_trainingBuffer, trainCap * 9 * sizeof(float));
    if (err != cudaSuccess) {
        cudaFree(m_stagingCount);
        cudaFree(m_stagingBuffer);
        m_stagingBuffer = nullptr;
        m_stagingCount = nullptr;
        std::cerr << "[PathGuideGrid] cudaMalloc training buffer failed: " << cudaGetErrorString(err) << "\n";
        return false;
    }
    err = cudaMalloc(&m_trainingCount, sizeof(uint32_t));
    if (err != cudaSuccess) {
        cudaFree(m_trainingBuffer);
        cudaFree(m_stagingCount);
        cudaFree(m_stagingBuffer);
        m_trainingBuffer = nullptr;
        m_stagingBuffer = nullptr;
        m_stagingCount = nullptr;
        std::cerr << "[PathGuideGrid] cudaMalloc training count failed: " << cudaGetErrorString(err) << "\n";
        return false;
    }
    m_trainingCapacity = trainCap;
    cudaMemset(m_trainingCount, 0, sizeof(uint32_t));

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
    if (m_mortonCodes) { cudaFree(m_mortonCodes); m_mortonCodes = nullptr; }
    if (m_data) { cudaFree(m_data); m_data = nullptr; }
    if (m_levelOffsetsDevice) { cudaFree(m_levelOffsetsDevice); m_levelOffsetsDevice = nullptr; }
    if (m_stagingBuffer) { cudaFree(m_stagingBuffer); m_stagingBuffer = nullptr; }
    if (m_stagingCount) { cudaFree(m_stagingCount); m_stagingCount = nullptr; }
    if (m_trainingBuffer) { cudaFree(m_trainingBuffer); m_trainingBuffer = nullptr; }
    if (m_trainingCount) { cudaFree(m_trainingCount); m_trainingCount = nullptr; }
    m_levelOffsets.clear();
    m_numLevels = 0;
    m_entryStride = 0;
    m_totalCells = 0;
    m_allocatedCells = 0;
    m_stagingCapacity = 0;
    m_trainingCapacity = 0;
}

void PathGuideGrid::clear(cudaStream_t stream) {
    if (!m_data || m_totalCells == 0) return;
    size_t numBytes = static_cast<size_t>(m_totalCells) * static_cast<size_t>(m_entryStride) * sizeof(float);
    if (stream)
        cudaMemsetAsync(m_data, 0, numBytes, stream);
    else
        cudaMemset(m_data, 0, numBytes);
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

void PathGuideGrid::resetTrainingCount(cudaStream_t stream) {
    if (!m_trainingCount) return;
    if (stream)
        cudaMemsetAsync(m_trainingCount, 0, sizeof(uint32_t), stream);
    else
        cudaMemset(m_trainingCount, 0, sizeof(uint32_t));
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
    if (count == 0) {
        // No new staging data this build — keep existing grid instead of clearing.
        // Clearing caused cells to disappear whenever a build frame had zero hits (e.g. camera
        // pointed at sky), and only some cells would come back on later builds.
        return true;
    }
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
    err = cudaMemcpy(staging.data(), m_stagingBuffer, numUints * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "[PathGuideGrid] read staging buffer failed: " << cudaGetErrorString(err) << "\n";
        return false;
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

    // Save previous grid BEFORE overwriting m_levelOffsets (needed for data carry-forward)
    // Uses persistent host copy (m_dataHost) to avoid blocking D2H readback
    std::vector<uint64_t> prevMorton = m_mortonCodesHost;
    std::vector<uint32_t> prevOffsets = m_levelOffsets;  // OLD offsets
    uint32_t prevTotalCells = m_mortonCodesHost.empty() ? 0 : static_cast<uint32_t>(m_mortonCodesHost.size());
    std::vector<float> prevData = m_dataHost;  // Already on host, no GPU readback needed

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

    // Carry forward existing cell data (vMF lobes + stats) for cells that persist
    if (prevTotalCells > 0 && !prevData.empty()) {
        for (uint32_t lev = 0; lev < m_numLevels && lev < static_cast<uint32_t>(prevOffsets.size()) - 1; lev++) {
            uint32_t prevStart = prevOffsets[lev];
            uint32_t prevEnd = prevOffsets[lev + 1];
            uint32_t newStart = m_levelOffsets[lev];
            uint32_t newEnd = m_levelOffsets[lev + 1];
            // Both are sorted by morton within level — merge with two pointers
            uint32_t pi = prevStart, ni = newStart;
            while (pi < prevEnd && ni < newEnd) {
                if (prevMorton[pi] < mortonHost[ni]) {
                    pi++;
                } else if (prevMorton[pi] > mortonHost[ni]) {
                    ni++;
                } else {
                    // Same cell exists in both — copy data
                    size_t srcOff = static_cast<size_t>(pi) * m_entryStride;
                    size_t dstOff = static_cast<size_t>(ni) * m_entryStride;
                    std::memcpy(&dataHost[dstOff], &prevData[srcOff], m_entryStride * sizeof(float));
                    pi++;
                    ni++;
                }
            }
        }
    }

    // Keep CPU copy for edge generation
    m_mortonCodesHost = mortonHost;

    // Merge training data into vMF lobes (Practical Path Guiding: fit direction distribution per cell)
    // Note: caller already synchronized the stream before calling buildFromStaging
    if (m_trainingBuffer && m_trainingCount && m_trainingCapacity > 0 && m_totalCells > 0) {
        uint32_t trainCount = 0;
        err = cudaMemcpy(&trainCount, m_trainingCount, sizeof(uint32_t), cudaMemcpyDeviceToHost);
        // Training count logged in build summary below

        if (err == cudaSuccess && trainCount > 0) {
            // Clamp to capacity
            if (trainCount > m_trainingCapacity) {
                trainCount = m_trainingCapacity;
            }

            size_t trainFloats = static_cast<size_t>(trainCount) * 9;
            std::vector<float> training(trainFloats);
            err = cudaMemcpy(training.data(), m_trainingBuffer, trainFloats * sizeof(float), cudaMemcpyDeviceToHost);
            if (err != cudaSuccess) {
                std::cerr << "[PathGuideGrid] Failed to read training buffer: " << cudaGetErrorString(err) << std::endl;
            } else {

                // Per-cell sample collection for 2-lobe vMF fitting
                struct CellSample { float dx, dy, dz, w; };
                std::vector<std::vector<CellSample>> cellSamples(m_totalCells);
                std::vector<float> sumX(m_totalCells, 0.0f), sumY(m_totalCells, 0.0f), sumZ(m_totalCells, 0.0f);
                std::vector<float> sumW(m_totalCells, 0.0f);
                std::vector<uint32_t> lastFrame(m_totalCells, 0);
                uint32_t matchedSamples = 0;

                for (size_t t = 0; t + 9 <= trainFloats; t += 9) {
                    uint32_t lev = static_cast<uint32_t>(training[t + 0]);
                    uint32_t ix  = static_cast<uint32_t>(training[t + 1]);
                    uint32_t iy  = static_cast<uint32_t>(training[t + 2]);
                    uint32_t iz  = static_cast<uint32_t>(training[t + 3]);
                    float dx = training[t + 4], dy = training[t + 5], dz = training[t + 6], w = training[t + 7];
                    uint32_t frame = static_cast<uint32_t>(training[t + 8]);
                    if (lev >= m_numLevels || w <= 0.0f) continue;
                    uint32_t resU = m_levelResolutions[lev];
                    if (ix >= resU) ix = resU - 1;
                    if (iy >= resU) iy = resU - 1;
                    if (iz >= resU) iz = resU - 1;
                    uint64_t morton = mortonEncode(ix, iy, iz);
                    uint32_t start = m_levelOffsets[lev], end = m_levelOffsets[lev + 1];
                    auto it = std::lower_bound(mortonHost.begin() + start, mortonHost.begin() + end, morton);
                    if (it == mortonHost.begin() + end || *it != morton) continue;
                    size_t g = static_cast<size_t>(it - mortonHost.begin());
                    if (g >= m_totalCells) continue;  // Safety check
                    cellSamples[g].push_back({dx, dy, dz, w});
                    sumX[g] += dx * w;
                    sumY[g] += dy * w;
                    sumZ[g] += dz * w;
                    sumW[g] += w;
                    if (frame > lastFrame[g]) lastFrame[g] = frame;
                    matchedSamples++;
                }

                uint32_t cellsWithData = 0;
                uint32_t cellsWith2Lobes = 0;
                for (size_t g = 0; g < m_totalCells; g++) {
                    size_t base = g * m_entryStride;

                    // Safety check for entry stride
                    if (m_entryStride < 12) {
                        std::cerr << "[PathGuideGrid] ERROR: entry_stride " << m_entryStride << " < 12, skipping stats" << std::endl;
                        break;
                    }
                    if (base + 11 >= dataHost.size()) {
                        std::cerr << "[PathGuideGrid] ERROR: base+11=" << (base+11) << " >= dataHost.size()=" << dataHost.size() << std::endl;
                        break;
                    }

                    uint32_t sampleCount = static_cast<uint32_t>(cellSamples[g].size());

                    if (sampleCount >= 20) {
                        // 2-lobe EM fit from per-sample data
                        std::vector<float> dxArr(sampleCount), dyArr(sampleCount), dzArr(sampleCount), wArr(sampleCount);
                        for (uint32_t s = 0; s < sampleCount; s++) {
                            dxArr[s] = cellSamples[g][s].dx;
                            dyArr[s] = cellSamples[g][s].dy;
                            dzArr[s] = cellSamples[g][s].dz;
                            wArr[s]  = cellSamples[g][s].w;
                        }
                        vmf_fitting::TwoLobeFitResult fit;
                        if (vmf_fitting::fitTwoLobes(dxArr.data(), dyArr.data(), dzArr.data(), wArr.data(), sampleCount, fit)) {
                            dataHost[base + 0] = fit.theta0;
                            dataHost[base + 1] = fit.phi0;
                            dataHost[base + 2] = fit.kappa0;
                            dataHost[base + 3] = fit.theta1;
                            dataHost[base + 4] = fit.phi1;
                            dataHost[base + 5] = fit.kappa1;
                            dataHost[base + PATH_GUIDE_MIX_WEIGHT_OFFSET] = fit.pi0;
                            cellsWithData++;
                            if (fit.kappa1 > 0.5f) cellsWith2Lobes++;
                        }
                    } else if (sumW[g] >= 1e-9f) {
                        // Single-lobe fit from sums
                        float theta, phi, kappa;
                        if (vmf_fitting::fitFromSums(sumX[g], sumY[g], sumZ[g], sumW[g], theta, phi, kappa)) {
                            dataHost[base + 0] = theta;
                            dataHost[base + 1] = phi;
                            dataHost[base + 2] = kappa;
                            dataHost[base + 3] = 0.0f;  // lobe 1 inactive
                            dataHost[base + 4] = 0.0f;
                            dataHost[base + 5] = 0.0f;
                            dataHost[base + PATH_GUIDE_MIX_WEIGHT_OFFSET] = 1.0f;
                            cellsWithData++;
                        }
                    }

                    // Store refinement stats (at offsets 6-9, 11)
                    dataHost[base + 6] += sumX[g];
                    dataHost[base + 7] += sumY[g];
                    dataHost[base + 8] += sumZ[g];
                    dataHost[base + 9] += sumW[g];
                    // offset 10 is pi_0 (already written above), don't accumulate
                    if (lastFrame[g] > 0) {
                        dataHost[base + 11] = static_cast<float>(lastFrame[g]);
                    }
                }
                std::cout << "[PathGuide] Build: " << m_totalCells << " cells, "
                          << cellsWithData << " fitted (" << cellsWith2Lobes << " 2-lobe), "
                          << matchedSamples << "/" << trainCount << " samples\n";
            }
        }
        resetTrainingCount(stream);
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

    // keys are already sorted by (level, morton), so mortonHost is sorted per level

    // Save host-side copy to avoid D2H readback on next build
    m_dataHost = std::move(dataHost);

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
    err = cudaMemcpy(m_data, m_dataHost.data(),
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
    // Reset staging count so next accumulation starts fresh
    resetStagingCount(stream);
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
    return d;
}

PathGuideStagingDescriptor PathGuideGrid::getStagingDescriptor() const {
    PathGuideStagingDescriptor d = {};
    d.buffer = m_stagingBuffer;
    d.count = m_stagingCount;
    d.capacity = m_stagingCapacity;
    return d;
}

PathGuideTrainingStagingDescriptor PathGuideGrid::getTrainingStagingDescriptor() const {
    PathGuideTrainingStagingDescriptor d = {};
    d.buffer = m_trainingBuffer;
    d.count = m_trainingCount;
    d.capacity = m_trainingCapacity;
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
        cudaMemcpy(m_levelOffsetsDevice, m_levelOffsets.data(),
            (m_numLevels + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice);
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

    cudaMemcpy(m_mortonCodes, mortonHost.data(), m_totalCells * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(m_data, m_dataHost.data(),
        static_cast<size_t>(m_totalCells) * m_entryStride * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(m_levelOffsetsDevice, m_levelOffsets.data(),
        (m_numLevels + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice);

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

uint32_t PathGuideGrid::readTrainingCount(cudaStream_t stream) const {
    if (!m_trainingCount) return 0;
    uint32_t count = 0;
    if (stream) cudaStreamSynchronize(stream);
    cudaMemcpy(&count, m_trainingCount, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    return count;
}

} // namespace spectra
