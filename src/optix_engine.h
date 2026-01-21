#pragma once

#include "shared_types.h"
#include <optix.h>
#include <cuda_runtime.h>
#include <filesystem>
#include <cstdint>
#include <vector>

namespace spectra {

// OptiX error checking macro
#define OPTIX_CHECK(call)                                                        \
    do {                                                                         \
        OptixResult res = call;                                                  \
        if (res != OPTIX_SUCCESS) {                                              \
            std::cerr << "[OptiX] Error: " << optixGetErrorName(res)             \
                      << " (" << optixGetErrorString(res) << ")"                 \
                      << " at " << __FILE__ << ":" << __LINE__ << "\n";          \
            return false;                                                        \
        }                                                                        \
    } while (0)

// Variant for log output
#define OPTIX_CHECK_LOG(call)                                                    \
    do {                                                                         \
        OptixResult res = call;                                                  \
        if (res != OPTIX_SUCCESS) {                                              \
            std::cerr << "[OptiX] Error: " << optixGetErrorName(res)             \
                      << " (" << optixGetErrorString(res) << ")"                 \
                      << " at " << __FILE__ << ":" << __LINE__ << "\n";          \
            if (logSize > 1) {                                                   \
                std::cerr << "[OptiX] Log: " << log << "\n";                     \
            }                                                                    \
            return false;                                                        \
        } else if (logSize > 1) {                                                \
            std::cout << "[OptiX] Log: " << log << "\n";                         \
        }                                                                        \
    } while (0)

class OptixEngine {
public:
    OptixEngine() = default;
    ~OptixEngine();

    // Non-copyable
    OptixEngine(const OptixEngine&) = delete;
    OptixEngine& operator=(const OptixEngine&) = delete;

    // Initialize OptiX context on existing CUDA context
    bool init(CUcontext cudaContext);

    // Load PTX modules and create pipeline
    // ptxDir: directory containing raygen.ptx, miss.ptx, closesthit.ptx
    bool createPipeline(const std::filesystem::path& ptxDir);

    // Shutdown and cleanup
    void shutdown();

    // Set render dimensions (call before render if changed)
    void setDimensions(uint32_t width, uint32_t height);

    // Set camera parameters
    void setCamera(const CameraParams& camera);

    // Set scene traversable handle
    void setSceneHandle(OptixTraversableHandle handle);

    // Set geometry buffers for shader access
    void setGeometryBuffers(CUdeviceptr* vertexBuffers, CUdeviceptr* indexBuffers);

    // Build/update SBT with materials
    // materials: array of GpuMaterial for each hit group
    // geometryIndices: geometry index per material for buffer lookup
    bool buildSBT(const std::vector<GpuMaterial>& materials,
                  const std::vector<uint32_t>& geometryIndices);

    // Render to output buffer
    // outputBuffer: CUDA device pointer to float4 array
    // stream: CUDA stream for async execution
    void render(float4* outputBuffer, cudaStream_t stream);

    // Get current frame index
    uint32_t getFrameIndex() const { return m_frameIndex; }

    // Get OptiX context (for BVH building)
    OptixDeviceContext getContext() const { return m_context; }

    // Get pipeline compile options (for BVH building)
    const OptixPipelineCompileOptions& getPipelineCompileOptions() const {
        return m_pipelineCompileOptions;
    }

private:
    bool createModule(const std::filesystem::path& ptxPath, OptixModule* module);
    bool createProgramGroups();
    bool createDefaultSBT();

    // OptiX context
    OptixDeviceContext m_context = nullptr;

    // Pipeline
    OptixModule m_raygenModule = nullptr;
    OptixModule m_missModule = nullptr;
    OptixModule m_closesthitModule = nullptr;
    OptixPipeline m_pipeline = nullptr;
    OptixPipelineCompileOptions m_pipelineCompileOptions = {};

    // Program groups
    OptixProgramGroup m_raygenPG = nullptr;
    OptixProgramGroup m_missPG = nullptr;
    OptixProgramGroup m_hitgroupPG = nullptr;

    // Shader Binding Table
    OptixShaderBindingTable m_sbt = {};
    CUdeviceptr m_raygenRecord = 0;
    CUdeviceptr m_missRecord = 0;
    CUdeviceptr m_hitgroupRecords = 0;
    size_t m_hitgroupRecordCount = 0;

    // Launch parameters
    LaunchParams m_launchParams = {};
    CUdeviceptr m_launchParamsBuffer = 0;

    // State
    uint32_t m_width = 0;
    uint32_t m_height = 0;
    uint32_t m_frameIndex = 0;
};

} // namespace spectra
