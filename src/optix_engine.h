#pragma once

#include <optix.h>
#include <cuda_runtime.h>
#include <filesystem>
#include <cstdint>
#include <vector>

namespace spectra {

// Launch parameters passed to OptiX programs
struct LaunchParams {
    float4* output_buffer;
    uint32_t width;
    uint32_t height;
    uint32_t frame_index;
};

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
    // ptxDir: directory containing raygen.ptx and miss.ptx
    bool createPipeline(const std::filesystem::path& ptxDir);

    // Shutdown and cleanup
    void shutdown();

    // Set render dimensions (call before render if changed)
    void setDimensions(uint32_t width, uint32_t height);

    // Render to output buffer
    // outputBuffer: CUDA device pointer to float4 array
    // stream: CUDA stream for async execution
    void render(float4* outputBuffer, cudaStream_t stream);

    // Get current frame index
    uint32_t getFrameIndex() const { return m_frameIndex; }

private:
    bool createModule(const std::filesystem::path& ptxPath, OptixModule* module);
    bool createProgramGroups();
    bool createSBT();

    // OptiX context
    OptixDeviceContext m_context = nullptr;

    // Pipeline
    OptixModule m_raygenModule = nullptr;
    OptixModule m_missModule = nullptr;
    OptixPipeline m_pipeline = nullptr;
    OptixPipelineCompileOptions m_pipelineCompileOptions = {};

    // Program groups
    OptixProgramGroup m_raygenPG = nullptr;
    OptixProgramGroup m_missPG = nullptr;

    // Shader Binding Table
    OptixShaderBindingTable m_sbt = {};
    CUdeviceptr m_raygenRecord = 0;
    CUdeviceptr m_missRecord = 0;

    // Launch parameters
    LaunchParams m_launchParams = {};
    CUdeviceptr m_launchParamsBuffer = 0;

    // State
    uint32_t m_width = 0;
    uint32_t m_height = 0;
    uint32_t m_frameIndex = 0;
};

} // namespace spectra
