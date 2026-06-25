#pragma once

#include <optix.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace spectra {

class OptixDenoiserWrapper {
public:
    OptixDenoiserWrapper() = default;
    ~OptixDenoiserWrapper();

    OptixDenoiserWrapper(const OptixDenoiserWrapper&) = delete;
    OptixDenoiserWrapper& operator=(const OptixDenoiserWrapper&) = delete;

    bool init(OptixDeviceContext context);
    bool resize(uint32_t width, uint32_t height);

    // Run the denoiser. input/output may be the same pointer.
    // albedo: first-hit baseColor guide (float4)
    // normal: first-hit camera-space normal guide (float4)
    // blendFactor: 0 = full denoise, 1 = passthrough
    void denoise(float4* input, float4* output,
                 float4* albedo, float4* normal,
                 uint32_t width, uint32_t height,
                 float blendFactor,
                 cudaStream_t stream);

    void shutdown();
    bool isInitialized() const { return m_initialized; }

private:
    OptixDeviceContext m_context = nullptr;
    OptixDenoiser m_denoiser = nullptr;

    CUdeviceptr m_stateBuffer = 0;
    size_t m_stateSize = 0;
    CUdeviceptr m_scratchBuffer = 0;
    size_t m_scratchSize = 0;
    CUdeviceptr m_intensityBuffer = 0;  // single float for HDR intensity
    CUdeviceptr m_outputBuffer = 0;     // intermediate denoised output
    size_t m_outputBufferSize = 0;

    uint32_t m_width = 0;
    uint32_t m_height = 0;
    bool m_initialized = false;
};

} // namespace spectra
