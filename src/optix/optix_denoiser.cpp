#include "optix_denoiser.h"
#include <optix_stubs.h>
#include <iostream>
#include <cstring>

namespace spectra {

static OptixImage2D makeImage2D(CUdeviceptr data, uint32_t w, uint32_t h, OptixPixelFormat fmt) {
    OptixImage2D img = {};
    img.data = data;
    img.width = w;
    img.height = h;
    img.pixelStrideInBytes = 0;  // dense
    img.format = fmt;
    switch (fmt) {
        case OPTIX_PIXEL_FORMAT_FLOAT4: img.rowStrideInBytes = w * 4 * sizeof(float); break;
        case OPTIX_PIXEL_FORMAT_FLOAT3: img.rowStrideInBytes = w * 3 * sizeof(float); break;
        case OPTIX_PIXEL_FORMAT_FLOAT2: img.rowStrideInBytes = w * 2 * sizeof(float); break;
        default: img.rowStrideInBytes = w * 4 * sizeof(float); break;
    }
    return img;
}

OptixDenoiserWrapper::~OptixDenoiserWrapper() {
    shutdown();
}

bool OptixDenoiserWrapper::init(OptixDeviceContext context) {
    if (m_initialized) return true;
    m_context = context;

    OptixDenoiserOptions options = {};
    options.guideAlbedo = 1;
    options.guideNormal = 1;
    options.denoiseAlpha = OPTIX_DENOISER_ALPHA_MODE_COPY;

    OptixResult res = optixDenoiserCreate(
        m_context,
        OPTIX_DENOISER_MODEL_KIND_AOV,
        &options,
        &m_denoiser);
    if (res != OPTIX_SUCCESS) {
        std::cerr << "[Denoiser] Failed to create: " << optixGetErrorName(res)
                  << " (" << optixGetErrorString(res) << ")\n";
        return false;
    }

    // HDR average color buffer (3 floats for AOV model)
    cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&m_intensityBuffer), 3 * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "[Denoiser] Failed to allocate intensity buffer\n";
        return false;
    }

    m_initialized = true;
    std::cout << "[Denoiser] Initialized (AOV model, albedo+normal guides)\n";
    return true;
}

bool OptixDenoiserWrapper::resize(uint32_t width, uint32_t height) {
    if (!m_initialized || width == 0 || height == 0) return false;
    if (width == m_width && height == m_height) return true;

    OptixDenoiserSizes sizes = {};
    OptixResult res = optixDenoiserComputeMemoryResources(m_denoiser, width, height, &sizes);
    if (res != OPTIX_SUCCESS) {
        std::cerr << "[Denoiser] Failed to compute memory: " << optixGetErrorName(res) << "\n";
        return false;
    }

    // Free old buffers
    if (m_stateBuffer) { cudaFree(reinterpret_cast<void*>(m_stateBuffer)); m_stateBuffer = 0; }
    if (m_scratchBuffer) { cudaFree(reinterpret_cast<void*>(m_scratchBuffer)); m_scratchBuffer = 0; }
    if (m_outputBuffer) { cudaFree(reinterpret_cast<void*>(m_outputBuffer)); m_outputBuffer = 0; }

    // Allocate state
    m_stateSize = sizes.stateSizeInBytes;
    cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&m_stateBuffer), m_stateSize);
    if (err != cudaSuccess) {
        std::cerr << "[Denoiser] Failed to allocate state buffer\n";
        return false;
    }

    // Scratch: use the larger of withoutOverlap and computeAverageColor
    m_scratchSize = sizes.withoutOverlapScratchSizeInBytes;
    if (sizes.computeAverageColorSizeInBytes > m_scratchSize)
        m_scratchSize = sizes.computeAverageColorSizeInBytes;
    if (sizes.computeIntensitySizeInBytes > m_scratchSize)
        m_scratchSize = sizes.computeIntensitySizeInBytes;

    err = cudaMalloc(reinterpret_cast<void**>(&m_scratchBuffer), m_scratchSize);
    if (err != cudaSuccess) {
        std::cerr << "[Denoiser] Failed to allocate scratch buffer\n";
        return false;
    }

    // Output buffer
    m_outputBufferSize = (size_t)width * height * sizeof(float4);
    err = cudaMalloc(reinterpret_cast<void**>(&m_outputBuffer), m_outputBufferSize);
    if (err != cudaSuccess) {
        std::cerr << "[Denoiser] Failed to allocate output buffer\n";
        return false;
    }

    // Setup denoiser
    res = optixDenoiserSetup(
        m_denoiser, 0,
        width, height,
        m_stateBuffer, m_stateSize,
        m_scratchBuffer, m_scratchSize);
    if (res != OPTIX_SUCCESS) {
        std::cerr << "[Denoiser] Setup failed: " << optixGetErrorName(res) << "\n";
        return false;
    }

    m_width = width;
    m_height = height;
    std::cout << "[Denoiser] Resized to " << width << "x" << height
              << " (state=" << (m_stateSize/1024) << "KB, scratch=" << (m_scratchSize/1024) << "KB)\n";
    return true;
}

void OptixDenoiserWrapper::denoise(float4* input, float4* output,
                                    float4* albedo, float4* normal,
                                    uint32_t width, uint32_t height,
                                    float blendFactor,
                                    cudaStream_t stream) {
    if (!m_initialized || !m_denoiser) return;
    if (blendFactor >= 1.0f) return;  // passthrough
    if (width != m_width || height != m_height) {
        if (!resize(width, height)) return;
    }

    CUdeviceptr inputPtr = reinterpret_cast<CUdeviceptr>(input);
    CUdeviceptr albedoPtr = reinterpret_cast<CUdeviceptr>(albedo);
    CUdeviceptr normalPtr = reinterpret_cast<CUdeviceptr>(normal);

    OptixImage2D inputImage = makeImage2D(inputPtr, width, height, OPTIX_PIXEL_FORMAT_FLOAT4);

    // Compute average color for HDR (AOV model)
    optixDenoiserComputeAverageColor(
        m_denoiser, stream,
        &inputImage,
        m_intensityBuffer,
        m_scratchBuffer, m_scratchSize);

    // Guide layers
    OptixDenoiserGuideLayer guideLayer = {};
    guideLayer.albedo = makeImage2D(albedoPtr, width, height, OPTIX_PIXEL_FORMAT_FLOAT4);
    guideLayer.normal = makeImage2D(normalPtr, width, height, OPTIX_PIXEL_FORMAT_FLOAT4);

    // Denoiser layer
    OptixDenoiserLayer layer = {};
    layer.input = inputImage;
    layer.output = makeImage2D(m_outputBuffer, width, height, OPTIX_PIXEL_FORMAT_FLOAT4);
    layer.type = OPTIX_DENOISER_AOV_TYPE_BEAUTY;

    // Params
    OptixDenoiserParams params = {};
    params.hdrAverageColor = m_intensityBuffer;
    params.blendFactor = blendFactor;

    OptixResult res = optixDenoiserInvoke(
        m_denoiser, stream,
        &params,
        m_stateBuffer, m_stateSize,
        &guideLayer,
        &layer, 1,
        0, 0,
        m_scratchBuffer, m_scratchSize);

    if (res != OPTIX_SUCCESS) {
        std::cerr << "[Denoiser] Invoke failed: " << optixGetErrorName(res) << "\n";
        return;
    }

    // Copy denoised result to output
    cudaMemcpyAsync(output, reinterpret_cast<void*>(m_outputBuffer),
        (size_t)width * height * sizeof(float4),
        cudaMemcpyDeviceToDevice, stream);
}

void OptixDenoiserWrapper::shutdown() {
    if (m_denoiser) { optixDenoiserDestroy(m_denoiser); m_denoiser = nullptr; }
    if (m_stateBuffer) { cudaFree(reinterpret_cast<void*>(m_stateBuffer)); m_stateBuffer = 0; }
    if (m_scratchBuffer) { cudaFree(reinterpret_cast<void*>(m_scratchBuffer)); m_scratchBuffer = 0; }
    if (m_intensityBuffer) { cudaFree(reinterpret_cast<void*>(m_intensityBuffer)); m_intensityBuffer = 0; }
    if (m_outputBuffer) { cudaFree(reinterpret_cast<void*>(m_outputBuffer)); m_outputBuffer = 0; }
    m_width = 0;
    m_height = 0;
    m_initialized = false;
}

} // namespace spectra
