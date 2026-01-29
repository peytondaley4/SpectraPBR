#include "texture_preview_cache.h"
#include <iostream>

namespace spectra {
namespace ui {

//------------------------------------------------------------------------------
// CUDA Kernel - Downsample texture to preview buffer
//------------------------------------------------------------------------------
__global__ void generatePreviewKernel(
    cudaTextureObject_t srcTexture,
    float4* output,
    uint32_t size
) {
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= size || y >= size) return;
    
    // Sample source texture with normalized coordinates
    float u = (static_cast<float>(x) + 0.5f) / static_cast<float>(size);
    float v = (static_cast<float>(y) + 0.5f) / static_cast<float>(size);
    
    float4 color = tex2D<float4>(srcTexture, u, v);
    output[y * size + x] = color;
}

//------------------------------------------------------------------------------
// TexturePreviewCache Implementation
//------------------------------------------------------------------------------

TexturePreviewCache::~TexturePreviewCache() {
    shutdown();
}

bool TexturePreviewCache::init() {
    if (m_initialized) return true;
    
    // Allocate device buffer for preview rendering
    size_t bufferSize = PREVIEW_SIZE * PREVIEW_SIZE * sizeof(float4);
    cudaError_t err = cudaMalloc(&m_deviceBuffer, bufferSize);
    if (err != cudaSuccess) {
        std::cerr << "[TexturePreviewCache] Failed to allocate device buffer: "
                  << cudaGetErrorString(err) << "\n";
        return false;
    }
    
    // Pre-allocate arrays and textures for caching
    m_cachedArrays.resize(MAX_CACHED_TEXTURES, nullptr);
    m_cachedTextures.resize(MAX_CACHED_TEXTURES, 0);
    
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
    
    for (uint32_t i = 0; i < MAX_CACHED_TEXTURES; i++) {
        err = cudaMallocArray(&m_cachedArrays[i], &channelDesc, PREVIEW_SIZE, PREVIEW_SIZE);
        if (err != cudaSuccess) {
            std::cerr << "[TexturePreviewCache] Failed to allocate cache array " << i << ": "
                      << cudaGetErrorString(err) << "\n";
            shutdown();
            return false;
        }
        
        // Create texture object for the cached array
        cudaResourceDesc resDesc = {};
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = m_cachedArrays[i];
        
        cudaTextureDesc texDesc = {};
        texDesc.addressMode[0] = cudaAddressModeClamp;
        texDesc.addressMode[1] = cudaAddressModeClamp;
        texDesc.filterMode = cudaFilterModeLinear;
        texDesc.readMode = cudaReadModeElementType;  // float4 stays as float4
        texDesc.normalizedCoords = 1;
        
        err = cudaCreateTextureObject(&m_cachedTextures[i], &resDesc, &texDesc, nullptr);
        if (err != cudaSuccess) {
            std::cerr << "[TexturePreviewCache] Failed to create cache texture " << i << ": "
                      << cudaGetErrorString(err) << "\n";
            shutdown();
            return false;
        }
    }
    
    m_initialized = true;
    std::cout << "[TexturePreviewCache] Initialized with " << MAX_CACHED_TEXTURES 
              << " cache slots at " << PREVIEW_SIZE << "x" << PREVIEW_SIZE << "\n";
    return true;
}

void TexturePreviewCache::shutdown() {
    for (uint32_t i = 0; i < m_cachedTextures.size(); i++) {
        if (m_cachedTextures[i] != 0) {
            cudaDestroyTextureObject(m_cachedTextures[i]);
            m_cachedTextures[i] = 0;
        }
    }
    
    for (uint32_t i = 0; i < m_cachedArrays.size(); i++) {
        if (m_cachedArrays[i] != nullptr) {
            cudaFreeArray(m_cachedArrays[i]);
            m_cachedArrays[i] = nullptr;
        }
    }
    
    if (m_deviceBuffer) {
        cudaFree(m_deviceBuffer);
        m_deviceBuffer = nullptr;
    }
    
    m_cachedArrays.clear();
    m_cachedTextures.clear();
    m_cachedCount = 0;
    m_initialized = false;
}

void TexturePreviewCache::generatePreviews(const cudaTextureObject_t* sourceTextures,
                                            uint32_t count,
                                            cudaStream_t stream) {
    if (!m_initialized || !sourceTextures) {
        m_cachedCount = 0;
        return;
    }
    
    m_cachedCount = std::min(count, MAX_CACHED_TEXTURES);
    
    dim3 blockSize(16, 16);
    dim3 gridSize((PREVIEW_SIZE + blockSize.x - 1) / blockSize.x,
                  (PREVIEW_SIZE + blockSize.y - 1) / blockSize.y);
    
    for (uint32_t i = 0; i < m_cachedCount; i++) {
        if (sourceTextures[i] == 0) continue;
        
        // Render source texture to preview buffer
        generatePreviewKernel<<<gridSize, blockSize, 0, stream>>>(
            sourceTextures[i], m_deviceBuffer, PREVIEW_SIZE);
        
        // Copy preview buffer to cached array
        cudaMemcpy2DToArrayAsync(
            m_cachedArrays[i], 0, 0,
            m_deviceBuffer,
            PREVIEW_SIZE * sizeof(float4),
            PREVIEW_SIZE * sizeof(float4),
            PREVIEW_SIZE,
            cudaMemcpyDeviceToDevice,
            stream
        );
    }
}

} // namespace ui
} // namespace spectra
