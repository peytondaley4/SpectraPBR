#include "texture_manager.h"
#include <stb_image.h>
#include <iostream>
#include <cstring>
#include <algorithm>

namespace spectra {

TextureManager::~TextureManager() {
    clear();
}

TextureHandle TextureManager::loadFromFile(const std::string& path, bool isSRGB) {
    // Check cache first
    auto it = m_pathToHandle.find(path);
    if (it != m_pathToHandle.end()) {
        addRef(it->second);
        return it->second;
    }

    // Load image with stb_image
    int width, height, channels;
    uint8_t* data = stbi_load(path.c_str(), &width, &height, &channels, 4);  // Force RGBA

    if (!data) {
        std::cerr << "[TextureManager] Failed to load: " << path
                  << " - " << stbi_failure_reason() << "\n";
        return INVALID_TEXTURE_HANDLE;
    }

    TextureHandle handle = createTextureInternal(data, width, height, 4, isSRGB, path);
    stbi_image_free(data);

    if (handle != INVALID_TEXTURE_HANDLE) {
        m_pathToHandle[path] = handle;
    }

    return handle;
}

TextureHandle TextureManager::createFromData(const uint8_t* data, uint32_t width, uint32_t height,
                                             uint32_t channels, bool isSRGB) {
    return createTextureInternal(data, width, height, channels, isSRGB, "procedural");
}

// Helper: calculate number of mip levels
static uint32_t calculateMipLevels(uint32_t width, uint32_t height) {
    uint32_t levels = 1;
    uint32_t size = std::max(width, height);
    while (size > 1) {
        size >>= 1;
        levels++;
    }
    return levels;
}

// Helper: generate next mip level using box filter
static void generateMipLevel(const uint8_t* src, uint8_t* dst,
                             uint32_t srcWidth, uint32_t srcHeight,
                             uint32_t dstWidth, uint32_t dstHeight) {
    for (uint32_t y = 0; y < dstHeight; y++) {
        for (uint32_t x = 0; x < dstWidth; x++) {
            // Sample 2x2 block from source
            uint32_t sx = x * 2;
            uint32_t sy = y * 2;
            
            uint32_t r = 0, g = 0, b = 0, a = 0;
            int count = 0;
            
            for (int dy = 0; dy < 2; dy++) {
                for (int dx = 0; dx < 2; dx++) {
                    uint32_t px = std::min(sx + dx, srcWidth - 1);
                    uint32_t py = std::min(sy + dy, srcHeight - 1);
                    uint32_t idx = (py * srcWidth + px) * 4;
                    r += src[idx + 0];
                    g += src[idx + 1];
                    b += src[idx + 2];
                    a += src[idx + 3];
                    count++;
                }
            }
            
            uint32_t dstIdx = (y * dstWidth + x) * 4;
            dst[dstIdx + 0] = static_cast<uint8_t>(r / count);
            dst[dstIdx + 1] = static_cast<uint8_t>(g / count);
            dst[dstIdx + 2] = static_cast<uint8_t>(b / count);
            dst[dstIdx + 3] = static_cast<uint8_t>(a / count);
        }
    }
}

TextureHandle TextureManager::createTextureInternal(const uint8_t* data, uint32_t width, uint32_t height,
                                                     uint32_t channels, bool isSRGB, const std::string& debugName) {
    if (!data || width == 0 || height == 0) {
        std::cerr << "[TextureManager] Invalid texture data\n";
        return INVALID_TEXTURE_HANDLE;
    }

    GpuTexture tex = {};
    tex.width = width;
    tex.height = height;
    tex.channels = channels;
    tex.refCount = 1;
    tex.isSRGB = isSRGB;
    tex.mipLevels = calculateMipLevels(width, height);

    // Convert to RGBA if needed
    std::vector<uint8_t> rgbaData;
    const uint8_t* srcData = data;

    if (channels != 4) {
        rgbaData.resize(width * height * 4);
        for (uint32_t i = 0; i < width * height; ++i) {
            if (channels == 1) {
                rgbaData[i * 4 + 0] = data[i];
                rgbaData[i * 4 + 1] = data[i];
                rgbaData[i * 4 + 2] = data[i];
                rgbaData[i * 4 + 3] = 255;
            } else if (channels == 2) {
                rgbaData[i * 4 + 0] = data[i * 2 + 0];
                rgbaData[i * 4 + 1] = data[i * 2 + 0];
                rgbaData[i * 4 + 2] = data[i * 2 + 0];
                rgbaData[i * 4 + 3] = data[i * 2 + 1];
            } else if (channels == 3) {
                rgbaData[i * 4 + 0] = data[i * 3 + 0];
                rgbaData[i * 4 + 1] = data[i * 3 + 1];
                rgbaData[i * 4 + 2] = data[i * 3 + 2];
                rgbaData[i * 4 + 3] = 255;
            }
        }
        srcData = rgbaData.data();
    }

    // Generate all mipmap levels on CPU
    std::vector<std::vector<uint8_t>> mipData(tex.mipLevels);
    mipData[0].assign(srcData, srcData + width * height * 4);
    
    uint32_t mipWidth = width;
    uint32_t mipHeight = height;
    size_t totalMemory = mipWidth * mipHeight * 4;
    
    for (uint32_t level = 1; level < tex.mipLevels; level++) {
        uint32_t prevWidth = mipWidth;
        uint32_t prevHeight = mipHeight;
        mipWidth = std::max(1u, mipWidth >> 1);
        mipHeight = std::max(1u, mipHeight >> 1);
        
        mipData[level].resize(mipWidth * mipHeight * 4);
        generateMipLevel(mipData[level - 1].data(), mipData[level].data(),
                        prevWidth, prevHeight, mipWidth, mipHeight);
        totalMemory += mipWidth * mipHeight * 4;
    }

    // Create mipmapped CUDA array
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();
    cudaExtent extent = make_cudaExtent(width, height, 0);

    cudaError_t err = cudaMallocMipmappedArray(&tex.mipmappedArray, &channelDesc, extent, tex.mipLevels);
    if (err != cudaSuccess) {
        std::cerr << "[TextureManager] Failed to allocate mipmapped array: "
                  << cudaGetErrorString(err) << "\n";
        return INVALID_TEXTURE_HANDLE;
    }

    // Copy each mip level to the array
    mipWidth = width;
    mipHeight = height;
    for (uint32_t level = 0; level < tex.mipLevels; level++) {
        cudaArray_t levelArray;
        err = cudaGetMipmappedArrayLevel(&levelArray, tex.mipmappedArray, level);
        if (err != cudaSuccess) {
            std::cerr << "[TextureManager] Failed to get mip level " << level << ": "
                      << cudaGetErrorString(err) << "\n";
            cudaFreeMipmappedArray(tex.mipmappedArray);
            return INVALID_TEXTURE_HANDLE;
        }

        err = cudaMemcpy2DToArray(levelArray, 0, 0, mipData[level].data(),
                                  mipWidth * 4, mipWidth * 4, mipHeight,
                                  cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            std::cerr << "[TextureManager] Failed to copy mip level " << level << ": "
                      << cudaGetErrorString(err) << "\n";
            cudaFreeMipmappedArray(tex.mipmappedArray);
            return INVALID_TEXTURE_HANDLE;
        }

        mipWidth = std::max(1u, mipWidth >> 1);
        mipHeight = std::max(1u, mipHeight >> 1);
    }

    // Create texture object with mipmapping
    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeMipmappedArray;
    resDesc.res.mipmap.mipmap = tex.mipmappedArray;

    cudaTextureDesc texDesc = {};
    texDesc.addressMode[0] = cudaAddressModeWrap;
    texDesc.addressMode[1] = cudaAddressModeWrap;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeNormalizedFloat;
    texDesc.sRGB = isSRGB ? 1 : 0;
    texDesc.normalizedCoords = 1;
    texDesc.maxAnisotropy = 16;
    texDesc.mipmapFilterMode = cudaFilterModeLinear;
    texDesc.minMipmapLevelClamp = 0.0f;
    texDesc.maxMipmapLevelClamp = static_cast<float>(tex.mipLevels - 1);

    err = cudaCreateTextureObject(&tex.texObj, &resDesc, &texDesc, nullptr);
    if (err != cudaSuccess) {
        std::cerr << "[TextureManager] Failed to create texture object: "
                  << cudaGetErrorString(err) << "\n";
        cudaFreeMipmappedArray(tex.mipmappedArray);
        return INVALID_TEXTURE_HANDLE;
    }

    TextureHandle handle = m_nextHandle++;
    m_textures[handle] = tex;
    m_totalMemory += totalMemory;

    std::cout << "[TextureManager] Loaded: " << debugName
              << " (" << width << "x" << height << ", "
              << tex.mipLevels << " mips, "
              << (isSRGB ? "sRGB" : "linear") << ")\n";

    return handle;
}

void TextureManager::addRef(TextureHandle handle) {
    auto it = m_textures.find(handle);
    if (it != m_textures.end()) {
        it->second.refCount++;
    }
}

void TextureManager::release(TextureHandle handle) {
    auto it = m_textures.find(handle);
    if (it == m_textures.end()) {
        return;
    }

    it->second.refCount--;
    if (it->second.refCount == 0) {
        GpuTexture& tex = it->second;

        // Calculate total memory including mipmaps (factor of ~1.33x base size)
        size_t freedMemory = 0;
        uint32_t w = tex.width, h = tex.height;
        for (uint32_t i = 0; i < tex.mipLevels; i++) {
            freedMemory += w * h * 4;
            w = std::max(1u, w >> 1);
            h = std::max(1u, h >> 1);
        }

        if (tex.texObj) {
            cudaDestroyTextureObject(tex.texObj);
        }
        if (tex.mipmappedArray) {
            cudaFreeMipmappedArray(tex.mipmappedArray);
        }

        m_totalMemory -= freedMemory;

        // Remove from path cache
        for (auto pathIt = m_pathToHandle.begin(); pathIt != m_pathToHandle.end(); ++pathIt) {
            if (pathIt->second == handle) {
                m_pathToHandle.erase(pathIt);
                break;
            }
        }

        m_textures.erase(it);

        std::cout << "[TextureManager] Released texture " << handle
                  << " (freed " << freedMemory / 1024 << " KB)\n";
    }
}

const GpuTexture* TextureManager::get(TextureHandle handle) const {
    auto it = m_textures.find(handle);
    if (it == m_textures.end()) {
        return nullptr;
    }
    return &it->second;
}

cudaTextureObject_t TextureManager::getTextureObject(TextureHandle handle) const {
    const GpuTexture* tex = get(handle);
    return tex ? tex->texObj : 0;
}

void TextureManager::clear() {
    for (auto& [handle, tex] : m_textures) {
        if (tex.texObj) {
            cudaDestroyTextureObject(tex.texObj);
        }
        if (tex.mipmappedArray) {
            cudaFreeMipmappedArray(tex.mipmappedArray);
        }
    }
    m_textures.clear();
    m_pathToHandle.clear();
    m_totalMemory = 0;
    std::cout << "[TextureManager] Cleared all textures\n";
}

size_t TextureManager::getGpuMemoryUsage() const {
    return m_totalMemory;
}

TextureHandle TextureManager::findByPath(const std::string& path) const {
    auto it = m_pathToHandle.find(path);
    if (it != m_pathToHandle.end()) {
        return it->second;
    }
    return INVALID_TEXTURE_HANDLE;
}

} // namespace spectra
