#include "texture_manager.h"
#include <stb_image.h>
#include <iostream>
#include <cstring>

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

    // Create CUDA array
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();

    cudaError_t err = cudaMallocArray(&tex.array, &channelDesc, width, height);
    if (err != cudaSuccess) {
        std::cerr << "[TextureManager] Failed to allocate CUDA array: "
                  << cudaGetErrorString(err) << "\n";
        return INVALID_TEXTURE_HANDLE;
    }

    // Copy data to array
    // Ensure data is RGBA (4 channels)
    std::vector<uint8_t> rgbaData;
    const uint8_t* srcData = data;

    if (channels != 4) {
        rgbaData.resize(width * height * 4);
        for (uint32_t i = 0; i < width * height; ++i) {
            if (channels == 1) {
                // Grayscale
                rgbaData[i * 4 + 0] = data[i];
                rgbaData[i * 4 + 1] = data[i];
                rgbaData[i * 4 + 2] = data[i];
                rgbaData[i * 4 + 3] = 255;
            } else if (channels == 2) {
                // Grayscale + Alpha
                rgbaData[i * 4 + 0] = data[i * 2 + 0];
                rgbaData[i * 4 + 1] = data[i * 2 + 0];
                rgbaData[i * 4 + 2] = data[i * 2 + 0];
                rgbaData[i * 4 + 3] = data[i * 2 + 1];
            } else if (channels == 3) {
                // RGB
                rgbaData[i * 4 + 0] = data[i * 3 + 0];
                rgbaData[i * 4 + 1] = data[i * 3 + 1];
                rgbaData[i * 4 + 2] = data[i * 3 + 2];
                rgbaData[i * 4 + 3] = 255;
            }
        }
        srcData = rgbaData.data();
    }

    err = cudaMemcpy2DToArray(tex.array, 0, 0, srcData,
                              width * 4, width * 4, height,
                              cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "[TextureManager] Failed to copy texture data: "
                  << cudaGetErrorString(err) << "\n";
        cudaFreeArray(tex.array);
        return INVALID_TEXTURE_HANDLE;
    }

    // Create texture object
    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = tex.array;

    cudaTextureDesc texDesc = {};
    texDesc.addressMode[0] = cudaAddressModeWrap;
    texDesc.addressMode[1] = cudaAddressModeWrap;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeNormalizedFloat;
    texDesc.sRGB = isSRGB ? 1 : 0;
    texDesc.normalizedCoords = 1;
    texDesc.maxAnisotropy = 16;
    texDesc.mipmapFilterMode = cudaFilterModeLinear;

    err = cudaCreateTextureObject(&tex.texObj, &resDesc, &texDesc, nullptr);
    if (err != cudaSuccess) {
        std::cerr << "[TextureManager] Failed to create texture object: "
                  << cudaGetErrorString(err) << "\n";
        cudaFreeArray(tex.array);
        return INVALID_TEXTURE_HANDLE;
    }

    TextureHandle handle = m_nextHandle++;
    m_textures[handle] = tex;
    m_totalMemory += width * height * 4;  // RGBA

    std::cout << "[TextureManager] Loaded: " << debugName
              << " (" << width << "x" << height << ", "
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

        size_t freedMemory = tex.width * tex.height * 4;

        if (tex.texObj) {
            cudaDestroyTextureObject(tex.texObj);
        }
        if (tex.array) {
            cudaFreeArray(tex.array);
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
        if (tex.array) {
            cudaFreeArray(tex.array);
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
