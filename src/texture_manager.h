#pragma once

#include <cuda_runtime.h>
#include <string>
#include <unordered_map>
#include <cstdint>

namespace spectra {

// Handle to a GPU texture
using TextureHandle = uint32_t;
constexpr TextureHandle INVALID_TEXTURE_HANDLE = UINT32_MAX;

// GPU texture info
struct GpuTexture {
    cudaTextureObject_t texObj;     // Texture object for sampling
    cudaArray_t array;               // CUDA array holding pixel data
    uint32_t width;
    uint32_t height;
    uint32_t channels;
    uint32_t refCount;
    bool isSRGB;                     // Whether texture is in sRGB color space
};

class TextureManager {
public:
    TextureManager() = default;
    ~TextureManager();

    // Non-copyable
    TextureManager(const TextureManager&) = delete;
    TextureManager& operator=(const TextureManager&) = delete;

    // Load texture from file (supports PNG, JPG, etc. via stb_image)
    // Returns handle on success, INVALID_TEXTURE_HANDLE on failure
    // isSRGB: set to true for color textures (albedo), false for linear (normal, roughness)
    TextureHandle loadFromFile(const std::string& path, bool isSRGB = true);

    // Create texture from raw data (RGBA8)
    TextureHandle createFromData(const uint8_t* data, uint32_t width, uint32_t height,
                                 uint32_t channels, bool isSRGB = true);

    // Increment reference count
    void addRef(TextureHandle handle);

    // Decrement reference count, free if zero
    void release(TextureHandle handle);

    // Get texture info
    const GpuTexture* get(TextureHandle handle) const;

    // Get CUDA texture object for shader use
    cudaTextureObject_t getTextureObject(TextureHandle handle) const;

    // Free all GPU resources
    void clear();

    // Get total GPU memory used
    size_t getGpuMemoryUsage() const;

    // Get number of active textures
    size_t getTextureCount() const { return m_textures.size(); }

    // Check if texture is already loaded (by path)
    TextureHandle findByPath(const std::string& path) const;

private:
    TextureHandle createTextureInternal(const uint8_t* data, uint32_t width, uint32_t height,
                                        uint32_t channels, bool isSRGB, const std::string& debugName);

    std::unordered_map<TextureHandle, GpuTexture> m_textures;
    std::unordered_map<std::string, TextureHandle> m_pathToHandle;  // Cache by path
    TextureHandle m_nextHandle = 0;
    size_t m_totalMemory = 0;
};

} // namespace spectra
