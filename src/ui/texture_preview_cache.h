#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <vector>

namespace spectra {
namespace ui {

// Caches texture previews at a fixed small resolution for fast UI display
// Instead of sampling full-res textures every frame, we render once to a cache
class TexturePreviewCache {
public:
    static constexpr uint32_t PREVIEW_SIZE = 160;  // Must match PropertyPanel::TEXTURE_PREVIEW_SIZE
    static constexpr uint32_t MAX_CACHED_TEXTURES = 8;

    TexturePreviewCache() = default;
    ~TexturePreviewCache();

    // Non-copyable
    TexturePreviewCache(const TexturePreviewCache&) = delete;
    TexturePreviewCache& operator=(const TexturePreviewCache&) = delete;

    bool init();
    void shutdown();

    // Generate cached previews from source textures
    // Call when selection changes, not every frame
    void generatePreviews(const cudaTextureObject_t* sourceTextures, 
                          uint32_t count, 
                          cudaStream_t stream = nullptr);

    // Get cached preview textures for rendering (fast path)
    const cudaTextureObject_t* getCachedTextures() const { return m_cachedTextures.data(); }
    uint32_t getCachedTextureCount() const { return m_cachedCount; }

    // Check if cache is valid
    bool isValid() const { return m_initialized && m_cachedCount > 0; }

private:
    bool m_initialized = false;
    
    // Device memory for preview rendering
    float4* m_deviceBuffer = nullptr;
    
    // Cached preview textures (small, fast to sample)
    std::vector<cudaArray_t> m_cachedArrays;
    std::vector<cudaTextureObject_t> m_cachedTextures;
    uint32_t m_cachedCount = 0;
};

} // namespace ui
} // namespace spectra
