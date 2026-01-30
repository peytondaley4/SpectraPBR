#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <vector>
#include "ui_types.h"

namespace spectra {
namespace ui {

// Tile size for tile-based rendering (must match CUDA block size)
constexpr int TILE_SIZE = 16;

// Maximum quads per tile (for fixed-size tile data)
constexpr int MAX_QUADS_PER_TILE = 64;

// Tile data structure for GPU - stores quad indices for each tile
struct TileData {
    uint16_t quadIndices[MAX_QUADS_PER_TILE];  // Indices of quads overlapping this tile
    uint16_t quadCount;                         // Number of quads in this tile
    uint16_t padding;                           // Alignment padding
};

//------------------------------------------------------------------------------
// UI Renderer - CUDA-based UI rendering with tile-based optimization
//------------------------------------------------------------------------------
class UIRenderer {
public:
    UIRenderer() = default;
    ~UIRenderer();

    // Non-copyable
    UIRenderer(const UIRenderer&) = delete;
    UIRenderer& operator=(const UIRenderer&) = delete;

    // Initialize the renderer
    bool init(uint32_t maxQuads = 4096);

    // Release resources
    void shutdown();

    // Set textures for UI preview quads (call before render)
    void setTextures(const cudaTextureObject_t* textures, uint32_t count);

    // Render UI quads to the output buffer
    void render(const std::vector<UIQuad>& quads,
                cudaTextureObject_t fontAtlas,
                float4* outputBuffer,
                uint32_t width, uint32_t height,
                cudaStream_t stream = nullptr);

    // Set SDF parameters for text rendering
    void setSDFParams(float threshold, float smoothing);

private:
    // Build tile data structure on CPU
    void buildTileData(const std::vector<UIQuad>& quads,
                       uint32_t width, uint32_t height,
                       int startX, int startY, int endX, int endY);

    UIQuad* m_deviceQuads = nullptr;
    uint32_t m_maxQuads = 0;

    float m_sdfThreshold = 0.5f;
    float m_sdfSmoothing = 0.1f;

    // Texture array for QUAD_FLAG_TEXTURE
    cudaTextureObject_t* m_deviceTextures = nullptr;
    uint32_t m_textureCount = 0;

    // Tile-based rendering data
    TileData* m_deviceTileData = nullptr;
    std::vector<TileData> m_hostTileData;
    uint32_t m_tileCountX = 0;
    uint32_t m_tileCountY = 0;
    uint32_t m_maxTiles = 0;

    // Cached state
    uint32_t m_lastQuadCount = 0;
    
    // Cache to avoid re-rendering unchanged UI
    uint64_t m_lastGeneration = 0;
    
public:
    // Call to invalidate cache (forces re-render next frame)
    void invalidate() { m_lastGeneration = 0; }
    
    // Render only if generation changed (returns true if rendered)
    bool renderIfChanged(const std::vector<UIQuad>& quads,
                        uint64_t generation,
                        cudaTextureObject_t fontAtlas,
                        float4* outputBuffer,
                        uint32_t width, uint32_t height,
                        cudaStream_t stream = nullptr);
};

} // namespace ui
} // namespace spectra
