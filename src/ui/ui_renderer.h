#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <vector>
#include "ui_types.h"

namespace spectra {
namespace ui {

//------------------------------------------------------------------------------
// UI Renderer - CUDA-based UI rendering to a separate buffer
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
    // Clears the buffer first (to transparent), then renders quads
    void render(const std::vector<UIQuad>& quads,
                cudaTextureObject_t fontAtlas,
                float4* outputBuffer,
                uint32_t width, uint32_t height,
                cudaStream_t stream = nullptr);

    // Set SDF parameters for text rendering
    void setSDFParams(float threshold, float smoothing);

private:
    UIQuad* m_deviceQuads = nullptr;
    uint32_t m_maxQuads = 0;

    float m_sdfThreshold = 0.5f;
    float m_sdfSmoothing = 0.1f;

    // Texture array for QUAD_FLAG_TEXTURE
    cudaTextureObject_t* m_deviceTextures = nullptr;
    uint32_t m_textureCount = 0;
};

} // namespace ui
} // namespace spectra
