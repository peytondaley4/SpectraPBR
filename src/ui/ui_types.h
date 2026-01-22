#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace spectra {
namespace ui {

//------------------------------------------------------------------------------
// UI Quad Flags
//------------------------------------------------------------------------------
constexpr uint32_t QUAD_FLAG_SOLID = 0;      // Solid color quad
constexpr uint32_t QUAD_FLAG_TEXT  = 1;      // SDF text quad (sample font atlas)

//------------------------------------------------------------------------------
// UI Vertex Structure (32 bytes)
// Used for rendering UI quads on the GPU
//------------------------------------------------------------------------------
struct UIVertex {
    float2 position;    // Screen position in pixels (8 bytes)
    float2 uv;          // Texture/atlas coordinates (8 bytes)
    float4 color;       // RGBA color (16 bytes)
};
static_assert(sizeof(UIVertex) == 32, "UIVertex must be 32 bytes");

//------------------------------------------------------------------------------
// UI Quad Structure
// Represents a single renderable UI element (4 vertices + metadata)
//------------------------------------------------------------------------------
struct UIQuad {
    UIVertex vertices[4];   // TL, TR, BL, BR order
    float depth;            // Z-order (higher = on top)
    uint32_t flags;         // QUAD_FLAG_SOLID or QUAD_FLAG_TEXT
    float clipMinX;         // Clip rect (0,0,0,0 = no clipping)
    float clipMinY;
    float clipMaxX;
    float clipMaxY;
};

//------------------------------------------------------------------------------
// Glyph Metrics (for font atlas lookup)
//------------------------------------------------------------------------------
struct GlyphMetrics {
    float2 uvMin;           // Top-left UV in atlas
    float2 uvMax;           // Bottom-right UV in atlas
    float2 size;            // Glyph size in pixels
    float2 bearing;         // Offset from baseline
    float advance;          // Horizontal advance to next character
    float _pad;
};

//------------------------------------------------------------------------------
// UI Render Parameters (passed to CUDA kernel)
//------------------------------------------------------------------------------
struct UIRenderParams {
    float4* output_buffer;          // Output buffer to render into
    uint32_t buffer_width;          // Buffer width in pixels
    uint32_t buffer_height;         // Buffer height in pixels
    const UIQuad* quads;            // Array of quads to render
    uint32_t quad_count;            // Number of quads
    cudaTextureObject_t font_atlas; // SDF font atlas texture
    float sdf_threshold;            // SDF threshold for text (typically 0.5)
    float sdf_smoothing;            // SDF smoothing amount for anti-aliasing
};

//------------------------------------------------------------------------------
// Text Alignment
//------------------------------------------------------------------------------
enum class TextAlign : uint32_t {
    Left   = 0,
    Center = 1,
    Right  = 2
};

//------------------------------------------------------------------------------
// Widget State Flags
//------------------------------------------------------------------------------
constexpr uint32_t WIDGET_STATE_NORMAL   = 0;
constexpr uint32_t WIDGET_STATE_HOVERED  = 1;
constexpr uint32_t WIDGET_STATE_ACTIVE   = 2;
constexpr uint32_t WIDGET_STATE_DISABLED = 4;
constexpr uint32_t WIDGET_STATE_FOCUSED  = 8;

//------------------------------------------------------------------------------
// Rect Helper (for layout calculations)
//------------------------------------------------------------------------------
struct Rect {
    float x, y;         // Top-left position
    float width, height;

    bool contains(float px, float py) const {
        return px >= x && px < x + width &&
               py >= y && py < y + height;
    }

    bool contains(float2 p) const {
        return contains(p.x, p.y);
    }

    float right() const { return x + width; }
    float bottom() const { return y + height; }
    float2 center() const { return make_float2(x + width * 0.5f, y + height * 0.5f); }
};

//------------------------------------------------------------------------------
// Helper Functions
//------------------------------------------------------------------------------
inline UIQuad makeSolidQuad(const Rect& rect, float4 color, float depth) {
    UIQuad quad;

    // Top-left
    quad.vertices[0].position = make_float2(rect.x, rect.y);
    quad.vertices[0].uv = make_float2(0.0f, 0.0f);
    quad.vertices[0].color = color;

    // Top-right
    quad.vertices[1].position = make_float2(rect.x + rect.width, rect.y);
    quad.vertices[1].uv = make_float2(1.0f, 0.0f);
    quad.vertices[1].color = color;

    // Bottom-left
    quad.vertices[2].position = make_float2(rect.x, rect.y + rect.height);
    quad.vertices[2].uv = make_float2(0.0f, 1.0f);
    quad.vertices[2].color = color;

    // Bottom-right
    quad.vertices[3].position = make_float2(rect.x + rect.width, rect.y + rect.height);
    quad.vertices[3].uv = make_float2(1.0f, 1.0f);
    quad.vertices[3].color = color;

    quad.depth = depth;
    quad.flags = QUAD_FLAG_SOLID;
    quad.clipMinX = 0.0f;
    quad.clipMinY = 0.0f;
    quad.clipMaxX = 0.0f;
    quad.clipMaxY = 0.0f;

    return quad;
}

inline UIQuad makeTextQuad(float2 pos, float2 size, float2 uvMin, float2 uvMax,
                           float4 color, float depth) {
    UIQuad quad;

    // Top-left
    quad.vertices[0].position = pos;
    quad.vertices[0].uv = make_float2(uvMin.x, uvMin.y);
    quad.vertices[0].color = color;

    // Top-right
    quad.vertices[1].position = make_float2(pos.x + size.x, pos.y);
    quad.vertices[1].uv = make_float2(uvMax.x, uvMin.y);
    quad.vertices[1].color = color;

    // Bottom-left
    quad.vertices[2].position = make_float2(pos.x, pos.y + size.y);
    quad.vertices[2].uv = make_float2(uvMin.x, uvMax.y);
    quad.vertices[2].color = color;

    // Bottom-right
    quad.vertices[3].position = make_float2(pos.x + size.x, pos.y + size.y);
    quad.vertices[3].uv = make_float2(uvMax.x, uvMax.y);
    quad.vertices[3].color = color;

    quad.depth = depth;
    quad.flags = QUAD_FLAG_TEXT;
    quad.clipMinX = 0.0f;
    quad.clipMinY = 0.0f;
    quad.clipMaxX = 0.0f;
    quad.clipMaxY = 0.0f;

    return quad;
}

} // namespace ui
} // namespace spectra
