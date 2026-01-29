#include "ui_renderer.h"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <array>

namespace spectra {
namespace ui {

//------------------------------------------------------------------------------
// CUDA Kernels
//------------------------------------------------------------------------------

// Clear buffer to transparent
__global__ void clearBufferKernel(float4* buffer, uint32_t width, uint32_t height) {
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    buffer[y * width + x] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
}

// Check if a point is inside a triangle (for quad rendering)
__device__ float edgeFunction(float2 a, float2 b, float2 c) {
    return (c.x - a.x) * (b.y - a.y) - (c.y - a.y) * (b.x - a.x);
}

// Bilinear interpolation for UV coordinates
__device__ float2 barycentricInterpolate(float2 v0, float2 v1, float2 v2,
                                          float w0, float w1, float w2) {
    return make_float2(
        v0.x * w0 + v1.x * w1 + v2.x * w2,
        v0.y * w0 + v1.y * w1 + v2.y * w2
    );
}

// Smoothstep function (not built-in to CUDA)
__device__ float smoothstep(float edge0, float edge1, float x) {
    float t = fminf(fmaxf((x - edge0) / (edge1 - edge0), 0.0f), 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

// Batched quad bounds structure for GPU
struct QuadBounds {
    int minX, minY, maxX, maxY;
};

// Batched rasterization - processes all quads in one kernel launch
// Each block handles a tile, threads iterate through quads that overlap the tile
__global__ void rasterizeQuadsBatchedKernel(
    const UIQuad* quads,
    uint32_t quadCount,
    cudaTextureObject_t fontAtlas,
    const cudaTextureObject_t* textures,
    uint32_t textureCount,
    float4* output,
    uint32_t width,
    uint32_t height,
    float sdfThreshold,
    float sdfSmoothing
) {
    // Each thread handles one pixel
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (px >= (int)width || py >= (int)height) return;
    
    float2 p = make_float2(static_cast<float>(px) + 0.5f, static_cast<float>(py) + 0.5f);
    uint32_t idx = py * width + px;
    float4 result = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    
    // Process quads back-to-front (they're already sorted)
    for (uint32_t q = 0; q < quadCount; q++) {
        const UIQuad& quad = quads[q];
        
        // Quick AABB test
        float minXf = fminf(fminf(quad.vertices[0].position.x, quad.vertices[1].position.x),
                           fminf(quad.vertices[2].position.x, quad.vertices[3].position.x));
        float maxXf = fmaxf(fmaxf(quad.vertices[0].position.x, quad.vertices[1].position.x),
                           fmaxf(quad.vertices[2].position.x, quad.vertices[3].position.x));
        float minYf = fminf(fminf(quad.vertices[0].position.y, quad.vertices[1].position.y),
                           fminf(quad.vertices[2].position.y, quad.vertices[3].position.y));
        float maxYf = fmaxf(fmaxf(quad.vertices[0].position.y, quad.vertices[1].position.y),
                           fmaxf(quad.vertices[2].position.y, quad.vertices[3].position.y));
        
        if (p.x < minXf || p.x > maxXf || p.y < minYf || p.y > maxYf) continue;
        
        // Check clip bounds
        bool hasClip = (quad.clipMaxX > quad.clipMinX) && (quad.clipMaxY > quad.clipMinY);
        if (hasClip) {
            if (px < quad.clipMinX || px >= quad.clipMaxX ||
                py < quad.clipMinY || py >= quad.clipMaxY) {
                continue;
            }
        }
        
        // Get quad vertices
        float2 v0 = quad.vertices[0].position;  // TL
        float2 v1 = quad.vertices[1].position;  // TR
        float2 v2 = quad.vertices[2].position;  // BL
        float2 v3 = quad.vertices[3].position;  // BR
        
        float4 color = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        float2 uv = make_float2(0.0f, 0.0f);
        bool inside = false;
        
        // Check triangle 1 (TL, BL, TR)
        float area1 = edgeFunction(v0, v2, v1);
        float w0_1 = edgeFunction(v2, v1, p);
        float w1_1 = edgeFunction(v1, v0, p);
        float w2_1 = edgeFunction(v0, v2, p);
        
        if (w0_1 >= 0 && w1_1 >= 0 && w2_1 >= 0) {
            inside = true;
            float invArea = 1.0f / area1;
            w0_1 *= invArea;
            w1_1 *= invArea;
            w2_1 *= invArea;
            
            color.x = quad.vertices[0].color.x * w0_1 + quad.vertices[2].color.x * w1_1 + quad.vertices[1].color.x * w2_1;
            color.y = quad.vertices[0].color.y * w0_1 + quad.vertices[2].color.y * w1_1 + quad.vertices[1].color.y * w2_1;
            color.z = quad.vertices[0].color.z * w0_1 + quad.vertices[2].color.z * w1_1 + quad.vertices[1].color.z * w2_1;
            color.w = quad.vertices[0].color.w * w0_1 + quad.vertices[2].color.w * w1_1 + quad.vertices[1].color.w * w2_1;
            
            uv.x = quad.vertices[0].uv.x * w0_1 + quad.vertices[2].uv.x * w1_1 + quad.vertices[1].uv.x * w2_1;
            uv.y = quad.vertices[0].uv.y * w0_1 + quad.vertices[2].uv.y * w1_1 + quad.vertices[1].uv.y * w2_1;
        }
        
        // Check triangle 2 (BL, BR, TR)
        if (!inside) {
            float area2 = edgeFunction(v2, v3, v1);
            float w0_2 = edgeFunction(v3, v1, p);
            float w1_2 = edgeFunction(v1, v2, p);
            float w2_2 = edgeFunction(v2, v3, p);
            
            if (w0_2 >= 0 && w1_2 >= 0 && w2_2 >= 0) {
                inside = true;
                float invArea = 1.0f / area2;
                w0_2 *= invArea;
                w1_2 *= invArea;
                w2_2 *= invArea;
                
                color.x = quad.vertices[2].color.x * w0_2 + quad.vertices[3].color.x * w1_2 + quad.vertices[1].color.x * w2_2;
                color.y = quad.vertices[2].color.y * w0_2 + quad.vertices[3].color.y * w1_2 + quad.vertices[1].color.y * w2_2;
                color.z = quad.vertices[2].color.z * w0_2 + quad.vertices[3].color.z * w1_2 + quad.vertices[1].color.z * w2_2;
                color.w = quad.vertices[2].color.w * w0_2 + quad.vertices[3].color.w * w1_2 + quad.vertices[1].color.w * w2_2;
                
                uv.x = quad.vertices[2].uv.x * w0_2 + quad.vertices[3].uv.x * w1_2 + quad.vertices[1].uv.x * w2_2;
                uv.y = quad.vertices[2].uv.y * w0_2 + quad.vertices[3].uv.y * w1_2 + quad.vertices[1].uv.y * w2_2;
            }
        }
        
        if (!inside) continue;
        
        // Handle text quads with SDF sampling
        if (quad.flags == QUAD_FLAG_TEXT && fontAtlas != 0) {
            float sdfValue = tex2D<float>(fontAtlas, uv.x, uv.y);
            float alpha = smoothstep(sdfThreshold - sdfSmoothing, sdfThreshold + sdfSmoothing, sdfValue);
            color.w *= alpha;
        }
        // Handle textured quads
        else if (quad.flags == QUAD_FLAG_TEXTURE && quad.textureIndex < textureCount && textures != nullptr) {
            cudaTextureObject_t tex = textures[quad.textureIndex];
            if (tex != 0) {
                float4 texColor = tex2D<float4>(tex, uv.x, uv.y);
                color.x *= texColor.x;
                color.y *= texColor.y;
                color.z *= texColor.z;
                color.w *= texColor.w;
            }
        }
        
        if (color.w <= 0.0f) continue;
        
        // Alpha blend
        float srcAlpha = color.w;
        float dstAlpha = result.w;
        float outAlpha = srcAlpha + dstAlpha * (1.0f - srcAlpha);
        
        if (outAlpha > 0.0f) {
            result.x = (color.x * srcAlpha + result.x * dstAlpha * (1.0f - srcAlpha)) / outAlpha;
            result.y = (color.y * srcAlpha + result.y * dstAlpha * (1.0f - srcAlpha)) / outAlpha;
            result.z = (color.z * srcAlpha + result.z * dstAlpha * (1.0f - srcAlpha)) / outAlpha;
            result.w = outAlpha;
        }
    }
    
    output[idx] = result;
}

// Region-bounded batched kernel - only processes pixels within startX/Y to endX/Y
__global__ void rasterizeQuadsBatchedKernelRegion(
    const UIQuad* quads,
    uint32_t quadCount,
    cudaTextureObject_t fontAtlas,
    const cudaTextureObject_t* textures,
    uint32_t textureCount,
    float4* output,
    uint32_t width,
    uint32_t height,
    float sdfThreshold,
    float sdfSmoothing,
    int startX, int startY, int endX, int endY
) {
    // Each thread handles one pixel within the bounded region
    int localX = blockIdx.x * blockDim.x + threadIdx.x;
    int localY = blockIdx.y * blockDim.y + threadIdx.y;
    
    int px = startX + localX;
    int py = startY + localY;
    
    if (px >= endX || py >= endY || px >= (int)width || py >= (int)height) return;
    
    float2 p = make_float2(static_cast<float>(px) + 0.5f, static_cast<float>(py) + 0.5f);
    uint32_t idx = py * width + px;
    float4 result = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    
    // Process quads back-to-front (they're already sorted)
    for (uint32_t q = 0; q < quadCount; q++) {
        const UIQuad& quad = quads[q];
        
        // Quick AABB test
        float minXf = fminf(fminf(quad.vertices[0].position.x, quad.vertices[1].position.x),
                           fminf(quad.vertices[2].position.x, quad.vertices[3].position.x));
        float maxXf = fmaxf(fmaxf(quad.vertices[0].position.x, quad.vertices[1].position.x),
                           fmaxf(quad.vertices[2].position.x, quad.vertices[3].position.x));
        float minYf = fminf(fminf(quad.vertices[0].position.y, quad.vertices[1].position.y),
                           fminf(quad.vertices[2].position.y, quad.vertices[3].position.y));
        float maxYf = fmaxf(fmaxf(quad.vertices[0].position.y, quad.vertices[1].position.y),
                           fmaxf(quad.vertices[2].position.y, quad.vertices[3].position.y));
        
        if (p.x < minXf || p.x > maxXf || p.y < minYf || p.y > maxYf) continue;
        
        // Check clip bounds
        bool hasClip = (quad.clipMaxX > quad.clipMinX) && (quad.clipMaxY > quad.clipMinY);
        if (hasClip) {
            if (px < quad.clipMinX || px >= quad.clipMaxX ||
                py < quad.clipMinY || py >= quad.clipMaxY) {
                continue;
            }
        }
        
        // Get quad vertices
        float2 v0 = quad.vertices[0].position;  // TL
        float2 v1 = quad.vertices[1].position;  // TR
        float2 v2 = quad.vertices[2].position;  // BL
        float2 v3 = quad.vertices[3].position;  // BR
        
        float4 color = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        float2 uv = make_float2(0.0f, 0.0f);
        bool inside = false;
        
        // Check triangle 1 (TL, BL, TR)
        float area1 = edgeFunction(v0, v2, v1);
        float w0_1 = edgeFunction(v2, v1, p);
        float w1_1 = edgeFunction(v1, v0, p);
        float w2_1 = edgeFunction(v0, v2, p);
        
        if (w0_1 >= 0 && w1_1 >= 0 && w2_1 >= 0) {
            inside = true;
            float invArea = 1.0f / area1;
            w0_1 *= invArea;
            w1_1 *= invArea;
            w2_1 *= invArea;
            
            color.x = quad.vertices[0].color.x * w0_1 + quad.vertices[2].color.x * w1_1 + quad.vertices[1].color.x * w2_1;
            color.y = quad.vertices[0].color.y * w0_1 + quad.vertices[2].color.y * w1_1 + quad.vertices[1].color.y * w2_1;
            color.z = quad.vertices[0].color.z * w0_1 + quad.vertices[2].color.z * w1_1 + quad.vertices[1].color.z * w2_1;
            color.w = quad.vertices[0].color.w * w0_1 + quad.vertices[2].color.w * w1_1 + quad.vertices[1].color.w * w2_1;
            
            uv.x = quad.vertices[0].uv.x * w0_1 + quad.vertices[2].uv.x * w1_1 + quad.vertices[1].uv.x * w2_1;
            uv.y = quad.vertices[0].uv.y * w0_1 + quad.vertices[2].uv.y * w1_1 + quad.vertices[1].uv.y * w2_1;
        }
        
        // Check triangle 2 (BL, BR, TR)
        if (!inside) {
            float area2 = edgeFunction(v2, v3, v1);
            float w0_2 = edgeFunction(v3, v1, p);
            float w1_2 = edgeFunction(v1, v2, p);
            float w2_2 = edgeFunction(v2, v3, p);
            
            if (w0_2 >= 0 && w1_2 >= 0 && w2_2 >= 0) {
                inside = true;
                float invArea = 1.0f / area2;
                w0_2 *= invArea;
                w1_2 *= invArea;
                w2_2 *= invArea;
                
                color.x = quad.vertices[2].color.x * w0_2 + quad.vertices[3].color.x * w1_2 + quad.vertices[1].color.x * w2_2;
                color.y = quad.vertices[2].color.y * w0_2 + quad.vertices[3].color.y * w1_2 + quad.vertices[1].color.y * w2_2;
                color.z = quad.vertices[2].color.z * w0_2 + quad.vertices[3].color.z * w1_2 + quad.vertices[1].color.z * w2_2;
                color.w = quad.vertices[2].color.w * w0_2 + quad.vertices[3].color.w * w1_2 + quad.vertices[1].color.w * w2_2;
                
                uv.x = quad.vertices[2].uv.x * w0_2 + quad.vertices[3].uv.x * w1_2 + quad.vertices[1].uv.x * w2_2;
                uv.y = quad.vertices[2].uv.y * w0_2 + quad.vertices[3].uv.y * w1_2 + quad.vertices[1].uv.y * w2_2;
            }
        }
        
        if (!inside) continue;
        
        // Handle text quads with SDF sampling
        if (quad.flags == QUAD_FLAG_TEXT && fontAtlas != 0) {
            float sdfValue = tex2D<float>(fontAtlas, uv.x, uv.y);
            float alpha = smoothstep(sdfThreshold - sdfSmoothing, sdfThreshold + sdfSmoothing, sdfValue);
            color.w *= alpha;
        }
        // Handle textured quads
        else if (quad.flags == QUAD_FLAG_TEXTURE && quad.textureIndex < textureCount && textures != nullptr) {
            cudaTextureObject_t tex = textures[quad.textureIndex];
            if (tex != 0) {
                float4 texColor = tex2D<float4>(tex, uv.x, uv.y);
                color.x *= texColor.x;
                color.y *= texColor.y;
                color.z *= texColor.z;
                color.w *= texColor.w;
            }
        }
        
        if (color.w <= 0.0f) continue;
        
        // Alpha blend
        float srcAlpha = color.w;
        float dstAlpha = result.w;
        float outAlpha = srcAlpha + dstAlpha * (1.0f - srcAlpha);
        
        if (outAlpha > 0.0f) {
            result.x = (color.x * srcAlpha + result.x * dstAlpha * (1.0f - srcAlpha)) / outAlpha;
            result.y = (color.y * srcAlpha + result.y * dstAlpha * (1.0f - srcAlpha)) / outAlpha;
            result.z = (color.z * srcAlpha + result.z * dstAlpha * (1.0f - srcAlpha)) / outAlpha;
            result.w = outAlpha;
        }
    }
    
    output[idx] = result;
}

// Legacy single-quad kernel for compatibility (unused but kept for reference)
__global__ void rasterizeQuadKernel(
    const UIQuad* quad,
    cudaTextureObject_t fontAtlas,
    const cudaTextureObject_t* textures,
    uint32_t textureCount,
    float4* output,
    uint32_t width,
    uint32_t height,
    float sdfThreshold,
    float sdfSmoothing,
    int minX, int minY, int maxX, int maxY
) {
    // Calculate pixel position within quad's bounding box
    int localX = blockIdx.x * blockDim.x + threadIdx.x;
    int localY = blockIdx.y * blockDim.y + threadIdx.y;
    
    int px = minX + localX;
    int py = minY + localY;
    
    if (px > maxX || py > maxY || px >= (int)width || py >= (int)height || px < 0 || py < 0) return;
    
    // Check clip bounds (if set - all zeros means no clipping)
    bool hasClip = (quad->clipMaxX > quad->clipMinX) && (quad->clipMaxY > quad->clipMinY);
    if (hasClip) {
        if (px < quad->clipMinX || px >= quad->clipMaxX ||
            py < quad->clipMinY || py >= quad->clipMaxY) {
            return; // Outside clip region
        }
    }
    
    float2 p = make_float2(static_cast<float>(px) + 0.5f, static_cast<float>(py) + 0.5f);
    
    // Get quad vertices
    float2 v0 = quad->vertices[0].position;  // TL
    float2 v1 = quad->vertices[1].position;  // TR
    float2 v2 = quad->vertices[2].position;  // BL
    float2 v3 = quad->vertices[3].position;  // BR
    
    float4 color = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float2 uv = make_float2(0.0f, 0.0f);
    bool inside = false;
    
    // Check triangle 1 (TL, BL, TR)
    float area1 = edgeFunction(v0, v2, v1);
    float w0_1 = edgeFunction(v2, v1, p);
    float w1_1 = edgeFunction(v1, v0, p);
    float w2_1 = edgeFunction(v0, v2, p);
    
    if (w0_1 >= 0 && w1_1 >= 0 && w2_1 >= 0) {
        inside = true;
        float invArea = 1.0f / area1;
        w0_1 *= invArea;
        w1_1 *= invArea;
        w2_1 *= invArea;
        
        color.x = quad->vertices[0].color.x * w0_1 + quad->vertices[2].color.x * w1_1 + quad->vertices[1].color.x * w2_1;
        color.y = quad->vertices[0].color.y * w0_1 + quad->vertices[2].color.y * w1_1 + quad->vertices[1].color.y * w2_1;
        color.z = quad->vertices[0].color.z * w0_1 + quad->vertices[2].color.z * w1_1 + quad->vertices[1].color.z * w2_1;
        color.w = quad->vertices[0].color.w * w0_1 + quad->vertices[2].color.w * w1_1 + quad->vertices[1].color.w * w2_1;
        
        uv.x = quad->vertices[0].uv.x * w0_1 + quad->vertices[2].uv.x * w1_1 + quad->vertices[1].uv.x * w2_1;
        uv.y = quad->vertices[0].uv.y * w0_1 + quad->vertices[2].uv.y * w1_1 + quad->vertices[1].uv.y * w2_1;
    }
    
    // Check triangle 2 (BL, BR, TR)
    if (!inside) {
        float area2 = edgeFunction(v2, v3, v1);
        float w0_2 = edgeFunction(v3, v1, p);
        float w1_2 = edgeFunction(v1, v2, p);
        float w2_2 = edgeFunction(v2, v3, p);
        
        if (w0_2 >= 0 && w1_2 >= 0 && w2_2 >= 0) {
            inside = true;
            float invArea = 1.0f / area2;
            w0_2 *= invArea;
            w1_2 *= invArea;
            w2_2 *= invArea;
            
            color.x = quad->vertices[2].color.x * w0_2 + quad->vertices[3].color.x * w1_2 + quad->vertices[1].color.x * w2_2;
            color.y = quad->vertices[2].color.y * w0_2 + quad->vertices[3].color.y * w1_2 + quad->vertices[1].color.y * w2_2;
            color.z = quad->vertices[2].color.z * w0_2 + quad->vertices[3].color.z * w1_2 + quad->vertices[1].color.z * w2_2;
            color.w = quad->vertices[2].color.w * w0_2 + quad->vertices[3].color.w * w1_2 + quad->vertices[1].color.w * w2_2;
            
            uv.x = quad->vertices[2].uv.x * w0_2 + quad->vertices[3].uv.x * w1_2 + quad->vertices[1].uv.x * w2_2;
            uv.y = quad->vertices[2].uv.y * w0_2 + quad->vertices[3].uv.y * w1_2 + quad->vertices[1].uv.y * w2_2;
        }
    }
    
    if (!inside) return;
    
    // Handle text quads with SDF sampling
    if (quad->flags == QUAD_FLAG_TEXT && fontAtlas != 0) {
        float sdfValue = tex2D<float>(fontAtlas, uv.x, uv.y);
        float alpha = smoothstep(sdfThreshold - sdfSmoothing, sdfThreshold + sdfSmoothing, sdfValue);
        color.w *= alpha;
    }
    // Handle textured quads
    else if (quad->flags == QUAD_FLAG_TEXTURE && quad->textureIndex < textureCount && textures != nullptr) {
        cudaTextureObject_t tex = textures[quad->textureIndex];
        if (tex != 0) {
            float4 texColor = tex2D<float4>(tex, uv.x, uv.y);
            // Multiply texture color with tint color
            color.x *= texColor.x;
            color.y *= texColor.y;
            color.z *= texColor.z;
            color.w *= texColor.w;
        }
    }
    
    if (color.w <= 0.0f) return;
    
    // Alpha blend with existing pixel (using atomics would be expensive, so we use simple over blend)
    uint32_t idx = py * width + px;
    float4 dst = output[idx];
    
    float srcAlpha = color.w;
    float dstAlpha = dst.w;
    float outAlpha = srcAlpha + dstAlpha * (1.0f - srcAlpha);
    
    if (outAlpha > 0.0f) {
        float4 result;
        result.x = (color.x * srcAlpha + dst.x * dstAlpha * (1.0f - srcAlpha)) / outAlpha;
        result.y = (color.y * srcAlpha + dst.y * dstAlpha * (1.0f - srcAlpha)) / outAlpha;
        result.z = (color.z * srcAlpha + dst.z * dstAlpha * (1.0f - srcAlpha)) / outAlpha;
        result.w = outAlpha;
        output[idx] = result;
    }
}

//------------------------------------------------------------------------------
// UIRenderer Implementation
//------------------------------------------------------------------------------

UIRenderer::~UIRenderer() {
    shutdown();
}

bool UIRenderer::init(uint32_t maxQuads) {
    m_maxQuads = maxQuads;

    cudaError_t err = cudaMalloc(&m_deviceQuads, maxQuads * sizeof(UIQuad));
    if (err != cudaSuccess) {
        std::cerr << "[UIRenderer] Failed to allocate device memory: "
                  << cudaGetErrorString(err) << "\n";
        return false;
    }

    // Allocate texture array
    err = cudaMalloc(&m_deviceTextures, MAX_UI_TEXTURES * sizeof(cudaTextureObject_t));
    if (err != cudaSuccess) {
        std::cerr << "[UIRenderer] Failed to allocate texture array: "
                  << cudaGetErrorString(err) << "\n";
        cudaFree(m_deviceQuads);
        m_deviceQuads = nullptr;
        return false;
    }

    std::cout << "[UIRenderer] Initialized with max " << maxQuads << " quads\n";
    return true;
}

void UIRenderer::shutdown() {
    if (m_deviceQuads) {
        cudaFree(m_deviceQuads);
        m_deviceQuads = nullptr;
    }
    if (m_deviceTextures) {
        cudaFree(m_deviceTextures);
        m_deviceTextures = nullptr;
    }
    m_maxQuads = 0;
    m_textureCount = 0;
}

void UIRenderer::setTextures(const cudaTextureObject_t* textures, uint32_t count) {
    if (!m_deviceTextures || !textures) return;
    m_textureCount = std::min(count, static_cast<uint32_t>(MAX_UI_TEXTURES));
    if (m_textureCount > 0) {
        cudaMemcpy(m_deviceTextures, textures, m_textureCount * sizeof(cudaTextureObject_t),
                   cudaMemcpyHostToDevice);
    }
}

void UIRenderer::render(const std::vector<UIQuad>& quads,
                        cudaTextureObject_t fontAtlas,
                        float4* outputBuffer,
                        uint32_t width, uint32_t height,
                        cudaStream_t stream) {
    // Always clear the buffer first
    dim3 clearBlockSize(16, 16);
    dim3 clearGridSize((width + clearBlockSize.x - 1) / clearBlockSize.x,
                       (height + clearBlockSize.y - 1) / clearBlockSize.y);
    clearBufferKernel<<<clearGridSize, clearBlockSize, 0, stream>>>(outputBuffer, width, height);
    
    if (quads.empty() || !outputBuffer) {
        return;
    }

    uint32_t quadCount = static_cast<uint32_t>(std::min(quads.size(), static_cast<size_t>(m_maxQuads)));

    // Compute per-quad bounding boxes
    struct QuadRegion {
        int minX, minY, maxX, maxY;
        uint32_t startIdx, endIdx;  // Range of quads in this region
    };
    
    std::vector<QuadRegion> regions;
    
    // Build regions by detecting horizontal gaps (common case: left panel, right panel)
    // First pass: compute bounds for each quad
    std::vector<std::array<int, 4>> quadBounds(quadCount);
    for (uint32_t q = 0; q < quadCount; q++) {
        const UIQuad& quad = quads[q];
        float minXf = width, minYf = height, maxXf = 0, maxYf = 0;
        for (int v = 0; v < 4; v++) {
            minXf = std::min(minXf, quad.vertices[v].position.x);
            minYf = std::min(minYf, quad.vertices[v].position.y);
            maxXf = std::max(maxXf, quad.vertices[v].position.x);
            maxYf = std::max(maxYf, quad.vertices[v].position.y);
        }
        quadBounds[q] = {
            std::max(0, static_cast<int>(std::floor(minXf))),
            std::max(0, static_cast<int>(std::floor(minYf))),
            std::min(static_cast<int>(width), static_cast<int>(std::ceil(maxXf))),
            std::min(static_cast<int>(height), static_cast<int>(std::ceil(maxYf)))
        };
    }
    
    // Detect disjoint horizontal regions (threshold: 100px gap)
    const int GAP_THRESHOLD = 100;
    int screenMidX = width / 2;
    
    // Separate quads into left and right groups
    std::vector<uint32_t> leftQuads, rightQuads;
    int leftMaxX = 0, rightMinX = width;
    
    for (uint32_t q = 0; q < quadCount; q++) {
        int quadCenterX = (quadBounds[q][0] + quadBounds[q][2]) / 2;
        if (quadCenterX < screenMidX) {
            leftQuads.push_back(q);
            leftMaxX = std::max(leftMaxX, quadBounds[q][2]);
        } else {
            rightQuads.push_back(q);
            rightMinX = std::min(rightMinX, quadBounds[q][0]);
        }
    }
    
    // Check if there's a significant gap between left and right
    bool useMultiRegion = !leftQuads.empty() && !rightQuads.empty() && 
                          (rightMinX - leftMaxX) > GAP_THRESHOLD;
    
    // Copy quads to device
    cudaMemcpyAsync(m_deviceQuads, quads.data(), quadCount * sizeof(UIQuad),
                    cudaMemcpyHostToDevice, stream);
    
    dim3 blockSize(16, 16);
    
    if (useMultiRegion) {
        // Render left region
        int leftMinX = width, leftMinY = height, leftMaxY = 0;
        for (uint32_t q : leftQuads) {
            leftMinX = std::min(leftMinX, quadBounds[q][0]);
            leftMinY = std::min(leftMinY, quadBounds[q][1]);
            leftMaxX = std::max(leftMaxX, quadBounds[q][2]);
            leftMaxY = std::max(leftMaxY, quadBounds[q][3]);
        }
        
        if (leftMinX < leftMaxX && leftMinY < leftMaxY) {
            int rw = leftMaxX - leftMinX;
            int rh = leftMaxY - leftMinY;
            dim3 gridSize((rw + blockSize.x - 1) / blockSize.x,
                          (rh + blockSize.y - 1) / blockSize.y);
            rasterizeQuadsBatchedKernelRegion<<<gridSize, blockSize, 0, stream>>>(
                m_deviceQuads, quadCount, fontAtlas, m_deviceTextures, m_textureCount,
                outputBuffer, width, height, m_sdfThreshold, m_sdfSmoothing,
                leftMinX, leftMinY, leftMaxX, leftMaxY);
        }
        
        // Render right region
        int rightMaxX = 0, rightMinY = height, rightMaxY = 0;
        for (uint32_t q : rightQuads) {
            rightMinX = std::min(rightMinX, quadBounds[q][0]);
            rightMinY = std::min(rightMinY, quadBounds[q][1]);
            rightMaxX = std::max(rightMaxX, quadBounds[q][2]);
            rightMaxY = std::max(rightMaxY, quadBounds[q][3]);
        }
        
        if (rightMinX < rightMaxX && rightMinY < rightMaxY) {
            int rw = rightMaxX - rightMinX;
            int rh = rightMaxY - rightMinY;
            dim3 gridSize((rw + blockSize.x - 1) / blockSize.x,
                          (rh + blockSize.y - 1) / blockSize.y);
            rasterizeQuadsBatchedKernelRegion<<<gridSize, blockSize, 0, stream>>>(
                m_deviceQuads, quadCount, fontAtlas, m_deviceTextures, m_textureCount,
                outputBuffer, width, height, m_sdfThreshold, m_sdfSmoothing,
                rightMinX, rightMinY, rightMaxX, rightMaxY);
        }
    } else {
        // Single region - original path
        int startX = width, startY = height, endX = 0, endY = 0;
        for (uint32_t q = 0; q < quadCount; q++) {
            startX = std::min(startX, quadBounds[q][0]);
            startY = std::min(startY, quadBounds[q][1]);
            endX = std::max(endX, quadBounds[q][2]);
            endY = std::max(endY, quadBounds[q][3]);
        }
        
        if (startX >= endX || startY >= endY) return;
        
        int renderWidth = endX - startX;
        int renderHeight = endY - startY;
        
        dim3 gridSize((renderWidth + blockSize.x - 1) / blockSize.x,
                      (renderHeight + blockSize.y - 1) / blockSize.y);
        
        rasterizeQuadsBatchedKernelRegion<<<gridSize, blockSize, 0, stream>>>(
            m_deviceQuads, quadCount, fontAtlas, m_deviceTextures, m_textureCount,
            outputBuffer, width, height, m_sdfThreshold, m_sdfSmoothing,
            startX, startY, endX, endY);
    }
}

void UIRenderer::setSDFParams(float threshold, float smoothing) {
    m_sdfThreshold = threshold;
    m_sdfSmoothing = smoothing;
}

} // namespace ui
} // namespace spectra
