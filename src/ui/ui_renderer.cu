#include "ui_renderer.h"
#include <iostream>
#include <cmath>
#include <algorithm>

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

// Rasterize a single quad - one block per quad, threads cover the quad's bounding box
__global__ void rasterizeQuadKernel(
    const UIQuad* quad,
    cudaTextureObject_t fontAtlas,
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

    std::cout << "[UIRenderer] Initialized with max " << maxQuads << " quads\n";
    return true;
}

void UIRenderer::shutdown() {
    if (m_deviceQuads) {
        cudaFree(m_deviceQuads);
        m_deviceQuads = nullptr;
    }
    m_maxQuads = 0;
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

    // Copy quads to device
    cudaMemcpyAsync(m_deviceQuads, quads.data(), quadCount * sizeof(UIQuad),
                    cudaMemcpyHostToDevice, stream);

    // Rasterize each quad with its own kernel launch
    // Quads are already sorted back-to-front, so we render in order for correct blending
    dim3 blockSize(16, 16);
    
    for (uint32_t q = 0; q < quadCount; q++) {
        const UIQuad& quad = quads[q];
        
        // Calculate bounding box
        float minXf = fminf(fminf(quad.vertices[0].position.x, quad.vertices[1].position.x),
                           fminf(quad.vertices[2].position.x, quad.vertices[3].position.x));
        float maxXf = fmaxf(fmaxf(quad.vertices[0].position.x, quad.vertices[1].position.x),
                           fmaxf(quad.vertices[2].position.x, quad.vertices[3].position.x));
        float minYf = fminf(fminf(quad.vertices[0].position.y, quad.vertices[1].position.y),
                           fminf(quad.vertices[2].position.y, quad.vertices[3].position.y));
        float maxYf = fmaxf(fmaxf(quad.vertices[0].position.y, quad.vertices[1].position.y),
                           fmaxf(quad.vertices[2].position.y, quad.vertices[3].position.y));
        
        int minX = static_cast<int>(floorf(minXf));
        int minY = static_cast<int>(floorf(minYf));
        int maxX = static_cast<int>(ceilf(maxXf));
        int maxY = static_cast<int>(ceilf(maxYf));
        
        // Clamp to screen bounds
        minX = std::max(0, minX);
        minY = std::max(0, minY);
        maxX = std::min(static_cast<int>(width) - 1, maxX);
        maxY = std::min(static_cast<int>(height) - 1, maxY);
        
        if (minX > maxX || minY > maxY) continue;
        
        int quadWidth = maxX - minX + 1;
        int quadHeight = maxY - minY + 1;
        
        dim3 gridSize((quadWidth + blockSize.x - 1) / blockSize.x,
                      (quadHeight + blockSize.y - 1) / blockSize.y);
        
        rasterizeQuadKernel<<<gridSize, blockSize, 0, stream>>>(
            m_deviceQuads + q, fontAtlas, outputBuffer,
            width, height, m_sdfThreshold, m_sdfSmoothing,
            minX, minY, maxX, maxY);
    }
}

void UIRenderer::setSDFParams(float threshold, float smoothing) {
    m_sdfThreshold = threshold;
    m_sdfSmoothing = smoothing;
}

} // namespace ui
} // namespace spectra
