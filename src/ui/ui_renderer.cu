#include "ui_renderer.h"
#include <iostream>

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

// Main UI rendering kernel
// Renders one quad per block (simple approach for now)
__global__ void renderQuadsKernel(
    const UIQuad* quads,
    uint32_t quadCount,
    cudaTextureObject_t fontAtlas,
    float4* output,
    uint32_t width,
    uint32_t height,
    float sdfThreshold,
    float sdfSmoothing
) {
    uint32_t px = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t py = blockIdx.y * blockDim.y + threadIdx.y;

    if (px >= width || py >= height) return;

    float2 p = make_float2(static_cast<float>(px) + 0.5f, static_cast<float>(py) + 0.5f);

    // Accumulate color with alpha blending (back to front)
    float4 result = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    for (uint32_t q = 0; q < quadCount; q++) {
        const UIQuad& quad = quads[q];

        // Get quad vertices (TL, TR, BL, BR)
        float2 v0 = quad.vertices[0].position;  // TL
        float2 v1 = quad.vertices[1].position;  // TR
        float2 v2 = quad.vertices[2].position;  // BL
        float2 v3 = quad.vertices[3].position;  // BR

        // Quick AABB test
        float minX = fminf(fminf(v0.x, v1.x), fminf(v2.x, v3.x));
        float maxX = fmaxf(fmaxf(v0.x, v1.x), fmaxf(v2.x, v3.x));
        float minY = fminf(fminf(v0.y, v1.y), fminf(v2.y, v3.y));
        float maxY = fmaxf(fmaxf(v0.y, v1.y), fmaxf(v2.y, v3.y));

        if (p.x < minX || p.x > maxX || p.y < minY || p.y > maxY) {
            continue;
        }

        // Check if point is inside quad (split into two triangles)
        // Triangle 1: TL, TR, BL (v0, v1, v2)
        // Triangle 2: TR, BR, BL (v1, v3, v2)

        float4 color = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        float2 uv = make_float2(0.0f, 0.0f);
        bool inside = false;

        // Check triangle 1 (TL, TR, BL)
        float area1 = edgeFunction(v0, v1, v2);
        float w0_1 = edgeFunction(v1, v2, p);
        float w1_1 = edgeFunction(v2, v0, p);
        float w2_1 = edgeFunction(v0, v1, p);

        if (w0_1 >= 0 && w1_1 >= 0 && w2_1 >= 0) {
            inside = true;
            float invArea = 1.0f / area1;
            w0_1 *= invArea;
            w1_1 *= invArea;
            w2_1 *= invArea;

            // Interpolate color
            color.x = quad.vertices[0].color.x * w0_1 + quad.vertices[1].color.x * w1_1 + quad.vertices[2].color.x * w2_1;
            color.y = quad.vertices[0].color.y * w0_1 + quad.vertices[1].color.y * w1_1 + quad.vertices[2].color.y * w2_1;
            color.z = quad.vertices[0].color.z * w0_1 + quad.vertices[1].color.z * w1_1 + quad.vertices[2].color.z * w2_1;
            color.w = quad.vertices[0].color.w * w0_1 + quad.vertices[1].color.w * w1_1 + quad.vertices[2].color.w * w2_1;

            // Interpolate UV
            uv.x = quad.vertices[0].uv.x * w0_1 + quad.vertices[1].uv.x * w1_1 + quad.vertices[2].uv.x * w2_1;
            uv.y = quad.vertices[0].uv.y * w0_1 + quad.vertices[1].uv.y * w1_1 + quad.vertices[2].uv.y * w2_1;
        }

        // Check triangle 2 (TR, BR, BL)
        if (!inside) {
            float area2 = edgeFunction(v1, v3, v2);
            float w0_2 = edgeFunction(v3, v2, p);
            float w1_2 = edgeFunction(v2, v1, p);
            float w2_2 = edgeFunction(v1, v3, p);

            if (w0_2 >= 0 && w1_2 >= 0 && w2_2 >= 0) {
                inside = true;
                float invArea = 1.0f / area2;
                w0_2 *= invArea;
                w1_2 *= invArea;
                w2_2 *= invArea;

                // Interpolate color (indices: TR=1, BR=3, BL=2)
                color.x = quad.vertices[1].color.x * w0_2 + quad.vertices[3].color.x * w1_2 + quad.vertices[2].color.x * w2_2;
                color.y = quad.vertices[1].color.y * w0_2 + quad.vertices[3].color.y * w1_2 + quad.vertices[2].color.y * w2_2;
                color.z = quad.vertices[1].color.z * w0_2 + quad.vertices[3].color.z * w1_2 + quad.vertices[2].color.z * w2_2;
                color.w = quad.vertices[1].color.w * w0_2 + quad.vertices[3].color.w * w1_2 + quad.vertices[2].color.w * w2_2;

                // Interpolate UV
                uv.x = quad.vertices[1].uv.x * w0_2 + quad.vertices[3].uv.x * w1_2 + quad.vertices[2].uv.x * w2_2;
                uv.y = quad.vertices[1].uv.y * w0_2 + quad.vertices[3].uv.y * w1_2 + quad.vertices[2].uv.y * w2_2;
            }
        }

        if (!inside) continue;

        // Handle text quads with SDF sampling
        if (quad.flags == QUAD_FLAG_TEXT && fontAtlas != 0) {
            // Sample SDF from font atlas
            float sdfValue = tex2D<float>(fontAtlas, uv.x, uv.y);

            // Calculate alpha using smoothstep for anti-aliasing
            float alpha = smoothstep(sdfThreshold - sdfSmoothing,
                                     sdfThreshold + sdfSmoothing,
                                     sdfValue);
            color.w *= alpha;
        }

        // Alpha blend (over operator)
        if (color.w > 0.0f) {
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
    }

    output[py * width + px] = result;
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
    if (quads.empty() || !outputBuffer) {
        // Clear buffer to transparent if no quads
        dim3 blockSize(16, 16);
        dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                      (height + blockSize.y - 1) / blockSize.y);
        clearBufferKernel<<<gridSize, blockSize, 0, stream>>>(outputBuffer, width, height);
        return;
    }

    uint32_t quadCount = static_cast<uint32_t>(std::min(quads.size(), static_cast<size_t>(m_maxQuads)));

    // Copy quads to device
    cudaMemcpyAsync(m_deviceQuads, quads.data(), quadCount * sizeof(UIQuad),
                    cudaMemcpyHostToDevice, stream);

    // Launch render kernel
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);

    renderQuadsKernel<<<gridSize, blockSize, 0, stream>>>(
        m_deviceQuads, quadCount, fontAtlas, outputBuffer,
        width, height, m_sdfThreshold, m_sdfSmoothing);
}

void UIRenderer::setSDFParams(float threshold, float smoothing) {
    m_sdfThreshold = threshold;
    m_sdfSmoothing = smoothing;
}

} // namespace ui
} // namespace spectra
