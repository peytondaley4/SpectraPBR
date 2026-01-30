#include "ui_renderer.h"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <cstring>

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

// Edge function for triangle rasterization
__device__ float edgeFunction(float2 a, float2 b, float2 c) {
    return (c.x - a.x) * (b.y - a.y) - (c.y - a.y) * (b.x - a.x);
}

// Smoothstep function
__device__ float smoothstep(float edge0, float edge1, float x) {
    float t = fminf(fmaxf((x - edge0) / (edge1 - edge0), 0.0f), 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

//------------------------------------------------------------------------------
// Tile-based rasterization kernel
// Each block handles one tile, only processing quads that overlap that tile
//------------------------------------------------------------------------------
__global__ void rasterizeTiledKernel(
    const UIQuad* __restrict__ quads,
    const TileData* __restrict__ tileData,
    uint32_t tileCountX,
    uint32_t tileCountY,
    cudaTextureObject_t fontAtlas,
    const cudaTextureObject_t* __restrict__ textures,
    uint32_t textureCount,
    float4* __restrict__ output,
    uint32_t width,
    uint32_t height,
    float sdfThreshold,
    float sdfSmoothing,
    int regionStartX,
    int regionStartY
) {
    // Calculate tile index
    int tileX = blockIdx.x;
    int tileY = blockIdx.y;

    if (tileX >= (int)tileCountX || tileY >= (int)tileCountY) return;

    // Calculate pixel position within tile
    int localX = threadIdx.x;
    int localY = threadIdx.y;

    // Calculate global pixel position
    int px = regionStartX + tileX * TILE_SIZE + localX;
    int py = regionStartY + tileY * TILE_SIZE + localY;

    if (px >= (int)width || py >= (int)height) return;

    // Get tile data
    int tileIdx = tileY * tileCountX + tileX;
    const TileData& tile = tileData[tileIdx];

    float2 p = make_float2(static_cast<float>(px) + 0.5f, static_cast<float>(py) + 0.5f);
    uint32_t outputIdx = py * width + px;
    float4 result = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    // Process only quads that overlap this tile
    for (int q = 0; q < tile.quadCount; q++) {
        uint32_t quadIdx = tile.quadIndices[q];
        const UIQuad& quad = quads[quadIdx];

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

    output[outputIdx] = result;
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

    // Pre-allocate tile data for a reasonable screen size (will grow if needed)
    // For 1920x1080 with 16x16 tiles: 120x68 = 8160 tiles
    m_maxTiles = 16384;  // Enough for 4K
    err = cudaMalloc(&m_deviceTileData, m_maxTiles * sizeof(TileData));
    if (err != cudaSuccess) {
        std::cerr << "[UIRenderer] Failed to allocate tile data: "
                  << cudaGetErrorString(err) << "\n";
        cudaFree(m_deviceQuads);
        cudaFree(m_deviceTextures);
        m_deviceQuads = nullptr;
        m_deviceTextures = nullptr;
        return false;
    }
    m_hostTileData.reserve(m_maxTiles);

    std::cout << "[UIRenderer] Initialized with tile-based rendering, max " << maxQuads << " quads\n";
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
    if (m_deviceTileData) {
        cudaFree(m_deviceTileData);
        m_deviceTileData = nullptr;
    }
    m_hostTileData.clear();
    m_maxQuads = 0;
    m_textureCount = 0;
    m_maxTiles = 0;
}

void UIRenderer::setTextures(const cudaTextureObject_t* textures, uint32_t count) {
    if (!m_deviceTextures || !textures) return;
    m_textureCount = std::min(count, static_cast<uint32_t>(MAX_UI_TEXTURES));
    if (m_textureCount > 0) {
        cudaMemcpy(m_deviceTextures, textures, m_textureCount * sizeof(cudaTextureObject_t),
                   cudaMemcpyHostToDevice);
    }
}

void UIRenderer::buildTileData(const std::vector<UIQuad>& quads,
                                uint32_t width, uint32_t height,
                                int startX, int startY, int endX, int endY) {
    // Calculate tile counts for the render region
    int regionWidth = endX - startX;
    int regionHeight = endY - startY;
    m_tileCountX = (regionWidth + TILE_SIZE - 1) / TILE_SIZE;
    m_tileCountY = (regionHeight + TILE_SIZE - 1) / TILE_SIZE;

    uint32_t totalTiles = m_tileCountX * m_tileCountY;

    // Resize host tile data
    m_hostTileData.resize(totalTiles);

    // Fast clear using memset (only need to zero quadCount, but clearing all is still faster than loop)
    std::memset(m_hostTileData.data(), 0, totalTiles * sizeof(TileData));

    // For each quad, add it to overlapping tiles
    uint32_t quadCount = static_cast<uint32_t>(std::min(quads.size(), static_cast<size_t>(m_maxQuads)));

    for (uint32_t q = 0; q < quadCount; q++) {
        const UIQuad& quad = quads[q];

        // Compute quad bounding box using direct access (faster than min/max calls)
        float minXf = quad.vertices[0].position.x;
        float minYf = quad.vertices[0].position.y;
        float maxXf = minXf;
        float maxYf = minYf;
        
        for (int v = 1; v < 4; v++) {
            float x = quad.vertices[v].position.x;
            float y = quad.vertices[v].position.y;
            if (x < minXf) minXf = x;
            if (x > maxXf) maxXf = x;
            if (y < minYf) minYf = y;
            if (y > maxYf) maxYf = y;
        }

        // Convert to integer bounds
        int quadMinX = static_cast<int>(minXf);
        int quadMinY = static_cast<int>(minYf);
        int quadMaxX = static_cast<int>(maxXf) + 1;
        int quadMaxY = static_cast<int>(maxYf) + 1;

        // Skip quads entirely outside the render region
        if (quadMaxX <= startX || quadMinX >= endX || quadMaxY <= startY || quadMinY >= endY) {
            continue;
        }

        // Clamp to render region
        if (quadMinX < startX) quadMinX = startX;
        if (quadMinY < startY) quadMinY = startY;
        if (quadMaxX > endX) quadMaxX = endX;
        if (quadMaxY > endY) quadMaxY = endY;

        // Convert to tile coordinates (relative to region start)
        int tileMinX = (quadMinX - startX) / TILE_SIZE;
        int tileMinY = (quadMinY - startY) / TILE_SIZE;
        int tileMaxX = (quadMaxX - startX - 1) / TILE_SIZE;
        int tileMaxY = (quadMaxY - startY - 1) / TILE_SIZE;

        // Clamp tile coordinates
        if (tileMinX < 0) tileMinX = 0;
        if (tileMinY < 0) tileMinY = 0;
        if (tileMaxX >= static_cast<int>(m_tileCountX)) tileMaxX = m_tileCountX - 1;
        if (tileMaxY >= static_cast<int>(m_tileCountY)) tileMaxY = m_tileCountY - 1;

        // Add quad index to all overlapping tiles
        for (int ty = tileMinY; ty <= tileMaxY; ty++) {
            int rowOffset = ty * m_tileCountX;
            for (int tx = tileMinX; tx <= tileMaxX; tx++) {
                TileData& tile = m_hostTileData[rowOffset + tx];
                if (tile.quadCount < MAX_QUADS_PER_TILE) {
                    tile.quadIndices[tile.quadCount++] = static_cast<uint16_t>(q);
                }
            }
        }
    }
}

bool UIRenderer::renderIfChanged(const std::vector<UIQuad>& quads,
                                  uint64_t generation,
                                  cudaTextureObject_t fontAtlas,
                                  float4* outputBuffer,
                                  uint32_t width, uint32_t height,
                                  cudaStream_t stream) {
    // Skip rendering if geometry hasn't changed (UI buffer persists between frames)
    if (generation == m_lastGeneration && generation != 0) {
        return false;  // Reuse previous frame's UI buffer
    }
    
    m_lastGeneration = generation;
    
    // Call the normal render function
    render(quads, fontAtlas, outputBuffer, width, height, stream);
    return true;
}

void UIRenderer::render(const std::vector<UIQuad>& quads,
                        cudaTextureObject_t fontAtlas,
                        float4* outputBuffer,
                        uint32_t width, uint32_t height,
                        cudaStream_t stream) {
    // Clear the full buffer
    dim3 clearBlockSize(16, 16);
    dim3 clearGridSize((width + clearBlockSize.x - 1) / clearBlockSize.x,
                       (height + clearBlockSize.y - 1) / clearBlockSize.y);
    clearBufferKernel<<<clearGridSize, clearBlockSize, 0, stream>>>(outputBuffer, width, height);

    if (quads.empty() || !outputBuffer) {
        m_lastQuadCount = 0;
        return;
    }

    uint32_t quadCount = static_cast<uint32_t>(std::min(quads.size(), static_cast<size_t>(m_maxQuads)));

    // Compute overall bounding box in single pass with direct comparisons
    const UIQuad& firstQuad = quads[0];
    float minXf = firstQuad.vertices[0].position.x;
    float minYf = firstQuad.vertices[0].position.y;
    float maxXf = minXf;
    float maxYf = minYf;

    for (uint32_t q = 0; q < quadCount; q++) {
        const UIQuad& quad = quads[q];
        for (int v = 0; v < 4; v++) {
            float x = quad.vertices[v].position.x;
            float y = quad.vertices[v].position.y;
            if (x < minXf) minXf = x;
            if (x > maxXf) maxXf = x;
            if (y < minYf) minYf = y;
            if (y > maxYf) maxYf = y;
        }
    }

    int startX = static_cast<int>(minXf);
    int startY = static_cast<int>(minYf);
    int endX = static_cast<int>(maxXf) + 1;
    int endY = static_cast<int>(maxYf) + 1;

    // Clamp to screen
    if (startX < 0) startX = 0;
    if (startY < 0) startY = 0;
    if (endX > static_cast<int>(width)) endX = width;
    if (endY > static_cast<int>(height)) endY = height;

    if (startX >= endX || startY >= endY) {
        m_lastQuadCount = 0;
        return;
    }

    // Build tile data
    buildTileData(quads, width, height, startX, startY, endX, endY);

    uint32_t totalTiles = m_tileCountX * m_tileCountY;

    // Reallocate device tile data if needed
    if (totalTiles > m_maxTiles) {
        if (m_deviceTileData) cudaFree(m_deviceTileData);
        m_maxTiles = totalTiles + 1024;  // Add some headroom
        cudaMalloc(&m_deviceTileData, m_maxTiles * sizeof(TileData));
        std::cout << "[UIRenderer] Reallocated tile data for " << m_maxTiles << " tiles\n";
    }

    // Copy quads and tile data to device
    cudaMemcpyAsync(m_deviceQuads, quads.data(), quadCount * sizeof(UIQuad),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(m_deviceTileData, m_hostTileData.data(), totalTiles * sizeof(TileData),
                    cudaMemcpyHostToDevice, stream);

    // Launch tile-based kernel
    dim3 blockSize(TILE_SIZE, TILE_SIZE);
    dim3 gridSize(m_tileCountX, m_tileCountY);

    rasterizeTiledKernel<<<gridSize, blockSize, 0, stream>>>(
        m_deviceQuads,
        m_deviceTileData,
        m_tileCountX,
        m_tileCountY,
        fontAtlas,
        m_deviceTextures,
        m_textureCount,
        outputBuffer,
        width,
        height,
        m_sdfThreshold,
        m_sdfSmoothing,
        startX,
        startY
    );

    m_lastQuadCount = quadCount;
}

void UIRenderer::setSDFParams(float threshold, float smoothing) {
    m_sdfThreshold = threshold;
    m_sdfSmoothing = smoothing;
}

} // namespace ui
} // namespace spectra
