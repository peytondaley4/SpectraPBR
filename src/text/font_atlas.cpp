#include "font_atlas.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <limits>

// Include stb_truetype implementation
#define STB_TRUETYPE_IMPLEMENTATION
#include "stb_truetype.h"

namespace spectra {
namespace text {

FontAtlas::~FontAtlas() {
    release();
}

void FontAtlas::release() {
    if (m_atlasTexture) {
        cudaDestroyTextureObject(m_atlasTexture);
        m_atlasTexture = 0;
    }
    if (m_atlasArray) {
        cudaFreeArray(m_atlasArray);
        m_atlasArray = nullptr;
    }
    m_glyphMetrics.clear();
    m_atlasData.clear();
    m_fontSize = 0.0f;
    m_lineHeight = 0.0f;
    m_atlasWidth = 0;
    m_atlasHeight = 0;
}

bool FontAtlas::load(const std::string& ttfPath, float fontSize,
                     uint32_t atlasSize, uint32_t sdfPadding) {
    // Release any existing data
    release();

    // Load TTF file
    std::ifstream file(ttfPath, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "[FontAtlas] Failed to open font file: " << ttfPath << "\n";
        return false;
    }

    std::streamsize fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<uint8_t> fontBuffer(static_cast<size_t>(fileSize));
    if (!file.read(reinterpret_cast<char*>(fontBuffer.data()), fileSize)) {
        std::cerr << "[FontAtlas] Failed to read font file: " << ttfPath << "\n";
        return false;
    }
    file.close();

    // Initialize stb_truetype
    stbtt_fontinfo fontInfo;
    if (!stbtt_InitFont(&fontInfo, fontBuffer.data(), 0)) {
        std::cerr << "[FontAtlas] Failed to initialize font: " << ttfPath << "\n";
        return false;
    }

    // Calculate scale for desired font size
    float scale = stbtt_ScaleForPixelHeight(&fontInfo, fontSize);

    // Get font metrics
    int ascent, descent, lineGap;
    stbtt_GetFontVMetrics(&fontInfo, &ascent, &descent, &lineGap);
    m_lineHeight = (ascent - descent + lineGap) * scale;
    m_fontSize = fontSize;
    m_atlasWidth = atlasSize;
    m_atlasHeight = atlasSize;

    // Allocate atlas
    m_atlasData.resize(atlasSize * atlasSize, 0);

    // Pack glyphs into atlas using simple row-based packing
    // ASCII 32 (space) to 126 (~)
    const int firstChar = 32;
    const int lastChar = 126;
    const int numChars = lastChar - firstChar + 1;

    // Calculate SDF parameters
    const int sdfSpread = static_cast<int>(sdfPadding);
    const float spreadScale = static_cast<float>(sdfSpread);

    // Current position in atlas
    int cursorX = sdfPadding;
    int cursorY = sdfPadding;
    int rowHeight = 0;

    // Scale up for higher resolution rendering, then generate SDF at target size
    const float renderScale = scale * 2.0f; // Render at 2x size for better SDF quality

    for (int c = firstChar; c <= lastChar; c++) {
        // Get glyph metrics
        int advanceWidth, leftSideBearing;
        stbtt_GetCodepointHMetrics(&fontInfo, c, &advanceWidth, &leftSideBearing);

        int x0, y0, x1, y1;
        stbtt_GetCodepointBitmapBox(&fontInfo, c, scale, scale, &x0, &y0, &x1, &y1);

        int glyphWidth = x1 - x0;
        int glyphHeight = y1 - y0;

        // Size including SDF padding
        int paddedWidth = glyphWidth + sdfPadding * 2;
        int paddedHeight = glyphHeight + sdfPadding * 2;

        // Check if we need to move to next row
        if (cursorX + paddedWidth > static_cast<int>(atlasSize - sdfPadding)) {
            cursorX = sdfPadding;
            cursorY += rowHeight + sdfPadding;
            rowHeight = 0;
        }

        // Check if we've run out of space
        if (cursorY + paddedHeight > static_cast<int>(atlasSize - sdfPadding)) {
            std::cerr << "[FontAtlas] Atlas too small for all glyphs\n";
            break;
        }

        // Render glyph bitmap at higher resolution
        int hiresBitmapWidth, hiresBitmapHeight, hiresXoff, hiresYoff;
        uint8_t* hiresBitmap = stbtt_GetCodepointBitmap(
            &fontInfo, renderScale, renderScale, c,
            &hiresBitmapWidth, &hiresBitmapHeight, &hiresXoff, &hiresYoff);

        if (hiresBitmap && hiresBitmapWidth > 0 && hiresBitmapHeight > 0) {
            // Generate SDF from high-res bitmap
            std::vector<uint8_t> sdfBuffer(paddedWidth * paddedHeight, 0);

            // Simple distance field generation
            // For each pixel in output, find distance to nearest edge in input
            for (int py = 0; py < paddedHeight; py++) {
                for (int px = 0; px < paddedWidth; px++) {
                    // Map output pixel to input bitmap space
                    float srcX = (px - static_cast<float>(sdfPadding)) * 2.0f;
                    float srcY = (py - static_cast<float>(sdfPadding)) * 2.0f;

                    // Find minimum distance to edge
                    float minDist = spreadScale;
                    bool inside = false;

                    // Sample the high-res bitmap
                    int sampleX = static_cast<int>(srcX);
                    int sampleY = static_cast<int>(srcY);
                    if (sampleX >= 0 && sampleX < hiresBitmapWidth &&
                        sampleY >= 0 && sampleY < hiresBitmapHeight) {
                        inside = hiresBitmap[sampleY * hiresBitmapWidth + sampleX] > 127;
                    }

                    // Search for edge in a radius
                    const int searchRadius = static_cast<int>(spreadScale * 2.0f);
                    for (int dy = -searchRadius; dy <= searchRadius; dy++) {
                        for (int dx = -searchRadius; dx <= searchRadius; dx++) {
                            int checkX = sampleX + dx;
                            int checkY = sampleY + dy;

                            if (checkX >= 0 && checkX < hiresBitmapWidth &&
                                checkY >= 0 && checkY < hiresBitmapHeight) {
                                bool checkInside = hiresBitmap[checkY * hiresBitmapWidth + checkX] > 127;
                                if (checkInside != inside) {
                                    float dist = std::sqrt(static_cast<float>(dx * dx + dy * dy)) / 2.0f;
                                    minDist = std::min(minDist, dist);
                                }
                            }
                        }
                    }

                    // Convert to 0-255 range (0.5 = edge)
                    float normalizedDist = minDist / spreadScale;
                    if (inside) {
                        normalizedDist = 0.5f + normalizedDist * 0.5f;
                    } else {
                        normalizedDist = 0.5f - normalizedDist * 0.5f;
                    }
                    normalizedDist = std::clamp(normalizedDist, 0.0f, 1.0f);

                    sdfBuffer[py * paddedWidth + px] = static_cast<uint8_t>(normalizedDist * 255.0f);
                }
            }

            // Copy SDF to atlas
            for (int py = 0; py < paddedHeight; py++) {
                for (int px = 0; px < paddedWidth; px++) {
                    int atlasIdx = (cursorY + py) * atlasSize + (cursorX + px);
                    m_atlasData[atlasIdx] = sdfBuffer[py * paddedWidth + px];
                }
            }

            stbtt_FreeBitmap(hiresBitmap, nullptr);
        } else {
            // Space or empty glyph - fill with "outside" value
            for (int py = 0; py < paddedHeight; py++) {
                for (int px = 0; px < paddedWidth; px++) {
                    int atlasIdx = (cursorY + py) * atlasSize + (cursorX + px);
                    m_atlasData[atlasIdx] = 0;
                }
            }
        }

        // Store glyph metrics
        ui::GlyphMetrics metrics;
        metrics.uvMin = make_float2(
            static_cast<float>(cursorX) / atlasSize,
            static_cast<float>(cursorY) / atlasSize
        );
        metrics.uvMax = make_float2(
            static_cast<float>(cursorX + paddedWidth) / atlasSize,
            static_cast<float>(cursorY + paddedHeight) / atlasSize
        );
        metrics.size = make_float2(static_cast<float>(paddedWidth),
                                    static_cast<float>(paddedHeight));
        metrics.bearing = make_float2(static_cast<float>(x0 - static_cast<int>(sdfPadding)),
                                       static_cast<float>(y0 - static_cast<int>(sdfPadding)));
        metrics.advance = advanceWidth * scale;
        metrics._pad = 0.0f;

        m_glyphMetrics[static_cast<char>(c)] = metrics;

        // Update cursor
        cursorX += paddedWidth + sdfPadding;
        rowHeight = std::max(rowHeight, paddedHeight);
    }

    // Upload to GPU
    if (!uploadToGPU()) {
        std::cerr << "[FontAtlas] Failed to upload atlas to GPU\n";
        release();
        return false;
    }

    std::cout << "[FontAtlas] Loaded font: " << ttfPath << "\n";
    std::cout << "[FontAtlas] Atlas size: " << atlasSize << "x" << atlasSize << "\n";
    std::cout << "[FontAtlas] Font size: " << fontSize << ", Line height: " << m_lineHeight << "\n";
    std::cout << "[FontAtlas] Glyphs loaded: " << m_glyphMetrics.size() << "\n";

    return true;
}

bool FontAtlas::uploadToGPU() {
    // Create CUDA array
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uint8_t>();

    cudaError_t err = cudaMallocArray(&m_atlasArray, &channelDesc,
                                       m_atlasWidth, m_atlasHeight);
    if (err != cudaSuccess) {
        std::cerr << "[FontAtlas] cudaMallocArray failed: "
                  << cudaGetErrorString(err) << "\n";
        return false;
    }

    // Copy data to array
    err = cudaMemcpy2DToArray(m_atlasArray, 0, 0,
                               m_atlasData.data(),
                               m_atlasWidth * sizeof(uint8_t),
                               m_atlasWidth * sizeof(uint8_t),
                               m_atlasHeight,
                               cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "[FontAtlas] cudaMemcpy2DToArray failed: "
                  << cudaGetErrorString(err) << "\n";
        return false;
    }

    // Create texture object
    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = m_atlasArray;

    cudaTextureDesc texDesc = {};
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModeLinear;  // Linear filtering for smooth SDF
    texDesc.readMode = cudaReadModeNormalizedFloat;  // Read as [0,1] float
    texDesc.normalizedCoords = 1;  // Use normalized [0,1] coordinates

    err = cudaCreateTextureObject(&m_atlasTexture, &resDesc, &texDesc, nullptr);
    if (err != cudaSuccess) {
        std::cerr << "[FontAtlas] cudaCreateTextureObject failed: "
                  << cudaGetErrorString(err) << "\n";
        return false;
    }

    return true;
}

const ui::GlyphMetrics* FontAtlas::getGlyph(char c) const {
    auto it = m_glyphMetrics.find(c);
    if (it != m_glyphMetrics.end()) {
        return &it->second;
    }
    // Return space glyph for unknown characters
    auto spaceIt = m_glyphMetrics.find(' ');
    if (spaceIt != m_glyphMetrics.end()) {
        return &spaceIt->second;
    }
    return nullptr;
}

void FontAtlas::generateSDF(const uint8_t* bitmap, int bitmapWidth, int bitmapHeight,
                             uint8_t* sdf, int sdfWidth, int sdfHeight,
                             int padding, float spread) {
    // Simple brute-force SDF generation
    // For production, consider using EDT (Euclidean Distance Transform)

    for (int y = 0; y < sdfHeight; y++) {
        for (int x = 0; x < sdfWidth; x++) {
            // Map SDF pixel to bitmap pixel
            int bx = x - padding;
            int by = y - padding;

            // Determine if we're inside or outside the glyph
            bool inside = false;
            if (bx >= 0 && bx < bitmapWidth && by >= 0 && by < bitmapHeight) {
                inside = bitmap[by * bitmapWidth + bx] > 127;
            }

            // Find minimum distance to edge
            float minDist = spread;
            int searchRadius = static_cast<int>(spread) + 1;

            for (int sy = -searchRadius; sy <= searchRadius; sy++) {
                for (int sx = -searchRadius; sx <= searchRadius; sx++) {
                    int checkBx = bx + sx;
                    int checkBy = by + sy;

                    bool checkInside = false;
                    if (checkBx >= 0 && checkBx < bitmapWidth &&
                        checkBy >= 0 && checkBy < bitmapHeight) {
                        checkInside = bitmap[checkBy * bitmapWidth + checkBx] > 127;
                    }

                    if (checkInside != inside) {
                        float dist = std::sqrt(static_cast<float>(sx * sx + sy * sy));
                        minDist = std::min(minDist, dist);
                    }
                }
            }

            // Normalize distance and convert to SDF value
            float normalizedDist = minDist / spread;
            float sdfValue;
            if (inside) {
                sdfValue = 0.5f + normalizedDist * 0.5f;
            } else {
                sdfValue = 0.5f - normalizedDist * 0.5f;
            }

            sdfValue = std::clamp(sdfValue, 0.0f, 1.0f);
            sdf[y * sdfWidth + x] = static_cast<uint8_t>(sdfValue * 255.0f);
        }
    }
}

} // namespace text
} // namespace spectra
