#pragma once

#include <cuda_runtime.h>
#include <string>
#include <unordered_map>
#include <vector>
#include "ui/ui_types.h"

namespace spectra {
namespace text {

//------------------------------------------------------------------------------
// Font Atlas - Generates and manages an SDF font atlas texture
// Uses stb_truetype for runtime atlas generation
//------------------------------------------------------------------------------
class FontAtlas {
public:
    FontAtlas() = default;
    ~FontAtlas();

    // Non-copyable
    FontAtlas(const FontAtlas&) = delete;
    FontAtlas& operator=(const FontAtlas&) = delete;

    // Load a TTF font and generate SDF atlas
    // fontSize: Base font size in pixels (glyphs are rendered at this size)
    // atlasSize: Width/height of the atlas texture (power of 2 recommended)
    // sdfPadding: Extra padding around each glyph for SDF spread
    bool load(const std::string& ttfPath, float fontSize = 32.0f,
              uint32_t atlasSize = 512, uint32_t sdfPadding = 8);

    // Release all resources
    void release();

    // Check if atlas is loaded
    bool isLoaded() const { return m_atlasTexture != 0; }

    // Get CUDA texture object for sampling
    cudaTextureObject_t getTexture() const { return m_atlasTexture; }

    // Get glyph metrics for a character
    // Returns nullptr if character not found
    const ui::GlyphMetrics* getGlyph(char c) const;

    // Get line height for this font
    float getLineHeight() const { return m_lineHeight; }

    // Get base font size
    float getFontSize() const { return m_fontSize; }

    // Get atlas dimensions
    uint32_t getAtlasWidth() const { return m_atlasWidth; }
    uint32_t getAtlasHeight() const { return m_atlasHeight; }

    // Get atlas data (for debugging)
    const std::vector<uint8_t>& getAtlasData() const { return m_atlasData; }

private:
    // Generate SDF from high-resolution glyph bitmap
    void generateSDF(const uint8_t* bitmap, int bitmapWidth, int bitmapHeight,
                     uint8_t* sdf, int sdfWidth, int sdfHeight,
                     int padding, float spread);

    // Upload atlas to CUDA texture
    bool uploadToGPU();

    cudaTextureObject_t m_atlasTexture = 0;
    cudaArray_t m_atlasArray = nullptr;

    std::unordered_map<char, ui::GlyphMetrics> m_glyphMetrics;
    std::vector<uint8_t> m_atlasData;

    float m_fontSize = 0.0f;
    float m_lineHeight = 0.0f;
    uint32_t m_atlasWidth = 0;
    uint32_t m_atlasHeight = 0;
};

} // namespace text
} // namespace spectra
