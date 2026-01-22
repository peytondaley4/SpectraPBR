#include "text_layout.h"
#include <cmath>

namespace spectra {
namespace text {

void TextLayout::layout(const std::string& text, float2 pos, float scale,
                        float4 color, ui::TextAlign align, float depth,
                        std::vector<ui::UIQuad>& outQuads) {
    if (!m_fontAtlas || text.empty()) {
        return;
    }

    // Calculate text width for alignment
    float textWidth = measureLineWidth(text, scale);
    float lineHeight = getLineHeight(scale);

    // Adjust starting position based on alignment
    float2 cursor = pos;
    switch (align) {
        case ui::TextAlign::Center:
            cursor.x -= textWidth * 0.5f;
            break;
        case ui::TextAlign::Right:
            cursor.x -= textWidth;
            break;
        case ui::TextAlign::Left:
        default:
            break;
    }

    // Process each character
    for (char c : text) {
        // Handle newlines
        if (c == '\n') {
            cursor.x = pos.x;
            cursor.y += lineHeight;

            // Recalculate alignment for next line if needed
            // (For simplicity, we use the same alignment offset)
            continue;
        }

        // Get glyph metrics
        const ui::GlyphMetrics* glyph = m_fontAtlas->getGlyph(c);
        if (!glyph) {
            continue;
        }

        // Calculate glyph position
        float2 glyphPos = make_float2(
            cursor.x + glyph->bearing.x * scale,
            cursor.y + glyph->bearing.y * scale
        );

        float2 glyphSize = make_float2(
            glyph->size.x * scale,
            glyph->size.y * scale
        );

        // Skip invisible characters (like spaces with no visible glyph)
        if (glyphSize.x > 0 && glyphSize.y > 0) {
            // Create text quad
            ui::UIQuad quad = ui::makeTextQuad(
                glyphPos, glyphSize,
                glyph->uvMin, glyph->uvMax,
                color, depth
            );
            outQuads.push_back(quad);
        }

        // Advance cursor
        cursor.x += glyph->advance * scale;
    }
}

float2 TextLayout::measure(const std::string& text, float scale) {
    if (!m_fontAtlas || text.empty()) {
        return make_float2(0.0f, 0.0f);
    }

    float maxWidth = 0.0f;
    float currentLineWidth = 0.0f;
    int lineCount = 1;

    for (char c : text) {
        if (c == '\n') {
            maxWidth = std::max(maxWidth, currentLineWidth);
            currentLineWidth = 0.0f;
            lineCount++;
            continue;
        }

        const ui::GlyphMetrics* glyph = m_fontAtlas->getGlyph(c);
        if (glyph) {
            currentLineWidth += glyph->advance * scale;
        }
    }

    maxWidth = std::max(maxWidth, currentLineWidth);
    float totalHeight = lineCount * getLineHeight(scale);

    return make_float2(maxWidth, totalHeight);
}

float TextLayout::measureLineWidth(const std::string& text, float scale) {
    if (!m_fontAtlas || text.empty()) {
        return 0.0f;
    }

    float width = 0.0f;
    for (char c : text) {
        if (c == '\n') {
            break; // Only measure first line
        }
        const ui::GlyphMetrics* glyph = m_fontAtlas->getGlyph(c);
        if (glyph) {
            width += glyph->advance * scale;
        }
    }
    return width;
}

float TextLayout::getLineHeight(float scale) const {
    if (!m_fontAtlas) {
        return 0.0f;
    }
    return m_fontAtlas->getLineHeight() * scale;
}

} // namespace text
} // namespace spectra
