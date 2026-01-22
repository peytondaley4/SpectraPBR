#pragma once

#include <cuda_runtime.h>
#include <string>
#include <vector>
#include "ui/ui_types.h"
#include "font_atlas.h"

namespace spectra {
namespace text {

//------------------------------------------------------------------------------
// Text Layout - Converts strings into positioned UI quads
//------------------------------------------------------------------------------
class TextLayout {
public:
    TextLayout() = default;

    // Set the font atlas to use for layout
    void setFontAtlas(FontAtlas* atlas) { m_fontAtlas = atlas; }

    // Layout a string of text into UI quads
    // text: The string to render
    // pos: Top-left position in screen pixels
    // scale: Scale factor (1.0 = font's natural size)
    // color: Text color (RGBA)
    // align: Text alignment
    // depth: Z-order for rendering
    // outQuads: Output vector to append quads to
    void layout(const std::string& text, float2 pos, float scale,
                float4 color, ui::TextAlign align, float depth,
                std::vector<ui::UIQuad>& outQuads);

    // Measure the dimensions of a text string without generating quads
    // Returns (width, height) in pixels
    float2 measure(const std::string& text, float scale);

    // Measure the width of a single line of text
    float measureLineWidth(const std::string& text, float scale);

    // Get the line height for the current font
    float getLineHeight(float scale) const;

private:
    FontAtlas* m_fontAtlas = nullptr;
};

} // namespace text
} // namespace spectra
