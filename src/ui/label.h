#pragma once

#include "widget.h"

namespace spectra {
namespace ui {

//------------------------------------------------------------------------------
// Label - Text display widget
//------------------------------------------------------------------------------
class Label : public Widget {
public:
    Label();
    explicit Label(const std::string& text);
    ~Label() override = default;

    // Text content
    void setText(const std::string& text) { m_text = text; markDirty(); }
    const std::string& getText() const { return m_text; }

    // Text scale
    void setTextScale(float scale) { m_textScale = scale; markDirty(); }
    float getTextScale() const { return m_textScale; }

    // Text alignment
    void setAlign(TextAlign align) { m_align = align; markDirty(); }
    TextAlign getAlign() const { return m_align; }

    // Color override (if set, overrides theme)
    void setColor(float4 color) { m_color = color; m_useCustomColor = true; markDirty(); }
    void clearColor() { m_useCustomColor = false; markDirty(); }

    // Secondary text style (uses textSecondary color)
    void setSecondary(bool secondary) { m_secondary = secondary; markDirty(); }
    bool isSecondary() const { return m_secondary; }

protected:
    void generateGeometry(std::vector<UIQuad>& outQuads, text::TextLayout* textLayout) override;

private:
    std::string m_text;
    float m_textScale = 0.6f;
    TextAlign m_align = TextAlign::Left;
    float4 m_color = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
    bool m_useCustomColor = false;
    bool m_secondary = false;
};

} // namespace ui
} // namespace spectra
