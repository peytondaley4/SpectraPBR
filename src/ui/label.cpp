#include "label.h"
#include "text/text_layout.h"

namespace spectra {
namespace ui {

Label::Label() {
    setSize(100.0f, 20.0f);
}

Label::Label(const std::string& text)
    : m_text(text) {
    setSize(100.0f, 20.0f);
}

void Label::generateGeometry(std::vector<UIQuad>& outQuads, text::TextLayout* textLayout) {
    if (m_text.empty() || !textLayout) return;

    const Theme* theme = getTheme();
    Rect bounds = getAbsoluteBounds();
    float depth = getEffectiveDepth();

    // Determine text color
    float4 textColor;
    if (!m_enabled) {
        textColor = theme->textDisabled;
    } else if (m_useCustomColor) {
        textColor = m_color;
    } else if (m_secondary) {
        textColor = theme->textSecondary;
    } else {
        textColor = theme->textPrimary;
    }

    // Calculate position based on alignment
    float2 textPos;
    switch (m_align) {
        case TextAlign::Center:
            textPos = make_float2(
                bounds.x + bounds.width * 0.5f,
                bounds.y + bounds.height * 0.5f - textLayout->getLineHeight(m_textScale) * 0.5f
            );
            break;
        case TextAlign::Right:
            textPos = make_float2(
                bounds.x + bounds.width,
                bounds.y + bounds.height * 0.5f - textLayout->getLineHeight(m_textScale) * 0.5f
            );
            break;
        case TextAlign::Left:
        default:
            textPos = make_float2(
                bounds.x,
                bounds.y + bounds.height * 0.5f - textLayout->getLineHeight(m_textScale) * 0.5f
            );
            break;
    }

    textLayout->layout(m_text, textPos, m_textScale, textColor, m_align, depth, outQuads);
}

} // namespace ui
} // namespace spectra
