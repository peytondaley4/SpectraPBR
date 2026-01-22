#include "button.h"
#include "text/text_layout.h"

namespace spectra {
namespace ui {

Button::Button() {
    setSize(100.0f, 28.0f);
}

Button::Button(const std::string& label)
    : m_label(label) {
    setSize(100.0f, 28.0f);
}

void Button::generateGeometry(std::vector<UIQuad>& outQuads, text::TextLayout* textLayout) {
    const Theme* theme = getTheme();
    Rect bounds = getAbsoluteBounds();
    float depth = getEffectiveDepth();

    // Determine background color based on state
    float4 bgColor;
    if (!m_enabled) {
        bgColor = theme->buttonDisabled;
    } else if (m_active || m_toggled) {
        bgColor = theme->buttonActive;
    } else if (m_hovered) {
        bgColor = theme->buttonHover;
    } else {
        bgColor = theme->buttonNormal;
    }

    // Background
    outQuads.push_back(makeSolidQuad(bounds, bgColor, depth));

    // Label
    if (!m_label.empty() && textLayout) {
        float4 textColor = m_enabled ? theme->textPrimary : theme->textDisabled;
        float2 labelPos = make_float2(
            bounds.x + bounds.width * 0.5f,
            bounds.y + bounds.height * 0.5f - textLayout->getLineHeight(m_textScale) * 0.5f
        );
        textLayout->layout(m_label, labelPos, m_textScale, textColor,
                          TextAlign::Center, depth + 0.001f, outQuads);
    }
}

bool Button::onMouseDown(float2 pos, int button) {
    if (!m_visible || !m_enabled) return false;
    if (button != 0) return false;

    if (containsPoint(pos)) {
        m_active = true;
        markDirty();
        return true;
    }
    return false;
}

bool Button::onMouseUp(float2 pos, int button) {
    if (!m_visible || !m_enabled) return false;
    if (button != 0) return false;

    bool wasActive = m_active;
    m_active = false;

    if (wasActive && containsPoint(pos)) {
        // Toggle mode
        if (m_toggleMode) {
            m_toggled = !m_toggled;
        }

        // Fire callback
        if (m_onClick) {
            m_onClick();
        }

        markDirty();
        return true;
    }

    if (wasActive) {
        markDirty();
    }

    return false;
}

} // namespace ui
} // namespace spectra
