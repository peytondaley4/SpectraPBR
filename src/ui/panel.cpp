#include "panel.h"
#include "text/text_layout.h"

namespace spectra {
namespace ui {

Panel::Panel() {
    setSize(300.0f, 200.0f);
}

Rect Panel::getContentBounds() const {
    Rect bounds = getAbsoluteBounds();
    if (m_showHeader) {
        bounds.y += m_headerHeight;
        bounds.height -= m_headerHeight;
    }
    if (m_showBorder) {
        bounds.x += m_borderWidth;
        bounds.y += m_borderWidth;
        bounds.width -= m_borderWidth * 2;
        bounds.height -= m_borderWidth * 2;
    }
    return bounds;
}

void Panel::generateGeometry(std::vector<UIQuad>& outQuads, text::TextLayout* textLayout) {
    const Theme* theme = getTheme();
    Rect bounds = getAbsoluteBounds();
    float depth = getEffectiveDepth();

    // Background
    outQuads.push_back(makeSolidQuad(bounds, theme->panelBackground, depth));

    // Border
    if (m_showBorder && m_borderWidth > 0) {
        float bw = m_borderWidth;
        // Top border
        outQuads.push_back(makeSolidQuad(
            { bounds.x, bounds.y, bounds.width, bw },
            theme->panelBorder, depth + 0.001f));
        // Bottom border
        outQuads.push_back(makeSolidQuad(
            { bounds.x, bounds.bottom() - bw, bounds.width, bw },
            theme->panelBorder, depth + 0.001f));
        // Left border
        outQuads.push_back(makeSolidQuad(
            { bounds.x, bounds.y, bw, bounds.height },
            theme->panelBorder, depth + 0.001f));
        // Right border
        outQuads.push_back(makeSolidQuad(
            { bounds.right() - bw, bounds.y, bw, bounds.height },
            theme->panelBorder, depth + 0.001f));
    }

    // Header
    if (m_showHeader) {
        Rect headerBounds = { bounds.x, bounds.y, bounds.width, m_headerHeight };
        outQuads.push_back(makeSolidQuad(headerBounds, theme->panelBackgroundAlt, depth + 0.002f));

        // Separator line under header
        outQuads.push_back(makeSolidQuad(
            { bounds.x, bounds.y + m_headerHeight - 1.0f, bounds.width, 1.0f },
            theme->separator, depth + 0.003f));

        // Title text
        if (!m_title.empty() && textLayout) {
            float textScale = 0.7f;
            float textHeight = textLayout->getLineHeight(textScale);
            // Center text vertically in header - position is top-left of text area
            float2 titlePos = make_float2(
                bounds.x + 8.0f,
                bounds.y + (m_headerHeight - textHeight) * 0.5f
            );
            textLayout->layout(m_title, titlePos, textScale, theme->textPrimary,
                              TextAlign::Left, depth + 0.004f, outQuads);
        }

        // Close button
        if (m_closeable) {
            float btnSize = m_headerHeight - 8.0f;
            Rect closeRect = {
                bounds.right() - btnSize - 4.0f,
                bounds.y + 4.0f,
                btnSize, btnSize
            };

            float4 closeColor = m_closeHovered ? theme->buttonHover : theme->buttonNormal;
            outQuads.push_back(makeSolidQuad(closeRect, closeColor, depth + 0.004f));

            // X mark
            if (textLayout) {
                float xScale = 0.5f;
                float xTextHeight = textLayout->getLineHeight(xScale);
                float2 xPos = make_float2(
                    closeRect.x + closeRect.width * 0.5f,
                    closeRect.y + (closeRect.height - xTextHeight) * 0.5f
                );
                textLayout->layout("X", xPos, xScale, theme->textPrimary,
                                  TextAlign::Center, depth + 0.005f, outQuads);
            }
        }
    }
}

bool Panel::onMouseDown(float2 pos, int button) {
    if (!m_visible || !m_enabled) return false;

    // Check children first
    for (auto it = m_children.rbegin(); it != m_children.rend(); ++it) {
        if ((*it)->onMouseDown(pos, button)) {
            return true;
        }
    }

    Rect bounds = getAbsoluteBounds();

    // Check close button
    if (m_showHeader && m_closeable) {
        float btnSize = m_headerHeight - 8.0f;
        Rect closeRect = {
            bounds.right() - btnSize - 4.0f,
            bounds.y + 4.0f,
            btnSize, btnSize
        };
        if (closeRect.contains(pos)) {
            return true; // Will handle on mouse up
        }
    }

    // Check header drag
    if (m_showHeader && m_draggable && button == 0) {
        Rect headerBounds = { bounds.x, bounds.y, bounds.width, m_headerHeight };
        if (headerBounds.contains(pos)) {
            m_dragging = true;
            m_dragOffset = make_float2(pos.x - bounds.x, pos.y - bounds.y);
            return true;
        }
    }

    if (bounds.contains(pos)) {
        return true;
    }

    return false;
}

bool Panel::onMouseUp(float2 pos, int button) {
    if (!m_visible || !m_enabled) return false;

    // Check children first
    for (auto it = m_children.rbegin(); it != m_children.rend(); ++it) {
        if ((*it)->onMouseUp(pos, button)) {
            return true;
        }
    }

    bool consumed = false;

    // Check close button click
    if (m_showHeader && m_closeable) {
        Rect bounds = getAbsoluteBounds();
        float btnSize = m_headerHeight - 8.0f;
        Rect closeRect = {
            bounds.right() - btnSize - 4.0f,
            bounds.y + 4.0f,
            btnSize, btnSize
        };
        if (closeRect.contains(pos)) {
            if (m_onClose) {
                m_onClose();
            }
            setVisible(false);
            consumed = true;
        }
    }

    if (m_dragging) {
        m_dragging = false;
        consumed = true;
    }

    return consumed;
}

bool Panel::onMouseMove(float2 pos) {
    if (!m_visible || !m_enabled) return false;

    // Handle dragging
    if (m_dragging) {
        setPosition(pos.x - m_dragOffset.x, pos.y - m_dragOffset.y);
        return true;
    }

    // Check children first
    for (auto it = m_children.rbegin(); it != m_children.rend(); ++it) {
        if ((*it)->onMouseMove(pos)) {
            return true;
        }
    }

    // Update close button hover state
    if (m_showHeader && m_closeable) {
        Rect bounds = getAbsoluteBounds();
        float btnSize = m_headerHeight - 8.0f;
        Rect closeRect = {
            bounds.right() - btnSize - 4.0f,
            bounds.y + 4.0f,
            btnSize, btnSize
        };
        bool wasHovered = m_closeHovered;
        m_closeHovered = closeRect.contains(pos);
        if (m_closeHovered != wasHovered) {
            markDirty();
        }
    }

    // Update hover state for this widget
    bool wasHovered = m_hovered;
    Rect bounds = getAbsoluteBounds();
    m_hovered = bounds.contains(pos);

    if (m_hovered != wasHovered) {
        markDirty();
        onHoverChanged();
    }

    return m_hovered;
}

} // namespace ui
} // namespace spectra
