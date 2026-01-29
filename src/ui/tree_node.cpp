#include "tree_node.h"
#include "text/text_layout.h"

namespace spectra {
namespace ui {

TreeNode::TreeNode() {
    setSize(200.0f, ROW_HEIGHT);
}

TreeNode::TreeNode(const std::string& label)
    : m_label(label) {
    setSize(200.0f, ROW_HEIGHT);
}

void TreeNode::setExpanded(bool expanded) {
    if (m_expanded != expanded) {
        m_expanded = expanded;
        markDirty();
        if (m_onExpand) {
            m_onExpand(this, expanded);
        }
    }
}

void TreeNode::setSelected(bool selected) {
    if (m_selected != selected) {
        m_selected = selected;
        markDirty();
        if (selected && m_onSelect) {
            m_onSelect(this);
        }
    }
}

Rect TreeNode::getExpanderBounds() const {
    Rect bounds = getAbsoluteBounds();
    float indent = m_indentLevel * INDENT_WIDTH;
    return {
        bounds.x + indent + 2.0f,
        bounds.y + (ROW_HEIGHT - EXPANDER_SIZE) * 0.5f,
        EXPANDER_SIZE,
        EXPANDER_SIZE
    };
}

void TreeNode::generateGeometry(std::vector<UIQuad>& outQuads, text::TextLayout* textLayout) {
    const Theme* theme = getTheme();
    Rect bounds = getAbsoluteBounds();
    float depth = getEffectiveDepth();
    float indent = m_indentLevel * INDENT_WIDTH;

    // Selection/hover background
    float4 bgColor;
    if (m_selected) {
        bgColor = theme->treeSelected;
    } else if (m_hovered) {
        bgColor = theme->treeHover;
    } else {
        bgColor = make_float4(0.0f, 0.0f, 0.0f, 0.0f); // Transparent
    }

    if (bgColor.w > 0.0f) {
        outQuads.push_back(makeSolidQuad(bounds, bgColor, depth));
    }

    // Expander triangle (if has children)
    if (m_hasChildren) {
        Rect expanderBounds = getExpanderBounds();
        float4 expanderColor = m_expanderHovered ? theme->highlight : theme->treeExpander;

        // Draw a simple triangle indicator
        // For expanded: pointing down (v)
        // For collapsed: pointing right (>)
        // Using a simple quad with different colors for now
        // In a full implementation, you'd draw actual triangles

        float cx = expanderBounds.x + expanderBounds.width * 0.5f;
        float cy = expanderBounds.y + expanderBounds.height * 0.5f;
        float triSize = EXPANDER_SIZE * 0.3f;

        if (m_expanded) {
            // Down arrow (simple quad representation)
            Rect tri = { cx - triSize, cy - triSize * 0.5f, triSize * 2.0f, triSize };
            outQuads.push_back(makeSolidQuad(tri, expanderColor, depth + 0.001f));
        } else {
            // Right arrow (simple quad representation)
            Rect tri = { cx - triSize * 0.5f, cy - triSize, triSize, triSize * 2.0f };
            outQuads.push_back(makeSolidQuad(tri, expanderColor, depth + 0.001f));
        }
    }

    // Label text
    if (!m_label.empty() && textLayout) {
        float4 textColor = m_enabled ? theme->textPrimary : theme->textDisabled;
        float textX = bounds.x + indent + INDENT_WIDTH + 4.0f;
        float textY = bounds.y + ROW_HEIGHT * 0.5f - textLayout->getLineHeight(0.55f) * 0.5f;

        textLayout->layout(m_label, make_float2(textX, textY), 0.55f, textColor,
                          TextAlign::Left, depth + 0.002f, outQuads);
    }
}

bool TreeNode::onMouseDown(float2 pos, int button) {
    if (!m_visible || !m_enabled) return false;
    if (button != 0) return false;

    // Check children first
    for (auto it = m_children.rbegin(); it != m_children.rend(); ++it) {
        if ((*it)->onMouseDown(pos, button)) {
            return true;
        }
    }

    if (!containsPoint(pos)) return false;

    // Check if clicked on expander
    if (m_hasChildren) {
        Rect expanderBounds = getExpanderBounds();
        if (expanderBounds.contains(pos)) {
            toggleExpanded();
            return true;
        }
    }

    // Check for double-click
    auto now = std::chrono::steady_clock::now();
    if (m_lastClickTime.has_value()) {
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            now - m_lastClickTime.value());
        if (elapsed.count() < DOUBLE_CLICK_MS) {
            // Double-click detected
            if (m_onDoubleClick) {
                m_onDoubleClick(this);
            }
            m_lastClickTime.reset();
            return true;
        }
    }
    m_lastClickTime = now;

    // Select this node
    setSelected(true);
    return true;
}

bool TreeNode::onMouseUp(float2 pos, int button) {
    if (!m_visible || !m_enabled) return false;

    // Check children first
    for (auto it = m_children.rbegin(); it != m_children.rend(); ++it) {
        if ((*it)->onMouseUp(pos, button)) {
            return true;
        }
    }

    return containsPoint(pos);
}

} // namespace ui
} // namespace spectra
