#include "widget.h"
#include "text/text_layout.h"
#include <algorithm>

namespace spectra {
namespace ui {

Widget::Widget() {
    // Default initialization
}

void Widget::setPosition(float x, float y) {
    if (m_position.x != x || m_position.y != y) {
        m_position = make_float2(x, y);
        markDirty();
    }
}

void Widget::setSize(float width, float height) {
    if (m_size.x != width || m_size.y != height) {
        m_size = make_float2(width, height);
        markDirty();
    }
}

Rect Widget::getBounds() const {
    return { m_position.x, m_position.y, m_size.x, m_size.y };
}

float2 Widget::getAbsolutePosition() const {
    if (m_parent) {
        float2 parentPos = m_parent->getAbsolutePosition();
        return make_float2(parentPos.x + m_position.x, parentPos.y + m_position.y);
    }
    return m_position;
}

Rect Widget::getAbsoluteBounds() const {
    float2 absPos = getAbsolutePosition();
    return { absPos.x, absPos.y, m_size.x, m_size.y };
}

void Widget::setVisible(bool visible) {
    if (m_visible != visible) {
        m_visible = visible;
        markDirty();
        onVisibilityChanged();
    }
}

void Widget::setEnabled(bool enabled) {
    if (m_enabled != enabled) {
        m_enabled = enabled;
        markDirty();
        onEnabledChanged();
    }
}

void Widget::addChild(std::unique_ptr<Widget> child) {
    child->m_parent = this;
    // Inherit theme if child doesn't have one
    if (!child->m_theme) {
        child->m_theme = m_theme;
    }
    m_children.push_back(std::move(child));
    markDirty();
}

void Widget::removeChild(Widget* child) {
    auto it = std::find_if(m_children.begin(), m_children.end(),
        [child](const std::unique_ptr<Widget>& w) { return w.get() == child; });

    if (it != m_children.end()) {
        (*it)->m_parent = nullptr;
        m_children.erase(it);
        markDirty();
    }
}

void Widget::clearChildren() {
    for (auto& child : m_children) {
        child->m_parent = nullptr;
    }
    m_children.clear();
    markDirty();
}

void Widget::setTheme(const Theme* theme) {
    m_theme = theme;
    // Propagate to children that don't have explicit themes
    for (auto& child : m_children) {
        if (!child->m_theme || child->m_theme == m_theme) {
            child->setTheme(theme);
        }
    }
    markDirty();
}

const Theme* Widget::getTheme() const {
    if (m_theme) return m_theme;
    if (m_parent) return m_parent->getTheme();
    return &THEME_DARK; // Default fallback
}

void Widget::update(float deltaTime) {
    if (!m_visible) return;

    // Update children
    for (auto& child : m_children) {
        child->update(deltaTime);
    }
}

void Widget::collectGeometry(std::vector<UIQuad>& outQuads, text::TextLayout* textLayout) {
    if (!m_visible) return;

    // Generate this widget's geometry
    generateGeometry(outQuads, textLayout);

    // Collect children's geometry
    for (auto& child : m_children) {
        child->collectGeometry(outQuads, textLayout);
    }

    m_dirty = false;
}

void Widget::markDirty() {
    m_dirty = true;
    // Propagate dirty flag up to parents so geometry regeneration triggers
    if (m_parent) m_parent->markDirty();
}

bool Widget::onMouseMove(float2 pos) {
    if (!m_visible || !m_enabled) return false;

    // Check children first (front to back, assuming children are rendered on top)
    for (auto it = m_children.rbegin(); it != m_children.rend(); ++it) {
        if ((*it)->onMouseMove(pos)) {
            return true;
        }
    }

    // Check if mouse is over this widget
    bool wasHovered = m_hovered;
    m_hovered = containsPoint(pos);

    if (m_hovered != wasHovered) {
        markDirty();
        onHoverChanged();
    }

    return m_hovered;
}

bool Widget::onMouseDown(float2 pos, int button) {
    if (!m_visible || !m_enabled) return false;

    // Check children first
    for (auto it = m_children.rbegin(); it != m_children.rend(); ++it) {
        if ((*it)->onMouseDown(pos, button)) {
            return true;
        }
    }

    if (containsPoint(pos)) {
        m_active = true;
        markDirty();
        onActiveChanged();
        return true;
    }

    return false;
}

bool Widget::onMouseUp(float2 pos, int button) {
    if (!m_visible || !m_enabled) return false;

    // Check children first
    for (auto it = m_children.rbegin(); it != m_children.rend(); ++it) {
        if ((*it)->onMouseUp(pos, button)) {
            return true;
        }
    }

    bool wasActive = m_active;
    m_active = false;

    if (wasActive) {
        markDirty();
        onActiveChanged();
        return containsPoint(pos);
    }

    return false;
}

bool Widget::onMouseScroll(float2 pos, float delta) {
    if (!m_visible || !m_enabled) return false;

    // Check children first
    for (auto it = m_children.rbegin(); it != m_children.rend(); ++it) {
        if ((*it)->onMouseScroll(pos, delta)) {
            return true;
        }
    }

    return false;
}

bool Widget::onKeyDown(int key, int mods) {
    if (!m_visible || !m_enabled) return false;

    // Forward to focused child
    for (auto& child : m_children) {
        if (child->isFocused() && child->onKeyDown(key, mods)) {
            return true;
        }
    }

    return false;
}

bool Widget::onKeyUp(int key, int mods) {
    if (!m_visible || !m_enabled) return false;

    // Forward to focused child
    for (auto& child : m_children) {
        if (child->isFocused() && child->onKeyUp(key, mods)) {
            return true;
        }
    }

    return false;
}

bool Widget::containsPoint(float2 pos) const {
    Rect bounds = getAbsoluteBounds();
    return bounds.contains(pos);
}

void Widget::generateGeometry(std::vector<UIQuad>& outQuads, text::TextLayout* textLayout) {
    // Base class generates no geometry - override in derived classes
    (void)outQuads;
    (void)textLayout;
}

float Widget::getEffectiveDepth() const {
    float depth = m_depth;
    if (m_parent) {
        depth += m_parent->getEffectiveDepth() + 0.001f; // Slight offset for children
    }
    return depth;
}

} // namespace ui
} // namespace spectra
