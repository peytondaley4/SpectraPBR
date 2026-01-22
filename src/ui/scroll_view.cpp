#include "scroll_view.h"
#include "text/text_layout.h"
#include <algorithm>

namespace spectra {
namespace ui {

ScrollView::ScrollView() {
    setSize(200.0f, 300.0f);
}

void ScrollView::setScrollOffset(float offset) {
    m_scrollOffset = offset;
    clampScrollOffset();
    markDirty();
}

void ScrollView::scrollBy(float delta) {
    setScrollOffset(m_scrollOffset + delta);
}

void ScrollView::scrollToY(float y) {
    // Scroll to make the given Y position visible
    if (y < m_scrollOffset) {
        setScrollOffset(y);
    } else if (y > m_scrollOffset + m_size.y) {
        setScrollOffset(y - m_size.y);
    }
}

Rect ScrollView::getVisibleBounds() const {
    Rect bounds = getAbsoluteBounds();
    if (m_showScrollbar && needsScrolling()) {
        bounds.width -= m_scrollbarWidth;
    }
    return bounds;
}

float2 ScrollView::getChildOffset() const {
    return make_float2(0.0f, -m_scrollOffset);
}

void ScrollView::clampScrollOffset() {
    float maxScroll = std::max(0.0f, m_contentHeight - m_size.y);
    m_scrollOffset = std::clamp(m_scrollOffset, 0.0f, maxScroll);
}

float ScrollView::getScrollbarHeight() const {
    if (m_contentHeight <= 0.0f) return m_size.y;
    float ratio = m_size.y / m_contentHeight;
    return std::max(20.0f, m_size.y * ratio); // Minimum scrollbar height of 20px
}

float ScrollView::getScrollbarY() const {
    float maxScroll = std::max(1.0f, m_contentHeight - m_size.y);
    float scrollRatio = m_scrollOffset / maxScroll;
    float trackHeight = m_size.y - getScrollbarHeight();
    return scrollRatio * trackHeight;
}

Rect ScrollView::getScrollbarRect() const {
    Rect bounds = getAbsoluteBounds();
    return {
        bounds.right() - m_scrollbarWidth,
        bounds.y + getScrollbarY(),
        m_scrollbarWidth,
        getScrollbarHeight()
    };
}

bool ScrollView::onMouseScroll(float2 pos, float delta) {
    if (!m_visible || !m_enabled) return false;
    
    Rect bounds = getAbsoluteBounds();
    if (!bounds.contains(pos)) return false;
    
    if (needsScrolling()) {
        scrollBy(-delta * m_scrollSpeed);
        return true;
    }
    
    return false;
}

bool ScrollView::onMouseDown(float2 pos, int button) {
    if (!m_visible || !m_enabled) return false;
    
    // Check scrollbar click
    if (button == 0 && m_showScrollbar && needsScrolling()) {
        Rect scrollbarRect = getScrollbarRect();
        if (scrollbarRect.contains(pos)) {
            m_scrollbarDragging = true;
            m_dragStartOffset = m_scrollOffset;
            m_dragStartY = pos.y;
            return true;
        }
        
        // Click on track - jump to position
        Rect bounds = getAbsoluteBounds();
        Rect trackRect = {
            bounds.right() - m_scrollbarWidth,
            bounds.y,
            m_scrollbarWidth,
            bounds.height
        };
        if (trackRect.contains(pos)) {
            float clickRatio = (pos.y - bounds.y) / bounds.height;
            float maxScroll = std::max(1.0f, m_contentHeight - m_size.y);
            setScrollOffset(clickRatio * maxScroll);
            return true;
        }
    }
    
    // Forward to children with scroll offset applied
    Rect visibleBounds = getVisibleBounds();
    if (visibleBounds.contains(pos)) {
        // Adjust position for scroll offset
        float2 adjustedPos = make_float2(pos.x, pos.y + m_scrollOffset);
        
        for (auto it = m_children.rbegin(); it != m_children.rend(); ++it) {
            if ((*it)->isVisible() && (*it)->onMouseDown(adjustedPos, button)) {
                return true;
            }
        }
    }
    
    return containsPoint(pos);
}

bool ScrollView::onMouseUp(float2 pos, int button) {
    if (!m_visible || !m_enabled) return false;
    
    bool wasScrollbarDragging = m_scrollbarDragging;
    m_scrollbarDragging = false;
    
    if (wasScrollbarDragging) {
        return true;
    }
    
    // Forward to children with scroll offset applied
    Rect visibleBounds = getVisibleBounds();
    float2 adjustedPos = make_float2(pos.x, pos.y + m_scrollOffset);
    
    for (auto it = m_children.rbegin(); it != m_children.rend(); ++it) {
        if ((*it)->isVisible() && (*it)->onMouseUp(adjustedPos, button)) {
            return true;
        }
    }
    
    return false;
}

bool ScrollView::onMouseMove(float2 pos) {
    if (!m_visible || !m_enabled) return false;
    
    // Handle scrollbar dragging
    if (m_scrollbarDragging) {
        float deltaY = pos.y - m_dragStartY;
        float trackHeight = m_size.y - getScrollbarHeight();
        if (trackHeight > 0) {
            float maxScroll = std::max(1.0f, m_contentHeight - m_size.y);
            float scrollDelta = (deltaY / trackHeight) * maxScroll;
            setScrollOffset(m_dragStartOffset + scrollDelta);
        }
        return true;
    }
    
    // Update scrollbar hover state
    if (m_showScrollbar && needsScrolling()) {
        Rect scrollbarRect = getScrollbarRect();
        bool wasHovered = m_scrollbarHovered;
        m_scrollbarHovered = scrollbarRect.contains(pos);
        if (wasHovered != m_scrollbarHovered) {
            markDirty();
        }
    }
    
    // Forward to children with scroll offset applied
    Rect visibleBounds = getVisibleBounds();
    if (visibleBounds.contains(pos)) {
        float2 adjustedPos = make_float2(pos.x, pos.y + m_scrollOffset);
        
        for (auto it = m_children.rbegin(); it != m_children.rend(); ++it) {
            if ((*it)->isVisible()) {
                (*it)->onMouseMove(adjustedPos);
            }
        }
    }
    
    return containsPoint(pos);
}

void ScrollView::collectGeometry(std::vector<UIQuad>& outQuads, text::TextLayout* textLayout) {
    if (!m_visible) return;
    
    // Generate our own geometry (background, scrollbar)
    generateGeometry(outQuads, textLayout);
    
    // Get clip bounds for children
    Rect clipBounds = getVisibleBounds();
    
    // Collect child geometry with scroll offset and clipping applied
    size_t startIdx = outQuads.size();
    
    for (auto& child : m_children) {
        if (!child->isVisible()) continue;
        
        // Get child's absolute position and check if visible
        Rect childBounds = child->getAbsoluteBounds();
        float scrolledY = childBounds.y - m_scrollOffset;
        
        // Skip if completely outside visible area
        if (scrolledY + childBounds.height < clipBounds.y || scrolledY > clipBounds.bottom()) {
            continue;
        }
        
        // Temporarily adjust child position for geometry generation
        float2 originalPos = child->getPosition();
        child->setPosition(originalPos.x, originalPos.y - m_scrollOffset);
        
        // Collect child geometry
        child->collectGeometry(outQuads, textLayout);
        
        // Restore original position
        child->setPosition(originalPos);
    }
    
    // Apply clip bounds to all child quads that were just added
    for (size_t i = startIdx; i < outQuads.size(); i++) {
        outQuads[i].clipMinX = clipBounds.x;
        outQuads[i].clipMinY = clipBounds.y;
        outQuads[i].clipMaxX = clipBounds.right();
        outQuads[i].clipMaxY = clipBounds.bottom();
    }
    
    m_dirty = false;
}

void ScrollView::generateGeometry(std::vector<UIQuad>& outQuads, text::TextLayout* textLayout) {
    (void)textLayout;
    
    const Theme* theme = getTheme();
    Rect bounds = getAbsoluteBounds();
    float depth = getEffectiveDepth();
    
    // Background
    outQuads.push_back(makeSolidQuad(bounds, theme->panelBackground, depth));
    
    // Draw scrollbar
    if (m_showScrollbar && needsScrolling()) {
        // Scrollbar track
        Rect trackRect = {
            bounds.right() - m_scrollbarWidth,
            bounds.y,
            m_scrollbarWidth,
            bounds.height
        };
        outQuads.push_back(makeSolidQuad(trackRect, theme->scrollbarTrack, depth + 0.01f));
        
        // Scrollbar thumb
        Rect thumbRect = getScrollbarRect();
        float4 thumbColor = m_scrollbarDragging ? theme->scrollbarThumbActive :
                           (m_scrollbarHovered ? theme->scrollbarThumbHover : theme->scrollbarThumb);
        outQuads.push_back(makeSolidQuad(thumbRect, thumbColor, depth + 0.02f));
    }
}

} // namespace ui
} // namespace spectra
