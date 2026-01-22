#pragma once
#include "widget.h"

namespace spectra {
namespace ui {

//------------------------------------------------------------------------------
// ScrollView - A scrollable container widget
//------------------------------------------------------------------------------
class ScrollView : public Widget {
public:
    ScrollView();
    virtual ~ScrollView() = default;

    // Set the total content height (determines scroll range)
    void setContentHeight(float height) { m_contentHeight = height; markDirty(); }
    float getContentHeight() const { return m_contentHeight; }

    // Get/set scroll offset
    void setScrollOffset(float offset);
    float getScrollOffset() const { return m_scrollOffset; }

    // Scroll by a delta amount
    void scrollBy(float delta);

    // Scroll to make a Y position visible
    void scrollToY(float y);

    // Check if scrolling is needed
    bool needsScrolling() const { return m_contentHeight > m_size.y; }

    // Get the visible content bounds (for clipping)
    Rect getVisibleBounds() const;

    // Scrollbar settings
    void setShowScrollbar(bool show) { m_showScrollbar = show; markDirty(); }
    bool getShowScrollbar() const { return m_showScrollbar; }
    
    void setScrollbarWidth(float width) { m_scrollbarWidth = width; markDirty(); }
    float getScrollbarWidth() const { return m_scrollbarWidth; }

    // Input handling
    bool onMouseScroll(float2 pos, float delta) override;
    bool onMouseDown(float2 pos, int button) override;
    bool onMouseUp(float2 pos, int button) override;
    bool onMouseMove(float2 pos) override;

    // Override to handle children with clipping (prevents double rendering)
    void collectGeometry(std::vector<UIQuad>& outQuads, text::TextLayout* textLayout) override;

protected:
    void generateGeometry(std::vector<UIQuad>& outQuads, text::TextLayout* textLayout) override;

    // Override to apply scroll offset to child positions
    float2 getChildOffset() const;

private:
    float m_contentHeight = 0.0f;     // Total height of content
    float m_scrollOffset = 0.0f;      // Current scroll position (0 = top)
    float m_scrollSpeed = 30.0f;      // Pixels per scroll tick
    
    // Scrollbar
    bool m_showScrollbar = true;
    float m_scrollbarWidth = 8.0f;
    bool m_scrollbarHovered = false;
    bool m_scrollbarDragging = false;
    float m_dragStartOffset = 0.0f;
    float m_dragStartY = 0.0f;

    // Calculate scrollbar dimensions
    float getScrollbarHeight() const;
    float getScrollbarY() const;
    Rect getScrollbarRect() const;
    
    // Clamp scroll offset to valid range
    void clampScrollOffset();
};

} // namespace ui
} // namespace spectra
