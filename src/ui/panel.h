#pragma once

#include "widget.h"

namespace spectra {
namespace ui {

//------------------------------------------------------------------------------
// Panel - Container widget with background and optional border
//------------------------------------------------------------------------------
class Panel : public Widget {
public:
    Panel();
    ~Panel() override = default;

    // Border settings
    void setBorderWidth(float width) { m_borderWidth = width; markDirty(); }
    float getBorderWidth() const { return m_borderWidth; }

    void setShowBorder(bool show) { m_showBorder = show; markDirty(); }
    bool getShowBorder() const { return m_showBorder; }

    // Header settings (optional header bar at top)
    void setTitle(const std::string& title) { m_title = title; markDirty(); }
    const std::string& getTitle() const { return m_title; }

    void setShowHeader(bool show) { m_showHeader = show; markDirty(); }
    bool getShowHeader() const { return m_showHeader; }

    void setHeaderHeight(float height) { m_headerHeight = height; markDirty(); }
    float getHeaderHeight() const { return m_headerHeight; }

    // Get content area (panel bounds minus header if shown)
    Rect getContentBounds() const;

    // Draggable panel by header
    void setDraggable(bool draggable) { m_draggable = draggable; }
    bool isDraggable() const { return m_draggable; }

    // Closeable (X button in header)
    void setCloseable(bool closeable) { m_closeable = closeable; markDirty(); }
    bool isCloseable() const { return m_closeable; }

    // Close callback
    using CloseCallback = std::function<void()>;
    void setOnClose(CloseCallback callback) { m_onClose = callback; }

protected:
    void generateGeometry(std::vector<UIQuad>& outQuads, text::TextLayout* textLayout) override;
    bool onMouseDown(float2 pos, int button) override;
    bool onMouseUp(float2 pos, int button) override;
    bool onMouseMove(float2 pos) override;

private:
    float m_borderWidth = 1.0f;
    bool m_showBorder = true;
    bool m_showHeader = false;
    float m_headerHeight = 24.0f;
    std::string m_title;
    bool m_draggable = false;
    bool m_closeable = false;
    bool m_dragging = false;
    float2 m_dragOffset = make_float2(0.0f, 0.0f);
    bool m_closeHovered = false;

    CloseCallback m_onClose;
};

} // namespace ui
} // namespace spectra
