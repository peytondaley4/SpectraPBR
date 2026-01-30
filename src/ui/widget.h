#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <memory>
#include <string>
#include "ui_types.h"
#include "theme.h"

namespace spectra {

// Forward declarations
namespace text {
    class TextLayout;
}

namespace ui {

//------------------------------------------------------------------------------
// Base Widget Class
// All UI elements derive from this class
//------------------------------------------------------------------------------
class Widget {
public:
    Widget();
    virtual ~Widget() = default;

    // Non-copyable, but movable
    Widget(const Widget&) = delete;
    Widget& operator=(const Widget&) = delete;
    Widget(Widget&&) = default;
    Widget& operator=(Widget&&) = default;

    //--------------------------------------------------------------------------
    // Layout and Transform
    //--------------------------------------------------------------------------

    // Set position (relative to parent if has parent, else screen coords)
    void setPosition(float x, float y);
    void setPosition(float2 pos) { setPosition(pos.x, pos.y); }
    float2 getPosition() const { return m_position; }
    
    // Set position without marking dirty (for temporary transforms during rendering)
    void setPositionDirect(float x, float y) { m_position = make_float2(x, y); }
    void setPositionDirect(float2 pos) { m_position = pos; }
    
    // Set size without marking dirty (for layout during rendering)
    void setSizeDirect(float w, float h) { m_size = make_float2(w, h); }
    void setSizeDirect(float2 size) { m_size = size; }

    // Set size
    void setSize(float width, float height);
    void setSize(float2 size) { setSize(size.x, size.y); }
    float2 getSize() const { return m_size; }

    // Get bounds as Rect
    Rect getBounds() const;

    // Get absolute position (in screen coordinates)
    float2 getAbsolutePosition() const;

    // Get absolute bounds
    Rect getAbsoluteBounds() const;

    // Set depth (z-order)
    void setDepth(float depth) { m_depth = depth; markDirty(); }
    float getDepth() const { return m_depth; }

    //--------------------------------------------------------------------------
    // Visibility and State
    //--------------------------------------------------------------------------

    void setVisible(bool visible);
    bool isVisible() const { return m_visible; }

    void setEnabled(bool enabled);
    bool isEnabled() const { return m_enabled; }

    bool isHovered() const { return m_hovered; }
    bool isActive() const { return m_active; }
    bool isFocused() const { return m_focused; }

    //--------------------------------------------------------------------------
    // Hierarchy
    //--------------------------------------------------------------------------

    void addChild(std::unique_ptr<Widget> child);
    void removeChild(Widget* child);
    void clearChildren();

    // Set parent without transferring ownership (for widgets managed elsewhere)
    void setParent(Widget* parent) { m_parent = parent; }

    Widget* getParent() const { return m_parent; }
    const std::vector<std::unique_ptr<Widget>>& getChildren() const { return m_children; }
    size_t getChildCount() const { return m_children.size(); }

    //--------------------------------------------------------------------------
    // Theme
    //--------------------------------------------------------------------------

    void setTheme(const Theme* theme);
    const Theme* getTheme() const;

    //--------------------------------------------------------------------------
    // Update and Rendering
    //--------------------------------------------------------------------------

    // Called each frame to update widget state
    virtual void update(float deltaTime);

    // Generate geometry for this widget and all children
    virtual void collectGeometry(std::vector<UIQuad>& outQuads, text::TextLayout* textLayout);

    // Mark widget as needing geometry regeneration
    void markDirty();
    bool isDirty() const { return m_dirty; }

    //--------------------------------------------------------------------------
    // Input Handling (return true if event was consumed)
    //--------------------------------------------------------------------------

    virtual bool onMouseMove(float2 pos);
    virtual bool onMouseDown(float2 pos, int button);
    virtual bool onMouseUp(float2 pos, int button);
    virtual bool onMouseScroll(float2 pos, float delta);
    virtual bool onKeyDown(int key, int mods);
    virtual bool onKeyUp(int key, int mods);

    // Check if point is inside widget bounds
    bool containsPoint(float2 pos) const;

    //--------------------------------------------------------------------------
    // Name/ID (for debugging and scene tree)
    //--------------------------------------------------------------------------

    void setName(const std::string& name) { m_name = name; }
    const std::string& getName() const { return m_name; }

protected:
    // Override to generate widget-specific geometry
    virtual void generateGeometry(std::vector<UIQuad>& outQuads, text::TextLayout* textLayout);

    // Override to handle state changes
    virtual void onVisibilityChanged() {}
    virtual void onEnabledChanged() {}
    virtual void onHoverChanged() {}
    virtual void onActiveChanged() {}
    virtual void onFocusChanged() {}

    // Helper to get effective depth (considering parent)
    float getEffectiveDepth() const;

    // Position and size
    float2 m_position = make_float2(0.0f, 0.0f);
    float2 m_size = make_float2(100.0f, 30.0f);
    float m_depth = 0.0f;

    // State flags
    bool m_visible = true;
    bool m_enabled = true;
    bool m_hovered = false;
    bool m_active = false;
    bool m_focused = false;
    bool m_dirty = true;

    // Hierarchy
    Widget* m_parent = nullptr;
    std::vector<std::unique_ptr<Widget>> m_children;

    // Theme (nullptr = use parent's theme)
    const Theme* m_theme = nullptr;

    // Name for identification
    std::string m_name;
};

} // namespace ui
} // namespace spectra
