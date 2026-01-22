#pragma once

#include "widget.h"
#include <functional>

namespace spectra {
namespace ui {

//------------------------------------------------------------------------------
// TreeNode - Expandable tree item for hierarchical lists
//------------------------------------------------------------------------------
class TreeNode : public Widget {
public:
    TreeNode();
    explicit TreeNode(const std::string& label);
    ~TreeNode() override = default;

    // Label
    void setLabel(const std::string& label) { m_label = label; markDirty(); }
    const std::string& getLabel() const { return m_label; }

    // Expand state
    void setExpanded(bool expanded);
    bool isExpanded() const { return m_expanded; }
    void toggleExpanded() { setExpanded(!m_expanded); }

    // Selection
    void setSelected(bool selected);
    bool isSelected() const { return m_selected; }

    // Indent level (for visual hierarchy)
    void setIndentLevel(int level) { m_indentLevel = level; markDirty(); }
    int getIndentLevel() const { return m_indentLevel; }

    // Has children (shows expand icon)
    void setHasChildren(bool hasChildren) { m_hasChildren = hasChildren; markDirty(); }
    bool hasChildren() const { return m_hasChildren; }

    // User data (e.g., instance ID)
    void setUserData(uint32_t data) { m_userData = data; }
    uint32_t getUserData() const { return m_userData; }

    // Callbacks
    using SelectCallback = std::function<void(TreeNode*)>;
    using ExpandCallback = std::function<void(TreeNode*, bool)>;

    void setOnSelect(SelectCallback callback) { m_onSelect = callback; }
    void setOnExpand(ExpandCallback callback) { m_onExpand = callback; }

    // Row height
    static constexpr float ROW_HEIGHT = 22.0f;
    static constexpr float INDENT_WIDTH = 16.0f;
    static constexpr float EXPANDER_SIZE = 14.0f;

protected:
    void generateGeometry(std::vector<UIQuad>& outQuads, text::TextLayout* textLayout) override;
    bool onMouseDown(float2 pos, int button) override;
    bool onMouseUp(float2 pos, int button) override;

private:
    Rect getExpanderBounds() const;

    std::string m_label;
    bool m_expanded = false;
    bool m_selected = false;
    int m_indentLevel = 0;
    bool m_hasChildren = false;
    uint32_t m_userData = 0;
    bool m_expanderHovered = false;

    SelectCallback m_onSelect;
    ExpandCallback m_onExpand;
};

} // namespace ui
} // namespace spectra
