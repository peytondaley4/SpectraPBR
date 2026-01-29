#pragma once

#include "widget.h"
#include "../scene/scene_hierarchy.h"
#include <functional>
#include <chrono>
#include <optional>

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

    // Scene node type (for property panel dispatching)
    void setNodeType(SceneNodeType type) { m_nodeType = type; }
    SceneNodeType getNodeType() const { return m_nodeType; }

    // Node index in hierarchy (for hierarchy lookups)
    void setNodeIndex(uint32_t index) { m_nodeIndex = index; }
    uint32_t getNodeIndex() const { return m_nodeIndex; }

    // Callbacks
    using SelectCallback = std::function<void(TreeNode*)>;
    using ExpandCallback = std::function<void(TreeNode*, bool)>;
    using DoubleClickCallback = std::function<void(TreeNode*)>;

    void setOnSelect(SelectCallback callback) { m_onSelect = callback; }
    void setOnExpand(ExpandCallback callback) { m_onExpand = callback; }
    void setOnDoubleClick(DoubleClickCallback callback) { m_onDoubleClick = callback; }

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

    // Scene node type for property panel
    SceneNodeType m_nodeType = SceneNodeType::Instance;
    uint32_t m_nodeIndex = UINT32_MAX;

    // Double-click tracking
    std::optional<std::chrono::steady_clock::time_point> m_lastClickTime;
    static constexpr int DOUBLE_CLICK_MS = 300;

    SelectCallback m_onSelect;
    ExpandCallback m_onExpand;
    DoubleClickCallback m_onDoubleClick;
};

} // namespace ui
} // namespace spectra
