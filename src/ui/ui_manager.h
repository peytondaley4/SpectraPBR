#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <memory>
#include <functional>
#include "ui_types.h"
#include "theme.h"
#include "widget.h"
#include "panel.h"
#include "button.h"
#include "label.h"
#include "tree_node.h"
#include "scroll_view.h"
#include "property_panel.h"
#include "text/font_atlas.h"
#include "text/text_layout.h"
#include "../scene/scene_hierarchy.h"

namespace spectra {

// Forward declarations
class SceneManager;
class MaterialManager;

namespace ui {

//------------------------------------------------------------------------------
// UI Manager - Central coordinator for the UI system
//------------------------------------------------------------------------------
class UIManager {
public:
    UIManager();
    ~UIManager();

    // Initialize the UI system
    bool init(text::FontAtlas* fontAtlas, uint32_t screenWidth, uint32_t screenHeight);

    // Shutdown the UI system
    void shutdown();

    // Set the current theme
    void setTheme(const Theme* theme);
    const Theme* getTheme() const { return m_theme; }

    // Toggle between light and dark themes
    void toggleTheme();
    bool isDarkTheme() const { return m_theme == &THEME_DARK; }

    // Update screen dimensions
    void setScreenSize(uint32_t width, uint32_t height);
    uint32_t getScreenWidth() const { return m_screenWidth; }
    uint32_t getScreenHeight() const { return m_screenHeight; }

    // Update all widgets
    void update(float deltaTime);

    // Collect all UI geometry for rendering (only regenerates when dirty)
    void collectGeometry();
    const std::vector<UIQuad>& getQuads() const { return m_quads; }
    
    // Force geometry regeneration on next collectGeometry call
    void markGeometryDirty() { m_geometryDirty = true; }

    //--------------------------------------------------------------------------
    // Input Handling (return true if event was consumed by UI)
    //--------------------------------------------------------------------------
    bool handleMouseMove(float2 pos);
    bool handleMouseDown(float2 pos, int button);
    bool handleMouseUp(float2 pos, int button);
    bool handleMouseScroll(float2 pos, float delta);
    bool handleKeyDown(int key, int mods);
    bool handleKeyUp(int key, int mods);

    //--------------------------------------------------------------------------
    // Scene Hierarchy Panel
    //--------------------------------------------------------------------------

    // Set the scene hierarchy data
    void setSceneHierarchy(SceneHierarchy* hierarchy) { m_hierarchy = hierarchy; }

    // Build the hierarchical scene tree from the hierarchy data
    void buildHierarchicalSceneTree();

    // Build the scene tree from the scene manager (legacy flat list)
    void buildSceneTree(const SceneManager* sceneManager);

    // Clear the scene tree
    void clearSceneTree();

    // Set callback for when a scene object is selected
    using SelectionCallback = std::function<void(uint32_t instanceId)>;
    void setSelectionCallback(SelectionCallback callback) { m_selectionCallback = callback; }

    // Get/set selected instance ID
    uint32_t getSelectedInstanceId() const { return m_selectedInstanceId; }
    void setSelectedInstanceId(uint32_t id);

    //--------------------------------------------------------------------------
    // Property Panel
    //--------------------------------------------------------------------------

    // Show/hide property panel
    void showPropertyPanel(bool show);
    bool isPropertyPanelVisible() const;

    // Toggle property panel
    void togglePropertyPanel();

    // Get property panel
    PropertyPanel* getPropertyPanel() { return m_propertyPanel.get(); }

    // Set material manager for property lookups
    void setMaterialManager(MaterialManager* matMgr) { m_materialManager = matMgr; }

    // Light edit callback
    using LightEditCallback = std::function<void(SceneNodeType, uint32_t, const LightInfo&)>;
    void setOnLightEdit(LightEditCallback callback);

    // Light info request callback (to get light data from LightManager)
    using LightInfoRequestCallback = std::function<LightInfo(SceneNodeType, uint32_t)>;
    void setLightInfoRequestCallback(LightInfoRequestCallback callback) { m_lightInfoRequestCallback = callback; }

    // Instance info request callback (to get material data per instance)
    using InstanceInfoRequestCallback = std::function<InstanceInfo(uint32_t)>;
    void setInstanceInfoRequestCallback(InstanceInfoRequestCallback callback) { m_instanceInfoRequestCallback = callback; }

    // Preview textures for UI rendering (set by the instance info callback)
    void setPreviewTextures(const std::vector<cudaTextureObject_t>& textures) { 
        if (m_previewTextures != textures) {
            m_previewTextures = textures;
            m_texturesChanged = true;
        }
    }
    const std::vector<cudaTextureObject_t>& getPreviewTextures() const { return m_previewTextures; }
    void clearPreviewTextures() { m_previewTextures.clear(); m_texturesChanged = true; }
    bool texturesChanged() const { return m_texturesChanged; }
    void clearTexturesChanged() { m_texturesChanged = false; }

    // Material edit callback
    using MaterialEditCallback = std::function<void(uint32_t instanceId, const GpuMaterial& material)>;
    void setOnMaterialEdit(MaterialEditCallback callback);

    //--------------------------------------------------------------------------
    // Top Bar
    //--------------------------------------------------------------------------

    // Toggle scene hierarchy panel visibility
    void toggleScenePanel();
    bool isScenePanelVisible() const;

    //--------------------------------------------------------------------------
    // Widget Access
    //--------------------------------------------------------------------------

    // Get the root widgets
    Panel* getTopBar() { return m_topBar.get(); }
    Panel* getScenePanel() { return m_scenePanel.get(); }

    // Add a custom widget to the root level
    void addRootWidget(std::unique_ptr<Widget> widget);

private:
    void createDefaultUI();
    void onSceneNodeSelected(TreeNode* node);
    void onSceneNodeDoubleClicked(TreeNode* node);
    void onSceneNodeExpanded(TreeNode* node, bool expanded);
    void clearTreeSelection(Widget* widget, TreeNode* except);
    void buildTreeNodeRecursive(uint32_t nodeIndex, int indentLevel, float& yOffset);

    text::FontAtlas* m_fontAtlas = nullptr;
    text::TextLayout m_textLayout;

    const Theme* m_theme = &THEME_DARK;

    uint32_t m_screenWidth = 1920;
    uint32_t m_screenHeight = 1080;

    // Collected geometry
    std::vector<UIQuad> m_quads;
    bool m_geometryDirty = true;  // Start dirty to collect on first frame
    uint64_t m_geometryGeneration = 0;  // Increments when geometry is regenerated
    
public:
    // Get geometry generation counter (changes when quads are regenerated)
    uint64_t getGeometryGeneration() const { return m_geometryGeneration; }
    
private:

    // Root widgets
    std::unique_ptr<Panel> m_topBar;
    std::unique_ptr<Panel> m_scenePanel;
    std::unique_ptr<PropertyPanel> m_propertyPanel;
    std::vector<std::unique_ptr<Widget>> m_rootWidgets;

    // Scene hierarchy data
    SceneHierarchy* m_hierarchy = nullptr;  // Non-owning
    MaterialManager* m_materialManager = nullptr;  // Non-owning

    // Scene tree view
    ScrollView* m_sceneScrollView = nullptr;  // Non-owning, owned by m_scenePanel
    std::vector<TreeNode*> m_sceneNodes;  // Non-owning pointers to nodes in scroll view
    uint32_t m_selectedInstanceId = UINT32_MAX;
    SelectionCallback m_selectionCallback;
    LightEditCallback m_lightEditCallback;
    LightInfoRequestCallback m_lightInfoRequestCallback;
    InstanceInfoRequestCallback m_instanceInfoRequestCallback;
    MaterialEditCallback m_materialEditCallback;

    // Preview textures for UI texture previews
    std::vector<cudaTextureObject_t> m_previewTextures;
    bool m_texturesChanged = false;
};

} // namespace ui
} // namespace spectra
