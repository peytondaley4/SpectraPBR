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
struct InspectedCellInfo;

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
    PropertyPanel* getPropertyPanel() { return m_propertyPanel; }

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
    Panel* getTopBar() { return m_topBar; }
    Panel* getScenePanel() { return m_scenePanel; }

    // Add a custom widget to the root level
    void addRootWidget(std::unique_ptr<Widget> widget);

    //--------------------------------------------------------------------------
    // Path Guide Grid Debug Panel
    //--------------------------------------------------------------------------

    // Add debug panel for path guide grid (bounds, sparse cells, show in viewport, level, automation)
    void addPathGuideGridDebugPanel(
        uint32_t numLevels, uint32_t totalCells, uint32_t entryStride,
        uint32_t baseResolution, float perLevelScale,
        const float boundsMin[3], const float boundsMax[3],
        bool initialVisualize,
        std::function<void(bool)> onVisualize,
        std::function<void(uint32_t)> onLevel,
        std::function<void()> onBuild,
        std::function<void(bool)> onEnableGuiding = nullptr,
        std::function<void()> onPause = nullptr,
        std::function<void()> onBuildAndStep = nullptr);

    void toggleGridDebugPanel();
    bool isGridDebugPanelVisible() const;

    // Update the "Sparse cells" (and related) label after a build; call with current grid stats
    void updatePathGuideGridStats(uint32_t numLevels, uint32_t totalCells, uint32_t entryStride);

    // Update automation status label
    void updatePathGuideAutomationStatus(const char* modeStr, uint32_t framesSinceBuild, uint32_t totalBuilds);

    //--------------------------------------------------------------------------
    // Cell Inspector Panel
    //--------------------------------------------------------------------------
    void addCellInspectorPanel();
    void updateCellInspectorData(const InspectedCellInfo& info);
    void showCellInspectorPanel(bool show);

private:
    void createDefaultUI();
    void onSceneNodeSelected(TreeNode* node);
    void onSceneNodeDoubleClicked(TreeNode* node);
    void onSceneNodeExpanded(TreeNode* node, bool expanded);
    void clearTreeSelection(Widget* widget, TreeNode* except);
    void buildTreeNodeRecursive(uint32_t nodeIndex, int indentLevel, float& yOffset);

public:
    // Open the property panel showing the given instance's material — used by
    // viewport double-click (works during path guiding, where single clicks
    // deliberately leave selection and accumulation untouched) and by the
    // scene-tree double-click.
    void showInstancePropertiesFor(uint32_t instanceId);

private:
    // Raise a root widget to the top of the stack (just below the pinned top
    // bar) and reassign root depths to match the new order.
    void bringToFront(Widget* widget);
    // Derive each root widget's render depth from its position in
    // m_rootWidgets so draw order and input order always agree. Called
    // whenever the root list changes.
    void updateRootDepths();
    // Keep a root widget reachable: clamp its position so part of it stays on
    // screen (used during drags and after window resizes).
    void clampRootToScreen(Widget* widget);

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

    // All root-level widgets in draw/input order. Last element is the pinned
    // top bar; clicking any other root raises it to just below the top bar
    // (bringToFront), and updateRootDepths keeps quad depths in sync with
    // this order so the panel that draws on top is also the one that gets
    // input first.
    std::vector<std::unique_ptr<Widget>> m_rootWidgets;
    // Mouse capture: the root widget that consumed the last mouse-down
    // receives ALL moves until mouse-up, so drags don't stall when the cursor
    // crosses other panels.
    Widget* m_mouseCaptureWidget = nullptr;
    Button* m_sceneToggleBtn = nullptr;   // top-bar toggle, synced with panel close
    Panel* m_topBar = nullptr;
    Panel* m_scenePanel = nullptr;
    PropertyPanel* m_propertyPanel = nullptr;
    Panel* m_gridDebugPanel = nullptr;
    Label* m_pathGuideMetaLabel = nullptr;  // "Levels: N  Sparse cells: M  Stride: S" (owned by panel)
    Label* m_pathGuideStatusLabel = nullptr;  // "Running | 45/60 frames | 3 builds"

    // Cell inspector panel
    Panel* m_cellInspectorPanel = nullptr;
    Label* m_cellPosLabel = nullptr;
    Label* m_cellLevelLabel = nullptr;
    Label* m_cellAABBLabel = nullptr;
    Label* m_cellLobe0Label = nullptr;
    Label* m_cellLobe1Label = nullptr;
    Label* m_cellStatsLabel = nullptr;
    Label* m_cellStatusLabel = nullptr;

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
