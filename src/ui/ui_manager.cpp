#include "ui_manager.h"
#include "application.h"
#include "scene_manager.h"
#include "material_manager.h"
#include "slider.h"
#include <algorithm>
#include <iostream>
#include <cstdio>
#include <cmath>
#include <GLFW/glfw3.h>

namespace spectra {
namespace ui {

UIManager::UIManager() = default;
UIManager::~UIManager() = default;

bool UIManager::init(text::FontAtlas* fontAtlas, uint32_t screenWidth, uint32_t screenHeight) {
    std::cout << "[UIManager] init() called, fontAtlas=" << fontAtlas 
              << ", isLoaded=" << (fontAtlas ? fontAtlas->isLoaded() : false) << "\n";
    
    if (!fontAtlas || !fontAtlas->isLoaded()) {
        std::cerr << "[UIManager] Font atlas not loaded - UI will be created without text\n";
        // Continue anyway - create UI even without fonts
    }

    m_fontAtlas = fontAtlas;
    if (fontAtlas) {
        m_textLayout.setFontAtlas(fontAtlas);
    }
    m_screenWidth = screenWidth;
    m_screenHeight = screenHeight;

    // Pre-allocate geometry vector to avoid per-frame reallocations
    // Typical UI with property panel uses 200-400 quads
    m_quads.reserve(512);

    createDefaultUI();

    std::cout << "[UIManager] Initialized with screen size " << screenWidth << "x" << screenHeight << "\n";
    return true;
}

void UIManager::shutdown() {
    m_rootWidgets.clear();
    m_topBar = nullptr;
    m_scenePanel = nullptr;
    m_propertyPanel = nullptr;
    m_gridDebugPanel = nullptr;
    m_pathGuideMetaLabel = nullptr;
    m_sceneScrollView = nullptr;
    m_sceneNodes.clear();
    m_quads.clear();
    m_hierarchy = nullptr;
    m_materialManager = nullptr;
}

void UIManager::setTheme(const Theme* theme) {
    if (m_theme == theme) return;
    m_theme = theme;
    m_geometryDirty = true;
    for (auto& widget : m_rootWidgets) {
        widget->setTheme(theme);
    }
}

void UIManager::toggleTheme() {
    if (m_theme == &THEME_DARK) {
        setTheme(&THEME_LIGHT);
    } else {
        setTheme(&THEME_DARK);
    }
}

void UIManager::setScreenSize(uint32_t width, uint32_t height) {
    if (m_screenWidth == width && m_screenHeight == height) return;

    m_screenWidth = width;
    m_screenHeight = height;
    m_geometryDirty = true;

    // Update top bar width
    if (m_topBar) {
        m_topBar->setSize(static_cast<float>(width), 40.0f);
    }

    // Pull panels back into reach — after shrinking the window, panels
    // positioned for the old size could end up entirely off screen.
    for (auto& widget : m_rootWidgets) {
        clampRootToScreen(widget.get());
    }
}

void UIManager::update(float deltaTime) {
    for (auto& widget : m_rootWidgets) {
        if (widget->isVisible()) {
            widget->update(deltaTime);
        }
    }
}

void UIManager::collectGeometry() {
    bool anyDirty = m_geometryDirty;
    if (!anyDirty) {
        for (auto& widget : m_rootWidgets) {
            if (widget->isVisible() && widget->isDirty()) {
                anyDirty = true;
                break;
            }
        }
    }
    if (!anyDirty) return;

    m_quads.clear();
    m_geometryDirty = false;
    m_geometryGeneration++;

    for (auto& widget : m_rootWidgets) {
        if (widget->isVisible()) {
            widget->collectGeometry(m_quads, &m_textLayout);
        }
    }
    // stable_sort: quads with equal depth (e.g. a widget's own layers) keep
    // emission order. Plain sort reorders equal keys arbitrarily per call,
    // which made overlapping same-depth geometry flicker between frames.
    std::stable_sort(m_quads.begin(), m_quads.end(),
        [](const UIQuad& a, const UIQuad& b) { return a.depth < b.depth; });
}

void UIManager::updateRootDepths() {
    // Depth from stack order, so quad sorting reproduces the input order.
    // Spacing of 10 dwarfs the small intra-widget offsets (+0.001 .. +0.02).
    // The top bar (last) is pinned far above everything.
    for (size_t i = 0; i < m_rootWidgets.size(); ++i) {
        Widget* w = m_rootWidgets[i].get();
        float depth = (w == m_topBar) ? 1000.0f : 10.0f + 10.0f * static_cast<float>(i);
        if (w->getDepth() != depth) {
            w->setDepth(depth);
        }
    }
    m_geometryDirty = true;
}

void UIManager::bringToFront(Widget* widget) {
    if (!widget || widget == m_topBar) return;
    if (!m_rootWidgets.empty() && m_rootWidgets.size() >= 2 &&
        m_rootWidgets[m_rootWidgets.size() - 2].get() == widget) {
        return;  // already frontmost (below the pinned top bar)
    }

    auto it = std::find_if(m_rootWidgets.begin(), m_rootWidgets.end(),
        [widget](const std::unique_ptr<Widget>& w) { return w.get() == widget; });
    if (it == m_rootWidgets.end()) return;

    std::unique_ptr<Widget> owned = std::move(*it);
    m_rootWidgets.erase(it);
    // Insert just below the top bar (which stays last/topmost)
    if (!m_rootWidgets.empty() && m_rootWidgets.back().get() == m_topBar) {
        m_rootWidgets.insert(m_rootWidgets.end() - 1, std::move(owned));
    } else {
        m_rootWidgets.push_back(std::move(owned));
    }
    updateRootDepths();
}

void UIManager::clampRootToScreen(Widget* widget) {
    if (!widget || widget == m_topBar) return;

    // Keep enough of the widget on screen to grab it again: at least 60 px
    // horizontally and the header strip vertically.
    const float keep = 60.0f;
    float2 pos = widget->getPosition();
    float2 size = widget->getSize();
    float maxX = static_cast<float>(m_screenWidth) - keep;
    float minX = keep - size.x;
    float maxY = static_cast<float>(m_screenHeight) - 30.0f;
    float minY = 0.0f;

    float x = std::min(std::max(pos.x, minX), maxX);
    float y = std::min(std::max(pos.y, minY), maxY);
    if (x != pos.x || y != pos.y) {
        widget->setPosition(x, y);
    }
}

bool UIManager::handleMouseMove(float2 pos) {
    // While a drag is in progress, the capturing widget gets EVERY move —
    // routing through the normal front-to-back walk let other panels steal
    // the event mid-drag, stalling panel/slider drags ("janky" drags).
    if (m_mouseCaptureWidget) {
        m_mouseCaptureWidget->onMouseMove(pos);
        clampRootToScreen(m_mouseCaptureWidget);
        return true;
    }

    bool consumed = false;
    for (auto it = m_rootWidgets.rbegin(); it != m_rootWidgets.rend(); ++it) {
        if (!(*it)->isVisible()) continue;
        if (!consumed && (*it)->onMouseMove(pos)) {
            consumed = true;
            continue;  // keep iterating: the roots below must clear hover
        }
        if (consumed) {
            (*it)->clearHoverRecursive();
        }
    }
    return consumed;
}

bool UIManager::handleMouseDown(float2 pos, int button) {
    m_mouseCaptureWidget = nullptr;  // safety: stale capture can't survive a new press
    for (auto it = m_rootWidgets.rbegin(); it != m_rootWidgets.rend(); ++it) {
        Widget* root = it->get();
        if (root->isVisible() && root->onMouseDown(pos, button)) {
            // Capture the mouse for the duration of this press and raise the
            // panel: the panel you click is the one on top and the one that
            // keeps receiving the drag.
            m_mouseCaptureWidget = root;
            bringToFront(root);
            return true;
        }
    }
    return false;
}

bool UIManager::handleMouseUp(float2 pos, int button) {
    if (m_mouseCaptureWidget) {
        Widget* captured = m_mouseCaptureWidget;
        m_mouseCaptureWidget = nullptr;
        captured->onMouseUp(pos, button);
        return true;  // the press was UI-owned, so the release is too
    }

    for (auto it = m_rootWidgets.rbegin(); it != m_rootWidgets.rend(); ++it) {
        if ((*it)->isVisible() && (*it)->onMouseUp(pos, button)) {
            return true;
        }
    }
    return false;
}

bool UIManager::handleMouseScroll(float2 pos, float delta) {
    // Scroll goes to the topmost root under the cursor only — no falling
    // through to a scroll view in a panel underneath.
    for (auto it = m_rootWidgets.rbegin(); it != m_rootWidgets.rend(); ++it) {
        Widget* root = it->get();
        if (!root->isVisible()) continue;
        if (!root->containsPoint(pos)) continue;
        return root->onMouseScroll(pos, delta);
    }
    return false;
}

bool UIManager::handleKeyDown(int key, int mods) {
    // Handle global shortcuts
    if (key == GLFW_KEY_H && mods == 0) {
        toggleScenePanel();
        return true;
    }

    if ((key == GLFW_KEY_L || key == GLFW_KEY_D) && mods == 0) {
        toggleTheme();
        return true;
    }
    if (key == GLFW_KEY_G && mods == 0) {
        toggleGridDebugPanel();
        return true;
    }

    // Forward to focused widgets
    for (auto& widget : m_rootWidgets) {
        if (widget->isVisible() && widget->onKeyDown(key, mods)) {
            return true;
        }
    }

    return false;
}

bool UIManager::handleKeyUp(int key, int mods) {
    for (auto& widget : m_rootWidgets) {
        if (widget->isVisible() && widget->onKeyUp(key, mods)) {
            return true;
        }
    }
    return false;
}

void UIManager::createDefaultUI() {
    // Order: scene panel, property panel, top bar (last = on top for input)
    auto scenePanel = std::make_unique<Panel>();
    m_scenePanel = scenePanel.get();
    m_scenePanel->setPosition(10.0f, 50.0f);
    m_scenePanel->setSize(280.0f, 400.0f);
    m_scenePanel->setShowHeader(true);
    m_scenePanel->setTitle("Scene Hierarchy");
    m_scenePanel->setCloseable(true);
    m_scenePanel->setDraggable(true);
    m_scenePanel->setTheme(m_theme);
    m_scenePanel->setOnClose([this]() {
        // Keep the top-bar toggle in sync when closed via the X button
        if (m_sceneToggleBtn) m_sceneToggleBtn->setToggled(false);
    });

    auto scrollView = std::make_unique<ScrollView>();
    scrollView->setPosition(0.0f, 30.0f);
    scrollView->setSize(280.0f, 370.0f);
    scrollView->setTheme(m_theme);
    m_sceneScrollView = scrollView.get();
    m_scenePanel->addChild(std::move(scrollView));
    m_rootWidgets.push_back(std::move(scenePanel));

    auto propertyPanel = std::make_unique<PropertyPanel>();
    m_propertyPanel = propertyPanel.get();
    m_propertyPanel->setPosition(static_cast<float>(m_screenWidth) - 310.0f, 50.0f);
    m_propertyPanel->setSize(300.0f, 500.0f);
    m_propertyPanel->setTheme(m_theme);
    m_propertyPanel->setDraggable(true);
    m_propertyPanel->setVisible(false);
    m_propertyPanel->setOnLightEdit([this](SceneNodeType type, uint32_t index, const LightInfo& info) {
        if (m_lightEditCallback) m_lightEditCallback(type, index, info);
    });
    m_rootWidgets.push_back(std::move(propertyPanel));

    auto topBar = std::make_unique<Panel>();
    m_topBar = topBar.get();
    m_topBar->setPosition(0.0f, 0.0f);
    m_topBar->setSize(static_cast<float>(m_screenWidth), 40.0f);
    m_topBar->setShowHeader(false);
    m_topBar->setShowBorder(false);
    m_topBar->setTheme(m_theme);

    auto titleLabel = std::make_unique<Label>("SpectraPBR");
    titleLabel->setPosition(12.0f, 10.0f);
    titleLabel->setSize(100.0f, 20.0f);
    titleLabel->setTextScale(0.7f);
    m_topBar->addChild(std::move(titleLabel));

    auto sceneBtn = std::make_unique<Button>("Scene");
    sceneBtn->setPosition(130.0f, 6.0f);
    sceneBtn->setSize(60.0f, 28.0f);
    sceneBtn->setToggleMode(true);
    sceneBtn->setToggled(true);
    m_sceneToggleBtn = sceneBtn.get();
    sceneBtn->setOnClick([this]() { toggleScenePanel(); });
    m_topBar->addChild(std::move(sceneBtn));

    auto themeBtn = std::make_unique<Button>("Theme");
    themeBtn->setPosition(200.0f, 6.0f);
    themeBtn->setSize(60.0f, 28.0f);
    themeBtn->setOnClick([this]() { toggleTheme(); });
    m_topBar->addChild(std::move(themeBtn));

    m_rootWidgets.push_back(std::move(topBar));

    updateRootDepths();
}

void UIManager::buildSceneTree(const SceneManager* sceneManager) {
    if (!sceneManager || !m_sceneScrollView) return;

    clearSceneTree();

    const auto& instances = sceneManager->getInstances();
    float yOffset = 4.0f;  // Small padding from top

    for (size_t i = 0; i < instances.size(); i++) {
        auto node = std::make_unique<TreeNode>("Instance " + std::to_string(i));
        node->setPosition(4.0f, yOffset);
        node->setSize(260.0f, TreeNode::ROW_HEIGHT);  // Narrower to leave room for scrollbar
        node->setUserData(static_cast<uint32_t>(i));
        node->setNodeType(SceneNodeType::Instance);
        node->setOnSelect([this](TreeNode* n) {
            onSceneNodeSelected(n);
        });
        node->setOnDoubleClick([this](TreeNode* n) {
            onSceneNodeDoubleClicked(n);
        });

        m_sceneNodes.push_back(node.get());
        m_sceneScrollView->addChild(std::move(node));

        yOffset += TreeNode::ROW_HEIGHT;
    }

    // Set content height for scroll view
    m_sceneScrollView->setContentHeight(yOffset + 4.0f);

    std::cout << "[UIManager] Built scene tree with " << instances.size() << " instances\n";
}

void UIManager::buildHierarchicalSceneTree() {
    if (!m_hierarchy || !m_sceneScrollView) return;

    clearSceneTree();

    float yOffset = 4.0f;

    // Build tree starting from root
    uint32_t rootIndex = m_hierarchy->getRootIndex();
    if (rootIndex != UINT32_MAX) {
        buildTreeNodeRecursive(rootIndex, 0, yOffset);
    }

    // Set content height for scroll view
    m_sceneScrollView->setContentHeight(yOffset + 4.0f);

    std::cout << "[UIManager] Built hierarchical scene tree with " << m_sceneNodes.size() << " nodes\n";
}

void UIManager::buildTreeNodeRecursive(uint32_t nodeIndex, int indentLevel, float& yOffset) {
    const HierarchyNode* hierNode = m_hierarchy->getNode(nodeIndex);
    if (!hierNode) return;

    // Create tree node widget
    auto treeNode = std::make_unique<TreeNode>(hierNode->name);
    treeNode->setPosition(4.0f, yOffset);
    treeNode->setSize(260.0f, TreeNode::ROW_HEIGHT);
    treeNode->setIndentLevel(indentLevel);
    treeNode->setHasChildren(!hierNode->childIndices.empty());
    treeNode->setExpanded(hierNode->expanded);
    treeNode->setUserData(hierNode->dataIndex);
    treeNode->setNodeType(hierNode->type);
    treeNode->setNodeIndex(nodeIndex);

    // Set callbacks
    treeNode->setOnSelect([this](TreeNode* n) {
        onSceneNodeSelected(n);
    });
    treeNode->setOnDoubleClick([this](TreeNode* n) {
        onSceneNodeDoubleClicked(n);
    });
    treeNode->setOnExpand([this](TreeNode* n, bool expanded) {
        onSceneNodeExpanded(n, expanded);
    });

    m_sceneNodes.push_back(treeNode.get());
    m_sceneScrollView->addChild(std::move(treeNode));
    yOffset += TreeNode::ROW_HEIGHT;

    // Build children if expanded
    if (hierNode->expanded) {
        for (uint32_t childIndex : hierNode->childIndices) {
            buildTreeNodeRecursive(childIndex, indentLevel + 1, yOffset);
        }
    }
}

void UIManager::clearSceneTree() {
    m_sceneNodes.clear();
    if (m_sceneScrollView) {
        m_sceneScrollView->clearChildren();
        m_sceneScrollView->setContentHeight(0.0f);
        m_sceneScrollView->setScrollOffset(0.0f);
    }
}

void UIManager::setSelectedInstanceId(uint32_t id) {
    if (m_selectedInstanceId == id) return;

    m_selectedInstanceId = id;

    // Update tree selection state
    for (auto* node : m_sceneNodes) {
        node->setSelected(node->getUserData() == id);
    }

    // Clear preview textures when no selection
    if (id == UINT32_MAX) {
        m_previewTextures.clear();
    }

    // Fire callback
    if (m_selectionCallback) {
        m_selectionCallback(id);
    }
}

void UIManager::toggleScenePanel() {
    if (m_scenePanel) {
        m_scenePanel->setVisible(!m_scenePanel->isVisible());
        if (m_sceneToggleBtn) {
            m_sceneToggleBtn->setToggled(m_scenePanel->isVisible());
        }
        m_geometryDirty = true;  // Ensure geometry regenerates on visibility change
    }
}

bool UIManager::isScenePanelVisible() const {
    return m_scenePanel && m_scenePanel->isVisible();
}

void UIManager::addRootWidget(std::unique_ptr<Widget> widget) {
    widget->setTheme(m_theme);
    if (m_rootWidgets.size() >= 1) {
        m_rootWidgets.insert(m_rootWidgets.end() - 1, std::move(widget));
    } else {
        m_rootWidgets.push_back(std::move(widget));
    }
    updateRootDepths();
}

void UIManager::onSceneNodeSelected(TreeNode* node) {
    // Deselect all other nodes
    for (auto* n : m_sceneNodes) {
        if (n != node) {
            n->setSelected(false);
        }
    }

    SceneNodeType nodeType = node->getNodeType();

    // Only fire selection callback for instances
    if (nodeType == SceneNodeType::Instance) {
        uint32_t newId = node->getUserData();
        if (m_selectedInstanceId != newId) {
            m_selectedInstanceId = newId;
            if (m_selectionCallback) {
                m_selectionCallback(newId);
            }
        }
    }
}

void UIManager::onSceneNodeDoubleClicked(TreeNode* node) {
    SceneNodeType nodeType = node->getNodeType();

    // Show property panel based on node type
    switch (nodeType) {
        case SceneNodeType::Instance: {
            uint32_t instanceId = node->getUserData();

            // Use callback to get instance info if available
            if (m_instanceInfoRequestCallback && m_propertyPanel) {
                InstanceInfo info = m_instanceInfoRequestCallback(instanceId);
                m_propertyPanel->showInstanceProperties(info);
                m_propertyPanel->setVisible(true);
            } else if (m_propertyPanel) {
                // Fallback: basic info with safe defaults
                InstanceInfo info = {};
                info.instanceId = instanceId;
                info.modelName = "Model";
                info.meshName = node->getLabel();
                info.baseColor = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
                info.metallic = 0.0f;
                info.roughness = 0.5f;
                info.emissive = make_float3(0.0f, 0.0f, 0.0f);
                m_propertyPanel->showInstanceProperties(info);
                m_propertyPanel->setVisible(true);
            }
            break;
        }

        case SceneNodeType::PointLight:
        case SceneNodeType::DirectionalLight:
        case SceneNodeType::AreaLight: {
            uint32_t lightIndex = node->getUserData();

            // Use callback to get light info
            if (m_lightInfoRequestCallback && m_propertyPanel) {
                LightInfo info = m_lightInfoRequestCallback(nodeType, lightIndex);
                m_propertyPanel->showLightProperties(info);
                m_propertyPanel->setVisible(true);
            } else if (m_propertyPanel) {
                m_propertyPanel->setVisible(true);
            }
            break;
        }

        case SceneNodeType::Model:
        case SceneNodeType::Root:
        case SceneNodeType::LightsGroup:
        case SceneNodeType::Mesh:
            // For parent nodes, toggle expansion
            node->toggleExpanded();
            break;
    }
}

void UIManager::onSceneNodeExpanded(TreeNode* node, bool expanded) {
    // Update hierarchy state
    if (m_hierarchy) {
        uint32_t nodeIndex = node->getNodeIndex();
        m_hierarchy->setExpanded(nodeIndex, expanded);
    }

    // Rebuild tree to reflect changes
    buildHierarchicalSceneTree();
}

void UIManager::showPropertyPanel(bool show) {
    if (m_propertyPanel) {
        m_propertyPanel->setVisible(show);
        m_geometryDirty = true;  // Ensure geometry regenerates on visibility change
    }
}

bool UIManager::isPropertyPanelVisible() const {
    return m_propertyPanel && m_propertyPanel->isVisible();
}

void UIManager::togglePropertyPanel() {
    if (m_propertyPanel) {
        m_propertyPanel->setVisible(!m_propertyPanel->isVisible());
        m_geometryDirty = true;  // Ensure geometry regenerates on visibility change
    }
}

void UIManager::addPathGuideGridDebugPanel(
    uint32_t numLevels, uint32_t totalCells, uint32_t entryStride,
    uint32_t baseResolution, float perLevelScale,
    const float boundsMin[3], const float boundsMax[3],
    bool initialVisualize,
    std::function<void(bool)> onVisualize,
    std::function<void(uint32_t)> onLevel,
    std::function<void()> onBuild,
    std::function<void(bool)> onEnableGuiding,
    std::function<void()> onPause,
    std::function<void()> onBuildAndStep)
{
    if (m_gridDebugPanel) return;

    auto gridPanel = std::make_unique<Panel>();
    m_gridDebugPanel = gridPanel.get();
    m_gridDebugPanel->setPosition(10.0f, 460.0f);
    m_gridDebugPanel->setSize(400.0f, 620.0f);
    m_gridDebugPanel->setShowHeader(true);
    m_gridDebugPanel->setTitle("Path Guide Grid (Sparse)");
    m_gridDebugPanel->setCloseable(true);
    m_gridDebugPanel->setDraggable(true);
    m_gridDebugPanel->setTheme(m_theme);
    m_gridDebugPanel->setVisible(false);

    float y = 32.0f;
    char buf[256];

    snprintf(buf, sizeof(buf), "Bounds: (%.1f,%.1f,%.1f) .. (%.1f,%.1f,%.1f)",
             boundsMin[0], boundsMin[1], boundsMin[2],
             boundsMax[0], boundsMax[1], boundsMax[2]);
    auto boundsLabel = std::make_unique<Label>(buf);
    boundsLabel->setPosition(8.0f, y);
    boundsLabel->setSize(384.0f, 18.0f);
    boundsLabel->setTextScale(0.55f);
    boundsLabel->setSecondary(true);
    m_gridDebugPanel->addChild(std::move(boundsLabel));
    y += 24.0f;

    snprintf(buf, sizeof(buf), "Levels: %u  Sparse cells: %u  Stride: %u", numLevels, totalCells, entryStride);
    auto metaLabel = std::make_unique<Label>(buf);
    metaLabel->setPosition(8.0f, y);
    metaLabel->setSize(384.0f, 18.0f);
    metaLabel->setTextScale(0.55f);
    metaLabel->setSecondary(true);
    m_pathGuideMetaLabel = metaLabel.get();
    m_gridDebugPanel->addChild(std::move(metaLabel));
    y += 26.0f;

    auto levelHeader = std::make_unique<Label>("Level | Res  | Cells");
    levelHeader->setPosition(8.0f, y);
    levelHeader->setSize(384.0f, 16.0f);
    levelHeader->setTextScale(0.5f);
    levelHeader->setSecondary(true);
    m_gridDebugPanel->addChild(std::move(levelHeader));
    y += 20.0f;

    for (uint32_t l = 0; l < numLevels && l < 16u; l++) {
        float res = static_cast<float>(baseResolution) * std::pow(perLevelScale, static_cast<float>(l));
        uint32_t resU = static_cast<uint32_t>(res);
        uint64_t cells = static_cast<uint64_t>(resU) * resU * resU;
        snprintf(buf, sizeof(buf), "  %2u   | %4u  | %llu", l, resU, static_cast<unsigned long long>(cells));
        auto row = std::make_unique<Label>(buf);
        row->setPosition(8.0f, y);
        row->setSize(384.0f, 16.0f);
        row->setTextScale(0.5f);
        row->setSecondary(true);
        m_gridDebugPanel->addChild(std::move(row));
        y += 18.0f;
    }
    y += 8.0f;

    Button* showBtn = nullptr;
    auto showButton = std::make_unique<Button>("Show in viewport");
    showButton->setPosition(8.0f, y);
    showButton->setSize(180.0f, 28.0f);
    showButton->setToggleMode(true);
    showButton->setToggled(initialVisualize);
    showBtn = showButton.get();
    showButton->setOnClick([onVisualize, showBtn]() {
        if (onVisualize) onVisualize(showBtn->isToggled());
    });
    m_gridDebugPanel->addChild(std::move(showButton));
    y += 36.0f;

    auto levelSlider = std::make_unique<Slider>();
    levelSlider->setPosition(8.0f, y);
    levelSlider->setSize(384.0f, 24.0f);
    levelSlider->setLabel("Level");
    levelSlider->setLabelWidth(50.0f);
    levelSlider->setRange(0.0f, numLevels > 1 ? static_cast<float>(numLevels - 1) : 0.0f);
    levelSlider->setValue(0.0f);
    levelSlider->setValueFormat("%.0f");
    levelSlider->setOnValueChanged([onLevel](Slider* s, float v) {
        if (onLevel) onLevel(static_cast<uint32_t>(v + 0.5f));
    });
    m_gridDebugPanel->addChild(std::move(levelSlider));
    y += 36.0f;

    // Path guiding automation controls
    if (onEnableGuiding) {
        auto enableBtn = std::make_unique<Button>("Enable Guiding");
        enableBtn->setPosition(8.0f, y);
        enableBtn->setSize(180.0f, 28.0f);
        enableBtn->setToggleMode(true);
        enableBtn->setToggled(false);
        Button* enableBtnPtr = enableBtn.get();
        enableBtn->setOnClick([onEnableGuiding, enableBtnPtr]() {
            if (onEnableGuiding) onEnableGuiding(enableBtnPtr->isToggled());
        });
        m_gridDebugPanel->addChild(std::move(enableBtn));
        y += 36.0f;
    }

    if (onPause) {
        auto pauseBtn = std::make_unique<Button>("Pause");
        pauseBtn->setPosition(8.0f, y);
        pauseBtn->setSize(120.0f, 28.0f);
        pauseBtn->setOnClick([onPause]() { if (onPause) onPause(); });
        m_gridDebugPanel->addChild(std::move(pauseBtn));
    }

    if (onBuildAndStep) {
        auto stepBtn = std::make_unique<Button>("Build & Step");
        stepBtn->setPosition(140.0f, y);
        stepBtn->setSize(140.0f, 28.0f);
        stepBtn->setOnClick([onBuildAndStep]() { if (onBuildAndStep) onBuildAndStep(); });
        m_gridDebugPanel->addChild(std::move(stepBtn));
    }

    if (onPause || onBuildAndStep) {
        y += 36.0f;
    }

    // Automation status label
    auto statusLabel = std::make_unique<Label>("Disabled | 0/60 frames | 0 builds");
    statusLabel->setPosition(8.0f, y);
    statusLabel->setSize(384.0f, 18.0f);
    statusLabel->setTextScale(0.55f);
    statusLabel->setSecondary(true);
    m_pathGuideStatusLabel = statusLabel.get();
    m_gridDebugPanel->addChild(std::move(statusLabel));
    y += 24.0f;

    if (onVisualize) onVisualize(initialVisualize);
    if (onLevel) onLevel(0);

    // Insert before top bar so it receives input in correct order
    m_rootWidgets.insert(m_rootWidgets.end() - 1, std::move(gridPanel));
    updateRootDepths();
}

void UIManager::toggleGridDebugPanel() {
    if (m_gridDebugPanel) {
        m_gridDebugPanel->setVisible(!m_gridDebugPanel->isVisible());
        m_geometryDirty = true;
    }
}

bool UIManager::isGridDebugPanelVisible() const {
    return m_gridDebugPanel && m_gridDebugPanel->isVisible();
}

void UIManager::updatePathGuideGridStats(uint32_t numLevels, uint32_t totalCells, uint32_t entryStride) {
    if (!m_pathGuideMetaLabel) return;
    char buf[256];
    snprintf(buf, sizeof(buf), "Levels: %u  Sparse cells: %u  Stride: %u", numLevels, totalCells, entryStride);
    m_pathGuideMetaLabel->setText(buf);
    m_geometryDirty = true;
}

void UIManager::setOnLightEdit(LightEditCallback callback) {
    m_lightEditCallback = callback;
    if (m_propertyPanel) {
        m_propertyPanel->setOnLightEdit(callback);
    }
}

void UIManager::setOnMaterialEdit(MaterialEditCallback callback) {
    m_materialEditCallback = callback;
    if (m_propertyPanel) {
        m_propertyPanel->setOnMaterialEdit(callback);
    }
}

void UIManager::clearTreeSelection(Widget* widget, TreeNode* except) {
    if (auto* node = dynamic_cast<TreeNode*>(widget)) {
        if (node != except) {
            node->setSelected(false);
        }
    }
    for (auto& child : widget->getChildren()) {
        clearTreeSelection(child.get(), except);
    }
}

void UIManager::updatePathGuideAutomationStatus(const char* modeStr, uint32_t framesSinceBuild, uint32_t totalBuilds) {
    if (!m_pathGuideStatusLabel) return;
    char buf[256];
    snprintf(buf, sizeof(buf), "%s | %u/60 frames | %u builds", modeStr, framesSinceBuild, totalBuilds);
    m_pathGuideStatusLabel->setText(buf);
    m_geometryDirty = true;
}

void UIManager::addCellInspectorPanel() {
    if (m_cellInspectorPanel) return;

    auto panel = std::make_unique<Panel>();
    m_cellInspectorPanel = panel.get();
    m_cellInspectorPanel->setPosition(static_cast<float>(m_screenWidth) - 320.0f, static_cast<float>(m_screenHeight) - 280.0f);
    m_cellInspectorPanel->setSize(310.0f, 260.0f);
    m_cellInspectorPanel->setShowHeader(true);
    m_cellInspectorPanel->setTitle("Cell Inspector");
    m_cellInspectorPanel->setCloseable(true);
    m_cellInspectorPanel->setDraggable(true);
    m_cellInspectorPanel->setTheme(m_theme);
    m_cellInspectorPanel->setVisible(false);

    float y = 32.0f;
    auto makeLabel = [&](const char* text) -> Label* {
        auto lbl = std::make_unique<Label>(text);
        lbl->setPosition(8.0f, y);
        lbl->setSize(294.0f, 18.0f);
        lbl->setTextScale(0.5f);
        lbl->setSecondary(true);
        Label* ptr = lbl.get();
        m_cellInspectorPanel->addChild(std::move(lbl));
        y += 22.0f;
        return ptr;
    };

    m_cellPosLabel = makeLabel("Pos: --");
    m_cellLevelLabel = makeLabel("Level: --  Cell: --");
    m_cellAABBLabel = makeLabel("AABB: --");
    y += 4.0f;
    m_cellLobe0Label = makeLabel("Lobe 0: --");
    m_cellLobe1Label = makeLabel("Lobe 1: --");
    y += 4.0f;
    m_cellStatsLabel = makeLabel("Samples: --  Variance: --");
    m_cellStatusLabel = makeLabel("Status: --");

    m_rootWidgets.insert(m_rootWidgets.end() - 1, std::move(panel));
    updateRootDepths();
}

void UIManager::updateCellInspectorData(const InspectedCellInfo& info) {
    if (!m_cellInspectorPanel) return;

    char buf[256];

    if (!info.valid) {
        if (m_cellPosLabel) m_cellPosLabel->setText("No cell at hit position");
        if (m_cellLevelLabel) m_cellLevelLabel->setText("");
        if (m_cellAABBLabel) m_cellAABBLabel->setText("");
        if (m_cellLobe0Label) m_cellLobe0Label->setText("");
        if (m_cellLobe1Label) m_cellLobe1Label->setText("");
        if (m_cellStatsLabel) m_cellStatsLabel->setText("");
        if (m_cellStatusLabel) m_cellStatusLabel->setText("");
        m_cellInspectorPanel->setVisible(true);
        m_geometryDirty = true;
        return;
    }

    snprintf(buf, sizeof(buf), "Pos: (%.2f, %.2f, %.2f)", info.worldPos[0], info.worldPos[1], info.worldPos[2]);
    if (m_cellPosLabel) m_cellPosLabel->setText(buf);

    snprintf(buf, sizeof(buf), "Level %u  Cell (%u, %u, %u)", info.level, info.ix, info.iy, info.iz);
    if (m_cellLevelLabel) m_cellLevelLabel->setText(buf);

    snprintf(buf, sizeof(buf), "AABB: (%.2f,%.2f,%.2f)..(%.2f,%.2f,%.2f)",
             info.cellAABBMin[0], info.cellAABBMin[1], info.cellAABBMin[2],
             info.cellAABBMax[0], info.cellAABBMax[1], info.cellAABBMax[2]);
    if (m_cellAABBLabel) m_cellAABBLabel->setText(buf);

    snprintf(buf, sizeof(buf), "Lobe 0: t=%.2f p=%.2f k=%.1f", info.theta0, info.phi0, info.kappa0);
    if (m_cellLobe0Label) m_cellLobe0Label->setText(buf);

    snprintf(buf, sizeof(buf), "Lobe 1: t=%.2f p=%.2f k=%.1f", info.theta1, info.phi1, info.kappa1);
    if (m_cellLobe1Label) m_cellLobe1Label->setText(buf);

    snprintf(buf, sizeof(buf), "Samples: %.0f  Var: %.3f  Mix: %.0f%%/%.0f%%",
             info.sumW, info.variance, info.pi0 * 100.0f, (1.0f - info.pi0) * 100.0f);
    if (m_cellStatsLabel) m_cellStatsLabel->setText(buf);

    const char* status = "stable";
    if (info.wouldSubdivide) status = "subdivide";
    else if (info.wouldCoarsen) status = "coarsen";
    snprintf(buf, sizeof(buf), "Status: %s  Last hit: frame %.0f", status, info.lastFrame);
    if (m_cellStatusLabel) m_cellStatusLabel->setText(buf);

    m_cellInspectorPanel->setVisible(true);
    m_geometryDirty = true;
}

void UIManager::showCellInspectorPanel(bool show) {
    if (m_cellInspectorPanel) {
        m_cellInspectorPanel->setVisible(show);
        m_geometryDirty = true;
    }
}

} // namespace ui
} // namespace spectra
