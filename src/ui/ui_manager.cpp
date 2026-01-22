#include "ui_manager.h"
#include "scene_manager.h"
#include <iostream>
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

    createDefaultUI();

    std::cout << "[UIManager] Initialized with screen size " << screenWidth << "x" << screenHeight << "\n";
    return true;
}

void UIManager::shutdown() {
    m_topBar.reset();
    m_scenePanel.reset();
    m_rootWidgets.clear();
    m_sceneNodes.clear();
    m_quads.clear();
}

void UIManager::setTheme(const Theme* theme) {
    m_theme = theme;
    if (m_topBar) m_topBar->setTheme(theme);
    if (m_scenePanel) m_scenePanel->setTheme(theme);
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
    m_screenWidth = width;
    m_screenHeight = height;

    // Update top bar width
    if (m_topBar) {
        m_topBar->setSize(static_cast<float>(width), 40.0f);
    }
}

void UIManager::update(float deltaTime) {
    if (m_topBar && m_topBar->isVisible()) {
        m_topBar->update(deltaTime);
    }
    if (m_scenePanel && m_scenePanel->isVisible()) {
        m_scenePanel->update(deltaTime);
    }
    for (auto& widget : m_rootWidgets) {
        if (widget->isVisible()) {
            widget->update(deltaTime);
        }
    }
}

void UIManager::collectGeometry() {
    m_quads.clear();

    // Collect geometry from all visible widgets
    if (m_scenePanel && m_scenePanel->isVisible()) {
        m_scenePanel->collectGeometry(m_quads, &m_textLayout);
    }
    if (m_topBar && m_topBar->isVisible()) {
        m_topBar->collectGeometry(m_quads, &m_textLayout);
    }
    for (auto& widget : m_rootWidgets) {
        if (widget->isVisible()) {
            widget->collectGeometry(m_quads, &m_textLayout);
        }
    }

    // Sort by depth (back to front)
    std::sort(m_quads.begin(), m_quads.end(),
        [](const UIQuad& a, const UIQuad& b) { return a.depth < b.depth; });
}

bool UIManager::handleMouseMove(float2 pos) {
    bool consumed = false;

    // Check widgets in reverse order (front to back)
    for (auto it = m_rootWidgets.rbegin(); it != m_rootWidgets.rend(); ++it) {
        if ((*it)->isVisible() && (*it)->onMouseMove(pos)) {
            consumed = true;
            break;
        }
    }

    if (!consumed && m_scenePanel && m_scenePanel->isVisible()) {
        consumed = m_scenePanel->onMouseMove(pos);
    }

    if (!consumed && m_topBar && m_topBar->isVisible()) {
        consumed = m_topBar->onMouseMove(pos);
    }

    return consumed;
}

bool UIManager::handleMouseDown(float2 pos, int button) {
    bool consumed = false;

    for (auto it = m_rootWidgets.rbegin(); it != m_rootWidgets.rend(); ++it) {
        if ((*it)->isVisible() && (*it)->onMouseDown(pos, button)) {
            consumed = true;
            break;
        }
    }

    if (!consumed && m_scenePanel && m_scenePanel->isVisible()) {
        consumed = m_scenePanel->onMouseDown(pos, button);
    }

    if (!consumed && m_topBar && m_topBar->isVisible()) {
        consumed = m_topBar->onMouseDown(pos, button);
    }

    return consumed;
}

bool UIManager::handleMouseUp(float2 pos, int button) {
    bool consumed = false;

    for (auto it = m_rootWidgets.rbegin(); it != m_rootWidgets.rend(); ++it) {
        if ((*it)->isVisible() && (*it)->onMouseUp(pos, button)) {
            consumed = true;
            break;
        }
    }

    if (!consumed && m_scenePanel && m_scenePanel->isVisible()) {
        consumed = m_scenePanel->onMouseUp(pos, button);
    }

    if (!consumed && m_topBar && m_topBar->isVisible()) {
        consumed = m_topBar->onMouseUp(pos, button);
    }

    return consumed;
}

bool UIManager::handleMouseScroll(float2 pos, float delta) {
    bool consumed = false;

    for (auto it = m_rootWidgets.rbegin(); it != m_rootWidgets.rend(); ++it) {
        if ((*it)->isVisible() && (*it)->onMouseScroll(pos, delta)) {
            consumed = true;
            break;
        }
    }

    if (!consumed && m_scenePanel && m_scenePanel->isVisible()) {
        consumed = m_scenePanel->onMouseScroll(pos, delta);
    }

    return consumed;
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
    // Create top bar
    m_topBar = std::make_unique<Panel>();
    m_topBar->setPosition(0.0f, 0.0f);
    m_topBar->setSize(static_cast<float>(m_screenWidth), 40.0f);
    m_topBar->setShowHeader(false);
    m_topBar->setShowBorder(false);
    m_topBar->setDepth(100.0f);  // High depth to be on top
    m_topBar->setTheme(m_theme);

    // App title
    auto titleLabel = std::make_unique<Label>("SpectraPBR");
    titleLabel->setPosition(12.0f, 10.0f);
    titleLabel->setSize(100.0f, 20.0f);
    titleLabel->setTextScale(0.7f);
    m_topBar->addChild(std::move(titleLabel));

    // Scene toggle button
    auto sceneBtn = std::make_unique<Button>("Scene");
    sceneBtn->setPosition(130.0f, 6.0f);
    sceneBtn->setSize(60.0f, 28.0f);
    sceneBtn->setToggleMode(true);
    sceneBtn->setToggled(true);
    sceneBtn->setOnClick([this]() {
        toggleScenePanel();
    });
    m_topBar->addChild(std::move(sceneBtn));

    // Theme toggle button
    auto themeBtn = std::make_unique<Button>("Theme");
    themeBtn->setPosition(200.0f, 6.0f);
    themeBtn->setSize(60.0f, 28.0f);
    themeBtn->setOnClick([this]() {
        toggleTheme();
    });
    m_topBar->addChild(std::move(themeBtn));

    // Create scene hierarchy panel
    m_scenePanel = std::make_unique<Panel>();
    m_scenePanel->setPosition(10.0f, 50.0f);
    m_scenePanel->setSize(280.0f, 400.0f);
    m_scenePanel->setShowHeader(true);
    m_scenePanel->setTitle("Scene Hierarchy");
    m_scenePanel->setCloseable(true);
    m_scenePanel->setDraggable(true);
    m_scenePanel->setDepth(50.0f);
    m_scenePanel->setTheme(m_theme);
    m_scenePanel->setOnClose([this]() {
        // Update the toggle button state
        // (In a real implementation, we'd have a reference to the button)
    });
    
    // Create scroll view for scene tree content
    auto scrollView = std::make_unique<ScrollView>();
    scrollView->setPosition(0.0f, 30.0f);  // Below header
    scrollView->setSize(280.0f, 370.0f);   // Fill remaining panel space
    scrollView->setTheme(m_theme);
    m_sceneScrollView = scrollView.get();
    m_scenePanel->addChild(std::move(scrollView));
    
    std::cout << "[UIManager] Scene panel created, visible: " << m_scenePanel->isVisible() << "\n";
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
        node->setOnSelect([this](TreeNode* n) {
            onSceneNodeSelected(n);
        });

        m_sceneNodes.push_back(node.get());
        m_sceneScrollView->addChild(std::move(node));

        yOffset += TreeNode::ROW_HEIGHT;
    }

    // Set content height for scroll view
    m_sceneScrollView->setContentHeight(yOffset + 4.0f);

    std::cout << "[UIManager] Built scene tree with " << instances.size() << " instances\n";
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

    // Fire callback
    if (m_selectionCallback) {
        m_selectionCallback(id);
    }
}

void UIManager::toggleScenePanel() {
    if (m_scenePanel) {
        bool wasVisible = m_scenePanel->isVisible();
        m_scenePanel->setVisible(!wasVisible);
        std::cout << "[UIManager] toggleScenePanel: " << wasVisible << " -> " << m_scenePanel->isVisible() << "\n";
    } else {
        std::cout << "[UIManager] toggleScenePanel: m_scenePanel is null!\n";
    }
}

bool UIManager::isScenePanelVisible() const {
    return m_scenePanel && m_scenePanel->isVisible();
}

void UIManager::addRootWidget(std::unique_ptr<Widget> widget) {
    widget->setTheme(m_theme);
    m_rootWidgets.push_back(std::move(widget));
}

void UIManager::onSceneNodeSelected(TreeNode* node) {
    // Deselect all other nodes
    for (auto* n : m_sceneNodes) {
        if (n != node) {
            n->setSelected(false);
        }
    }

    // Update selected ID
    uint32_t newId = node->getUserData();
    if (m_selectedInstanceId != newId) {
        m_selectedInstanceId = newId;
        if (m_selectionCallback) {
            m_selectionCallback(newId);
        }
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

} // namespace ui
} // namespace spectra
