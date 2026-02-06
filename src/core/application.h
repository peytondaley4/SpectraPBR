#pragma once

#include "frame_timer.h"
#include "shared_types.h"
#include "gl_context.h"
#include "cuda_interop.h"
#include "optix_engine.h"
#include "camera.h"
#include "scene_manager.h"
#include "geometry_manager.h"
#include "texture_manager.h"
#include "material_manager.h"
#include "light_manager.h"
#include "environment_map.h"
#include "text/font_atlas.h"
#include "ui/ui_manager.h"
#include "ui/ui_renderer.h"
#include "ui/input_handler.h"
#include "ui/texture_preview_cache.h"
#include "scene/selection_manager.h"
#include "scene/scene_serializer.h"
#include "scene/scene_hierarchy.h"

#include <filesystem>
#include <string>
#include <memory>

namespace spectra {

struct AppConfig {
    std::filesystem::path modelPath;
    std::filesystem::path hdrPath;
    bool remoteMode = false;
    uint32_t width = 1920;
    uint32_t height = 1080;
};

class Application {
public:
    Application() = default;
    ~Application();

    // Parse command line and configure
    bool parseArgs(int argc, char* argv[]);

    // Initialize all subsystems
    bool init();

    // Run main loop
    void run();

    // Shutdown
    void shutdown();

private:
    // Initialization helpers
    bool initGraphics();
    bool initCuda();
    bool initOptix();
    bool initUI();
    bool loadScene();
    void setupCallbacks();
    void setupDefaultScene();
    void wireUICallbacks();
    void printControls();

    // Utility
    static bool cameraChanged(const CameraParams& a, const CameraParams& b);

    // Input handling
    void processInput();
    void updateCamera(float deltaTime);

    // Rendering
    void renderFrame();

    // GLFW callbacks (static to work with C API)
    static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
    static void cursorPosCallback(GLFWwindow* window, double xpos, double ypos);
    static void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
    static void scrollCallback(GLFWwindow* window, double xoffset, double yoffset);

    // Configuration
    AppConfig m_config;
    std::filesystem::path m_exePath;
    std::filesystem::path m_shaderDir;
    std::filesystem::path m_ptxDir;
    std::filesystem::path m_fontsDir;

    // Core systems
    std::unique_ptr<GLContext> m_glContext;
    std::unique_ptr<CudaInterop> m_cudaInterop;
    std::unique_ptr<OptixEngine> m_optixEngine;
    std::unique_ptr<Camera> m_camera;

    // Scene management
    std::unique_ptr<GeometryManager> m_geometryManager;
    std::unique_ptr<TextureManager> m_textureManager;
    std::unique_ptr<MaterialManager> m_materialManager;
    std::unique_ptr<SceneManager> m_sceneManager;
    std::unique_ptr<LightManager> m_lightManager;
    std::unique_ptr<EnvironmentMap> m_environmentMap;
    std::unique_ptr<SceneHierarchy> m_sceneHierarchy;

    // UI
    std::unique_ptr<text::FontAtlas> m_fontAtlas;
    std::unique_ptr<ui::UIManager> m_uiManager;
    std::unique_ptr<ui::UIRenderer> m_uiRenderer;
    std::unique_ptr<ui::InputHandler> m_inputHandler;
    std::unique_ptr<ui::TexturePreviewCache> m_texturePreviewCache;
    std::unique_ptr<SelectionManager> m_selectionManager;
    std::unique_ptr<SceneSerializer> m_sceneSerializer;

    // Buffers
    float4* m_accumulationBuffer = nullptr;

    // State
    FrameTimer m_timer;
    QualityMode m_qualityMode = QUALITY_BALANCED;
    CameraParams m_prevCameraParams = {};
    bool m_running = false;
    uint64_t m_lastTimingPrint = 0;

    // Input state
    bool m_mouseCaptured = false;
    bool m_remoteMode = false;
    double m_lastMouseX = 0.0;
    double m_lastMouseY = 0.0;
    bool m_firstMouse = true;
    bool m_keyW = false, m_keyS = false, m_keyA = false, m_keyD = false;
    bool m_keyQ = false, m_keyE = false, m_keyShift = false;
    bool m_diagEnabled = false;

    // Triple buffering state
    int m_writeIdx = 0;
    int m_displayIdx = 0;
    int m_framesPipelined = 0;

    // Static instance for callbacks
    static Application* s_instance;
};

} // namespace spectra
