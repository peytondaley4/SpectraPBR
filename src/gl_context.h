#pragma once

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <filesystem>
#include <functional>
#include <cstdint>

namespace spectra {

// Resolution presets for debug testing
struct Resolution {
    uint32_t width;
    uint32_t height;
    const char* name;
};

constexpr Resolution RESOLUTION_720P  = { 1280, 720, "720p" };
constexpr Resolution RESOLUTION_1080P = { 1920, 1080, "1080p" };
constexpr Resolution RESOLUTION_1440P = { 2560, 1440, "1440p" };
constexpr Resolution RESOLUTION_4K    = { 3840, 2160, "4K" };

// Callback for resize events (width, height)
using ResizeCallback = std::function<void(uint32_t, uint32_t)>;
// Callback before resize (to unregister CUDA resources before buffers are invalidated)
using PreResizeCallback = std::function<void()>;

class GLContext {
public:
    GLContext() = default;
    ~GLContext();

    // Non-copyable
    GLContext(const GLContext&) = delete;
    GLContext& operator=(const GLContext&) = delete;

    // Initialize GLFW and create OpenGL context
    // Returns false on failure
    bool init(uint32_t width, uint32_t height, const char* title);

    // Shutdown and cleanup
    void shutdown();

    // Check if window should close
    bool shouldClose() const;

    // Poll events and handle input
    void pollEvents();

    // Swap buffers
    void swapBuffers();

    // Create display resources (texture, PBO, shader)
    // Call after init() and before render loop
    // shaderDir: directory containing display.vert and display.frag
    bool createDisplayResources(const std::filesystem::path& shaderDir);

    // Update display texture from PBO
    // Call after CUDA has written to the mapped PBO
    void updateTextureFromPBO();

    // Update UI texture from UI PBO
    // Call after CUDA has written to the mapped UI PBO
    void updateUITextureFromPBO();

    // Render fullscreen quad with display texture
    // If UI is enabled, composites UI on top of scene
    void renderFullscreenQuad();

    // Enable/disable UI compositing
    void setUIEnabled(bool enabled) { m_uiEnabled = enabled; }
    bool isUIEnabled() const { return m_uiEnabled; }

    // Get PBO for CUDA interop registration
    GLuint getPBO() const { return m_pbo; }

    // Get UI PBO for CUDA interop registration
    GLuint getUIPBO() const { return m_uiPbo; }

    // Get current dimensions
    uint32_t getWidth() const { return m_width; }
    uint32_t getHeight() const { return m_height; }

    // Get buffer size in bytes (RGBA32F)
    size_t getBufferSize() const { return m_width * m_height * 4 * sizeof(float); }

    // Set resize callback (called AFTER buffers are recreated)
    void setResizeCallback(ResizeCallback callback) { m_resizeCallback = callback; }

    // Set pre-resize callback (called BEFORE buffers are recreated - use to unregister CUDA resources)
    void setPreResizeCallback(PreResizeCallback callback) { m_preResizeCallback = callback; }

    // VSync control
    void setVSync(bool enabled);
    bool isVSyncEnabled() const { return m_vsyncEnabled; }

    // Fullscreen control
    void toggleFullscreen();
    bool isFullscreen() const { return m_fullscreen; }

    // Set resolution (triggers resize callback)
    void setResolution(uint32_t width, uint32_t height);

    // Get GLFW window (for input handling in main)
    GLFWwindow* getWindow() const { return m_window; }

private:
    void recreateBuffers();

    static void framebufferSizeCallback(GLFWwindow* window, int width, int height);
    static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);

    GLFWwindow* m_window = nullptr;
    uint32_t m_width = 0;
    uint32_t m_height = 0;

    // Display resources
    GLuint m_displayTexture = 0;
    GLuint m_pbo = 0;
    GLuint m_displayProgram = 0;
    GLuint m_emptyVAO = 0;  // For fullscreen triangle

    // UI resources
    GLuint m_uiTexture = 0;
    GLuint m_uiPbo = 0;
    bool m_uiEnabled = false;
    GLint m_uiTextureLoc = -1;

    // State
    bool m_vsyncEnabled = true;
    bool m_fullscreen = false;
    int m_windowedX = 0, m_windowedY = 0;
    int m_windowedWidth = 0, m_windowedHeight = 0;

    ResizeCallback m_resizeCallback;
    PreResizeCallback m_preResizeCallback;
};

} // namespace spectra
