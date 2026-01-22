#pragma once

#include <cuda_runtime.h>
#include <GLFW/glfw3.h>
#include <functional>

namespace spectra {
namespace ui {

// Forward declaration
class UIManager;

//------------------------------------------------------------------------------
// Input Handler - Manages mouse and keyboard input for UI
//------------------------------------------------------------------------------
class InputHandler {
public:
    InputHandler() = default;
    ~InputHandler() = default;

    // Initialize with GLFW window
    void init(GLFWwindow* window, UIManager* uiManager);

    // Shutdown and unregister callbacks
    void shutdown();

    // Check if UI consumed the last input event
    // Use this to determine if camera should process input
    bool wasMouseConsumed() const { return m_mouseConsumed; }
    bool wasKeyConsumed() const { return m_keyConsumed; }

    // Get current mouse position
    float2 getMousePosition() const { return m_mousePos; }

    // Get mouse button state
    bool isMouseButtonDown(int button) const;

    // Manually update mouse consumed state (e.g., when UI overlay changes)
    void setMouseConsumed(bool consumed) { m_mouseConsumed = consumed; }

    // Key modifiers
    bool isShiftDown() const { return m_mods & GLFW_MOD_SHIFT; }
    bool isControlDown() const { return m_mods & GLFW_MOD_CONTROL; }
    bool isAltDown() const { return m_mods & GLFW_MOD_ALT; }

    // Static callback wrappers (GLFW requires static functions)
    static void mousePositionCallback(GLFWwindow* window, double xpos, double ypos);
    static void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
    static void scrollCallback(GLFWwindow* window, double xoffset, double yoffset);
    static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);

private:
    GLFWwindow* m_window = nullptr;
    UIManager* m_uiManager = nullptr;

    float2 m_mousePos = make_float2(0.0f, 0.0f);
    bool m_mouseButtons[3] = { false, false, false };
    int m_mods = 0;

    bool m_mouseConsumed = false;
    bool m_keyConsumed = false;
};

} // namespace ui
} // namespace spectra
