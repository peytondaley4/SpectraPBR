#include "input_handler.h"
#include "ui_manager.h"

namespace spectra {
namespace ui {

// Static instance for callback access
static InputHandler* g_inputHandler = nullptr;

void InputHandler::init(GLFWwindow* window, UIManager* uiManager) {
    m_window = window;
    m_uiManager = uiManager;
    g_inputHandler = this;

    // Set up GLFW callbacks
    glfwSetCursorPosCallback(window, mousePositionCallback);
    glfwSetMouseButtonCallback(window, mouseButtonCallback);
    glfwSetScrollCallback(window, scrollCallback);
    // Note: Key callback is handled separately to avoid conflicts with existing handlers
}

void InputHandler::shutdown() {
    if (m_window) {
        glfwSetCursorPosCallback(m_window, nullptr);
        glfwSetMouseButtonCallback(m_window, nullptr);
        glfwSetScrollCallback(m_window, nullptr);
    }
    g_inputHandler = nullptr;
    m_window = nullptr;
    m_uiManager = nullptr;
}

bool InputHandler::isMouseButtonDown(int button) const {
    if (button >= 0 && button < 3) {
        return m_mouseButtons[button];
    }
    return false;
}

void InputHandler::mousePositionCallback(GLFWwindow* window, double xpos, double ypos) {
    if (!g_inputHandler || !g_inputHandler->m_uiManager) return;

    g_inputHandler->m_mousePos = make_float2(static_cast<float>(xpos), static_cast<float>(ypos));

    // Forward to UI manager
    g_inputHandler->m_mouseConsumed = g_inputHandler->m_uiManager->handleMouseMove(
        g_inputHandler->m_mousePos);
}

void InputHandler::mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
    if (!g_inputHandler || !g_inputHandler->m_uiManager) return;

    g_inputHandler->m_mods = mods;

    if (button >= 0 && button < 3) {
        g_inputHandler->m_mouseButtons[button] = (action == GLFW_PRESS);
    }

    // Forward to UI manager
    if (action == GLFW_PRESS) {
        g_inputHandler->m_mouseConsumed = g_inputHandler->m_uiManager->handleMouseDown(
            g_inputHandler->m_mousePos, button);
    } else if (action == GLFW_RELEASE) {
        g_inputHandler->m_mouseConsumed = g_inputHandler->m_uiManager->handleMouseUp(
            g_inputHandler->m_mousePos, button);
    }
}

void InputHandler::scrollCallback(GLFWwindow* window, double xoffset, double yoffset) {
    if (!g_inputHandler || !g_inputHandler->m_uiManager) return;

    g_inputHandler->m_mouseConsumed = g_inputHandler->m_uiManager->handleMouseScroll(
        g_inputHandler->m_mousePos, static_cast<float>(yoffset));
}

void InputHandler::keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (!g_inputHandler || !g_inputHandler->m_uiManager) return;

    g_inputHandler->m_mods = mods;

    if (action == GLFW_PRESS) {
        g_inputHandler->m_keyConsumed = g_inputHandler->m_uiManager->handleKeyDown(key, mods);
    } else if (action == GLFW_RELEASE) {
        g_inputHandler->m_keyConsumed = g_inputHandler->m_uiManager->handleKeyUp(key, mods);
    }
}

} // namespace ui
} // namespace spectra
