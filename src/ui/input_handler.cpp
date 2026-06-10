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

    // Deliberately registers NO GLFW callbacks: the Application owns all of
    // them. GLFW keeps a single callback per event — when both this class and
    // the Application registered, whichever came last silently won and the
    // other's state went permanently stale (wasMouseConsumed() never updated).
    // The Application now feeds state in via setMouseConsumed().
}

void InputHandler::shutdown() {
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

} // namespace ui
} // namespace spectra
