#include "gl_context.h"
#include "shader_utils.h"
#include <iostream>

namespace spectra {

GLContext::~GLContext() {
    shutdown();
}

bool GLContext::init(uint32_t width, uint32_t height, const char* title) {
    m_width = width;
    m_height = height;

    // Initialize GLFW
    if (!glfwInit()) {
        std::cerr << "[GL] Failed to initialize GLFW\n";
        return false;
    }

    // Request OpenGL 4.5 core profile
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

#ifdef _DEBUG
    glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GL_TRUE);
#endif

    // Create window
    m_window = glfwCreateWindow(width, height, title, nullptr, nullptr);
    if (!m_window) {
        std::cerr << "[GL] Failed to create GLFW window\n";
        glfwTerminate();
        return false;
    }

    glfwMakeContextCurrent(m_window);

    // Store this pointer for callbacks
    glfwSetWindowUserPointer(m_window, this);
    glfwSetFramebufferSizeCallback(m_window, framebufferSizeCallback);
    glfwSetKeyCallback(m_window, keyCallback);

    // Load OpenGL functions
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "[GL] Failed to initialize GLAD\n";
        glfwDestroyWindow(m_window);
        glfwTerminate();
        m_window = nullptr;
        return false;
    }

    // Print OpenGL info
    std::cout << "[GL] OpenGL Version: " << glGetString(GL_VERSION) << "\n";
    std::cout << "[GL] GLSL Version: " << glGetString(GL_SHADING_LANGUAGE_VERSION) << "\n";
    std::cout << "[GL] Vendor: " << glGetString(GL_VENDOR) << "\n";
    std::cout << "[GL] Renderer: " << glGetString(GL_RENDERER) << "\n";

    // Enable VSync by default
    setVSync(true);

    // Set up viewport
    glViewport(0, 0, width, height);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

    // Store windowed position/size for fullscreen toggle
    glfwGetWindowPos(m_window, &m_windowedX, &m_windowedY);
    m_windowedWidth = width;
    m_windowedHeight = height;

    return true;
}

void GLContext::shutdown() {
    if (m_emptyVAO) {
        glDeleteVertexArrays(1, &m_emptyVAO);
        m_emptyVAO = 0;
    }
    if (m_displayProgram) {
        glDeleteProgram(m_displayProgram);
        m_displayProgram = 0;
    }
    if (m_pbo) {
        glDeleteBuffers(1, &m_pbo);
        m_pbo = 0;
    }
    if (m_displayTexture) {
        glDeleteTextures(1, &m_displayTexture);
        m_displayTexture = 0;
    }
    if (m_window) {
        glfwDestroyWindow(m_window);
        m_window = nullptr;
    }
    glfwTerminate();
}

bool GLContext::shouldClose() const {
    return m_window && glfwWindowShouldClose(m_window);
}

void GLContext::pollEvents() {
    glfwPollEvents();
}

void GLContext::swapBuffers() {
    if (m_window) {
        glfwSwapBuffers(m_window);
    }
}

bool GLContext::createDisplayResources(const std::filesystem::path& shaderDir) {
    // Create display shader program
    auto vertPath = shaderDir / "display.vert";
    auto fragPath = shaderDir / "display.frag";

    m_displayProgram = createProgramFromFiles(vertPath, fragPath);
    if (m_displayProgram == 0) {
        std::cerr << "[GL] Failed to create display shader program\n";
        return false;
    }

    // Set texture sampler uniform
    glUseProgram(m_displayProgram);
    glUniform1i(glGetUniformLocation(m_displayProgram, "uTexture"), 0);
    glUseProgram(0);

    // Create empty VAO for fullscreen triangle
    glGenVertexArrays(1, &m_emptyVAO);

    // Create display texture (RGBA32F for HDR)
    glGenTextures(1, &m_displayTexture);
    glBindTexture(GL_TEXTURE_2D, m_displayTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, m_width, m_height, 0,
                 GL_RGBA, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glBindTexture(GL_TEXTURE_2D, 0);

    // Create PBO for efficient CUDA -> texture transfer
    glGenBuffers(1, &m_pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, getBufferSize(), nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    std::cout << "[GL] Display resources created: " << m_width << "x" << m_height
              << " (" << getBufferSize() / (1024 * 1024) << " MB)\n";

    return true;
}

void GLContext::updateTextureFromPBO() {
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_pbo);
    glBindTexture(GL_TEXTURE_2D, m_displayTexture);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, m_width, m_height,
                    GL_RGBA, GL_FLOAT, nullptr);  // nullptr = read from bound PBO
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
}

void GLContext::renderFullscreenQuad() {
    glClear(GL_COLOR_BUFFER_BIT);
    glUseProgram(m_displayProgram);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, m_displayTexture);
    glBindVertexArray(m_emptyVAO);
    glDrawArrays(GL_TRIANGLES, 0, 3);  // Fullscreen triangle
    glBindVertexArray(0);
    glUseProgram(0);
}

void GLContext::setVSync(bool enabled) {
    m_vsyncEnabled = enabled;
    glfwSwapInterval(enabled ? 1 : 0);
    std::cout << "[GL] VSync: " << (enabled ? "ON" : "OFF") << "\n";
}

void GLContext::toggleFullscreen() {
    m_fullscreen = !m_fullscreen;

    if (m_fullscreen) {
        // Save windowed position and size
        glfwGetWindowPos(m_window, &m_windowedX, &m_windowedY);
        glfwGetWindowSize(m_window, &m_windowedWidth, &m_windowedHeight);

        // Get primary monitor
        GLFWmonitor* monitor = glfwGetPrimaryMonitor();
        const GLFWvidmode* mode = glfwGetVideoMode(monitor);

        glfwSetWindowMonitor(m_window, monitor, 0, 0,
                             mode->width, mode->height, mode->refreshRate);
    } else {
        // Restore windowed mode
        glfwSetWindowMonitor(m_window, nullptr,
                             m_windowedX, m_windowedY,
                             m_windowedWidth, m_windowedHeight, 0);
    }

    std::cout << "[GL] Fullscreen: " << (m_fullscreen ? "ON" : "OFF") << "\n";
}

void GLContext::setResolution(uint32_t width, uint32_t height) {
    if (width == m_width && height == m_height) {
        return;
    }

    m_width = width;
    m_height = height;
    glViewport(0, 0, width, height);

    if (!m_fullscreen) {
        glfwSetWindowSize(m_window, width, height);
    }

    recreateBuffers();

    if (m_resizeCallback) {
        m_resizeCallback(width, height);
    }
}

void GLContext::recreateBuffers() {
    // Recreate texture
    if (m_displayTexture) {
        glBindTexture(GL_TEXTURE_2D, m_displayTexture);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, m_width, m_height, 0,
                     GL_RGBA, GL_FLOAT, nullptr);
        glBindTexture(GL_TEXTURE_2D, 0);
    }

    // Recreate PBO
    if (m_pbo) {
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_pbo);
        glBufferData(GL_PIXEL_UNPACK_BUFFER, getBufferSize(), nullptr, GL_DYNAMIC_DRAW);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    }

    std::cout << "[GL] Buffers resized: " << m_width << "x" << m_height << "\n";
}

void GLContext::framebufferSizeCallback(GLFWwindow* window, int width, int height) {
    auto* ctx = static_cast<GLContext*>(glfwGetWindowUserPointer(window));
    if (ctx && width > 0 && height > 0) {
        ctx->m_width = width;
        ctx->m_height = height;
        glViewport(0, 0, width, height);
        ctx->recreateBuffers();

        if (ctx->m_resizeCallback) {
            ctx->m_resizeCallback(width, height);
        }
    }
}

void GLContext::keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    (void)scancode;
    (void)mods;

    if (action != GLFW_PRESS) {
        return;
    }

    auto* ctx = static_cast<GLContext*>(glfwGetWindowUserPointer(window));
    if (!ctx) {
        return;
    }

    switch (key) {
        case GLFW_KEY_ESCAPE:
            glfwSetWindowShouldClose(window, GLFW_TRUE);
            break;

        case GLFW_KEY_V:
            ctx->setVSync(!ctx->isVSyncEnabled());
            break;

        case GLFW_KEY_F:
            ctx->toggleFullscreen();
            break;

        case GLFW_KEY_1:
            ctx->setResolution(RESOLUTION_720P.width, RESOLUTION_720P.height);
            std::cout << "[GL] Resolution: " << RESOLUTION_720P.name << "\n";
            break;

        case GLFW_KEY_2:
            ctx->setResolution(RESOLUTION_1080P.width, RESOLUTION_1080P.height);
            std::cout << "[GL] Resolution: " << RESOLUTION_1080P.name << "\n";
            break;

        case GLFW_KEY_3:
            ctx->setResolution(RESOLUTION_1440P.width, RESOLUTION_1440P.height);
            std::cout << "[GL] Resolution: " << RESOLUTION_1440P.name << "\n";
            break;

        case GLFW_KEY_4:
            ctx->setResolution(RESOLUTION_4K.width, RESOLUTION_4K.height);
            std::cout << "[GL] Resolution: " << RESOLUTION_4K.name << "\n";
            break;

        default:
            break;
    }
}

} // namespace spectra
