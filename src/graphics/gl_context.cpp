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

    // Disable VSync by default
    setVSync(false);

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
    // Clean up triple-buffered PBOs
    for (int i = 0; i < NUM_SCENE_BUFFERS; ++i) {
        if (m_pbos[i]) {
            glDeleteBuffers(1, &m_pbos[i]);
            m_pbos[i] = 0;
        }
    }
    // Clean up triple-buffered textures
    for (int i = 0; i < NUM_SCENE_BUFFERS; ++i) {
        if (m_displayTextures[i]) {
            glDeleteTextures(1, &m_displayTextures[i]);
            m_displayTextures[i] = 0;
        }
    }
    if (m_uiPbo) {
        glDeleteBuffers(1, &m_uiPbo);
        m_uiPbo = 0;
    }
    if (m_uiTexture) {
        glDeleteTextures(1, &m_uiTexture);
        m_uiTexture = 0;
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

    // Set texture sampler uniforms
    glUseProgram(m_displayProgram);
    glUniform1i(glGetUniformLocation(m_displayProgram, "uSceneTexture"), 0);
    m_uiTextureLoc = glGetUniformLocation(m_displayProgram, "uUITexture");
    glUniform1i(m_uiTextureLoc, 1);
    glUniform1i(glGetUniformLocation(m_displayProgram, "uUIEnabled"), 0);
    m_exposureLoc = glGetUniformLocation(m_displayProgram, "uExposure");
    glUniform1f(m_exposureLoc, m_exposure);
    glUseProgram(0);

    // Create empty VAO for fullscreen triangle
    glGenVertexArrays(1, &m_emptyVAO);

    // Create triple-buffered display textures (RGBA32F for HDR)
    for (int i = 0; i < NUM_SCENE_BUFFERS; ++i) {
        glGenTextures(1, &m_displayTextures[i]);
        glBindTexture(GL_TEXTURE_2D, m_displayTextures[i]);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, m_width, m_height, 0,
                     GL_RGBA, GL_FLOAT, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    }
    glBindTexture(GL_TEXTURE_2D, 0);

    // Create triple-buffered PBOs for efficient CUDA -> texture transfer
    for (int i = 0; i < NUM_SCENE_BUFFERS; ++i) {
        glGenBuffers(1, &m_pbos[i]);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_pbos[i]);
        glBufferData(GL_PIXEL_UNPACK_BUFFER, getBufferSize(), nullptr, GL_DYNAMIC_DRAW);
    }
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    // Initialize buffer indices
    m_writeBuffer = 0;
    m_displayBuffer = 0;

    // Create UI texture (RGBA32F for alpha compositing) - single buffered
    glGenTextures(1, &m_uiTexture);
    glBindTexture(GL_TEXTURE_2D, m_uiTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, m_width, m_height, 0,
                 GL_RGBA, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glBindTexture(GL_TEXTURE_2D, 0);

    // Create UI PBO
    glGenBuffers(1, &m_uiPbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_uiPbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, getBufferSize(), nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    std::cout << "[GL] Display resources created: " << m_width << "x" << m_height
              << " (" << (getBufferSize() * NUM_SCENE_BUFFERS) / (1024 * 1024) << " MB scene ["
              << NUM_SCENE_BUFFERS << " buffers] + " << getBufferSize() / (1024 * 1024) << " MB UI)\n";

    return true;
}

void GLContext::updateTextureFromPBO() {
    // Legacy single-buffer path: update from first buffer
    updateTextureFromPBO(0);
}

void GLContext::updateTextureFromPBO(int bufferIndex) {
    if (bufferIndex < 0 || bufferIndex >= NUM_SCENE_BUFFERS) return;
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_pbos[bufferIndex]);
    glBindTexture(GL_TEXTURE_2D, m_displayTextures[bufferIndex]);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, m_width, m_height,
                    GL_RGBA, GL_FLOAT, nullptr);  // nullptr = read from bound PBO
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
}

void GLContext::updateUITextureFromPBO() {
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_uiPbo);
    glBindTexture(GL_TEXTURE_2D, m_uiTexture);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, m_width, m_height,
                    GL_RGBA, GL_FLOAT, nullptr);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
}

void GLContext::renderFullscreenQuad() {
    // Legacy single-buffer path: render from first buffer
    renderFullscreenQuad(0);
}

void GLContext::renderFullscreenQuad(int displayBufferIndex) {
    if (displayBufferIndex < 0 || displayBufferIndex >= NUM_SCENE_BUFFERS) {
        displayBufferIndex = 0;
    }

    glClear(GL_COLOR_BUFFER_BIT);

    glUseProgram(m_displayProgram);

    // Bind scene texture from specified buffer
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, m_displayTextures[displayBufferIndex]);

    // Bind UI texture and set enabled flag
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, m_uiTexture);
    glUniform1i(glGetUniformLocation(m_displayProgram, "uUIEnabled"), m_uiEnabled ? 1 : 0);
    glUniform1f(m_exposureLoc, m_exposure);

    // Enable blending for UI compositing
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glBindVertexArray(m_emptyVAO);
    glDrawArrays(GL_TRIANGLES, 0, 3);  // Fullscreen triangle
    glBindVertexArray(0);

    glDisable(GL_BLEND);
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

    if (!m_fullscreen) {
        // In windowed mode, just set the window size.
        // This will trigger framebufferSizeCallback which handles everything:
        // pre-resize callback, buffer recreation, and post-resize callback.
        glfwSetWindowSize(m_window, width, height);
    } else {
        // In fullscreen mode, glfwSetWindowSize doesn't trigger the callback,
        // so we need to handle everything manually here.
        if (m_preResizeCallback) {
            m_preResizeCallback();
        }

        m_width = width;
        m_height = height;
        glViewport(0, 0, width, height);
        recreateBuffers();

        if (m_resizeCallback) {
            m_resizeCallback(width, height);
        }
    }
}

void GLContext::recreateBuffers() {
    // Recreate triple-buffered scene textures
    for (int i = 0; i < NUM_SCENE_BUFFERS; ++i) {
        if (m_displayTextures[i]) {
            glBindTexture(GL_TEXTURE_2D, m_displayTextures[i]);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, m_width, m_height, 0,
                         GL_RGBA, GL_FLOAT, nullptr);
        }
    }
    glBindTexture(GL_TEXTURE_2D, 0);

    // Recreate UI texture
    if (m_uiTexture) {
        glBindTexture(GL_TEXTURE_2D, m_uiTexture);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, m_width, m_height, 0,
                     GL_RGBA, GL_FLOAT, nullptr);
        glBindTexture(GL_TEXTURE_2D, 0);
    }

    // Completely delete and recreate triple-buffered PBOs (not just reallocate)
    // This ensures CUDA gets fresh GL objects after re-registration
    for (int i = 0; i < NUM_SCENE_BUFFERS; ++i) {
        if (m_pbos[i]) {
            glDeleteBuffers(1, &m_pbos[i]);
            m_pbos[i] = 0;
        }

        glGenBuffers(1, &m_pbos[i]);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_pbos[i]);
        glBufferData(GL_PIXEL_UNPACK_BUFFER, getBufferSize(), nullptr, GL_DYNAMIC_DRAW);
    }
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    // Reset buffer indices
    m_writeBuffer = 0;
    m_displayBuffer = 0;

    // Recreate UI PBO
    if (m_uiPbo) {
        glDeleteBuffers(1, &m_uiPbo);
        m_uiPbo = 0;
    }

    glGenBuffers(1, &m_uiPbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_uiPbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, getBufferSize(), nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    std::cout << "[GL] Buffers resized: " << m_width << "x" << m_height
              << " (PBOs " << m_pbos[0] << "/" << m_pbos[1] << "/" << m_pbos[2]
              << ", UI PBO " << m_uiPbo << ", " << getBufferSize() / (1024 * 1024) << " MB each)\n";
}

void GLContext::framebufferSizeCallback(GLFWwindow* window, int width, int height) {
    auto* ctx = static_cast<GLContext*>(glfwGetWindowUserPointer(window));
    if (ctx && width > 0 && height > 0) {
        // Notify pre-resize callback BEFORE recreating buffers
        // This allows CUDA to unregister resources before the PBO is invalidated
        if (ctx->m_preResizeCallback) {
            ctx->m_preResizeCallback();
        }

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
