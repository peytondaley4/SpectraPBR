#include "gl_context.h"
#include "cuda_interop.h"
#include "optix_engine.h"
#include "camera.h"
#include "model_loader.h"
#include "geometry_manager.h"
#include "texture_manager.h"
#include "material_manager.h"
#include "scene_manager.h"
#include <iostream>
#include <chrono>
#include <filesystem>

using namespace spectra;

// Frame timing
struct FrameTimer {
    using Clock = std::chrono::high_resolution_clock;
    using TimePoint = Clock::time_point;

    TimePoint frameStart;
    TimePoint lastFrame;
    double frameTimeMs = 0.0;
    double deltaTime = 0.0;  // In seconds
    double fps = 0.0;
    uint64_t frameCount = 0;

    // Rolling average
    static constexpr int SAMPLE_COUNT = 60;
    double samples[SAMPLE_COUNT] = {};
    int sampleIndex = 0;

    void beginFrame() {
        frameStart = Clock::now();
        if (frameCount > 0) {
            deltaTime = std::chrono::duration<double>(frameStart - lastFrame).count();
        } else {
            deltaTime = 1.0 / 60.0;  // Assume 60 FPS for first frame
        }
        lastFrame = frameStart;
    }

    void endFrame() {
        auto now = Clock::now();
        frameTimeMs = std::chrono::duration<double, std::milli>(now - frameStart).count();

        samples[sampleIndex] = frameTimeMs;
        sampleIndex = (sampleIndex + 1) % SAMPLE_COUNT;

        double sum = 0.0;
        for (int i = 0; i < SAMPLE_COUNT; i++) {
            sum += samples[i];
        }
        double avgMs = sum / SAMPLE_COUNT;
        fps = 1000.0 / avgMs;

        frameCount++;
    }

    void print() const {
        std::cout << "[Timing] Frame: " << frameTimeMs << " ms, FPS: " << fps
                  << " (avg over " << SAMPLE_COUNT << " frames), dt: " << deltaTime << "s\n";
    }
};

// Global state for callbacks
static FrameTimer g_timer;
static CudaInterop* g_cudaInterop = nullptr;
static Camera* g_camera = nullptr;
static bool g_mouseCaptured = false;
static double g_lastMouseX = 0.0, g_lastMouseY = 0.0;
static bool g_firstMouse = true;

// Input state
static bool g_keyW = false, g_keyS = false, g_keyA = false, g_keyD = false;
static bool g_keyQ = false, g_keyE = false, g_keyShift = false;

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    (void)scancode; (void)mods;

    // Track key state
    if (action == GLFW_PRESS || action == GLFW_RELEASE) {
        bool pressed = (action == GLFW_PRESS);
        switch (key) {
            case GLFW_KEY_W: g_keyW = pressed; break;
            case GLFW_KEY_S: g_keyS = pressed; break;
            case GLFW_KEY_A: g_keyA = pressed; break;
            case GLFW_KEY_D: g_keyD = pressed; break;
            case GLFW_KEY_Q: g_keyQ = pressed; break;
            case GLFW_KEY_E: g_keyE = pressed; break;
            case GLFW_KEY_LEFT_SHIFT:
            case GLFW_KEY_RIGHT_SHIFT:
                g_keyShift = pressed;
                break;
            default: break;
        }
    }

    if (action != GLFW_PRESS) return;

    auto* ctx = static_cast<GLContext*>(glfwGetWindowUserPointer(window));

    switch (key) {
        case GLFW_KEY_ESCAPE:
            if (g_mouseCaptured) {
                // Release mouse first
                g_mouseCaptured = false;
                glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
                std::cout << "[Main] Mouse released\n";
            } else {
                glfwSetWindowShouldClose(window, GLFW_TRUE);
            }
            break;

        case GLFW_KEY_TAB:
            g_mouseCaptured = !g_mouseCaptured;
            if (g_mouseCaptured) {
                glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
                g_firstMouse = true;
                std::cout << "[Main] Mouse captured\n";
            } else {
                glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
                std::cout << "[Main] Mouse released\n";
            }
            break;

        case GLFW_KEY_V:
            if (ctx) ctx->setVSync(!ctx->isVSyncEnabled());
            break;

        case GLFW_KEY_F:
            if (ctx) ctx->toggleFullscreen();
            break;

        case GLFW_KEY_T:
            g_timer.print();
            break;

        case GLFW_KEY_G:
            if (g_cudaInterop) {
                g_cudaInterop->printDeviceInfo();
                g_cudaInterop->printMemoryUsage();
            }
            break;

        case GLFW_KEY_C:
            if (g_camera) {
                std::cout << "[Camera] Position: " << g_camera->getPosition().x << ", "
                          << g_camera->getPosition().y << ", " << g_camera->getPosition().z
                          << " Yaw: " << g_camera->getYaw() << " Pitch: " << g_camera->getPitch()
                          << " FOV: " << g_camera->getFOV() << "\n";
            }
            break;

        case GLFW_KEY_1:
            if (ctx) {
                ctx->setResolution(RESOLUTION_720P.width, RESOLUTION_720P.height);
                if (g_camera) g_camera->setAspectRatio(1280.0f / 720.0f);
            }
            break;

        case GLFW_KEY_2:
            if (ctx) {
                ctx->setResolution(RESOLUTION_1080P.width, RESOLUTION_1080P.height);
                if (g_camera) g_camera->setAspectRatio(1920.0f / 1080.0f);
            }
            break;

        case GLFW_KEY_3:
            if (ctx) {
                ctx->setResolution(RESOLUTION_1440P.width, RESOLUTION_1440P.height);
                if (g_camera) g_camera->setAspectRatio(2560.0f / 1440.0f);
            }
            break;

        case GLFW_KEY_4:
            if (ctx) {
                ctx->setResolution(RESOLUTION_4K.width, RESOLUTION_4K.height);
                if (g_camera) g_camera->setAspectRatio(3840.0f / 2160.0f);
            }
            break;

        default:
            break;
    }
}

void cursorPosCallback(GLFWwindow* window, double xpos, double ypos) {
    (void)window;

    if (!g_mouseCaptured || !g_camera) return;

    if (g_firstMouse) {
        g_lastMouseX = xpos;
        g_lastMouseY = ypos;
        g_firstMouse = false;
        return;
    }

    float deltaX = static_cast<float>(xpos - g_lastMouseX);
    float deltaY = static_cast<float>(ypos - g_lastMouseY);
    g_lastMouseX = xpos;
    g_lastMouseY = ypos;

    g_camera->processMouseMovement(deltaX, deltaY);
}

void scrollCallback(GLFWwindow* window, double xoffset, double yoffset) {
    (void)window; (void)xoffset;

    if (g_camera) {
        g_camera->processMouseScroll(static_cast<float>(yoffset));
    }
}

void updateCamera(float deltaTime) {
    if (!g_camera) return;

    float forward = 0.0f, right = 0.0f, up = 0.0f;

    if (g_keyW) forward += 1.0f;
    if (g_keyS) forward -= 1.0f;
    if (g_keyD) right += 1.0f;
    if (g_keyA) right -= 1.0f;
    if (g_keyE) up += 1.0f;
    if (g_keyQ) up -= 1.0f;

    g_camera->processKeyboard(forward, right, up, deltaTime, g_keyShift);
}

int main(int argc, char* argv[]) {
    std::cout << "=== SpectraPBR - Phase 2 ===\n\n";

    // Parse command line for model path
    std::filesystem::path modelPath;
    if (argc > 1) {
        modelPath = argv[1];
    }

    // Get executable directory for finding shaders and PTX files
    std::filesystem::path exePath = std::filesystem::absolute(argv[0]).parent_path();
    std::filesystem::path shaderDir = exePath / "shaders";
    std::filesystem::path ptxDir = exePath / "optix_programs";

    // Fall back to source directories if running from build directory
    if (!std::filesystem::exists(shaderDir)) {
        shaderDir = std::filesystem::current_path() / "shaders";
    }
    if (!std::filesystem::exists(ptxDir)) {
        ptxDir = std::filesystem::current_path() / "optix_programs";
    }

    std::cout << "[Main] Shader directory: " << shaderDir << "\n";
    std::cout << "[Main] PTX directory: " << ptxDir << "\n";
    if (!modelPath.empty()) {
        std::cout << "[Main] Model path: " << modelPath << "\n";
    }
    std::cout << "\n";

    // Initialize OpenGL context
    GLContext glContext;
    if (!glContext.init(RESOLUTION_1080P.width, RESOLUTION_1080P.height, "SpectraPBR")) {
        std::cerr << "[Main] Failed to initialize OpenGL context\n";
        return 1;
    }

    // Initialize CUDA (must be after OpenGL)
    CudaInterop cudaInterop;
    g_cudaInterop = &cudaInterop;

    if (!cudaInterop.init()) {
        std::cerr << "[Main] Failed to initialize CUDA\n";
        return 1;
    }

    // Initialize OptiX
    OptixEngine optixEngine;
    if (!optixEngine.init(cudaInterop.getCudaContext())) {
        std::cerr << "[Main] Failed to initialize OptiX\n";
        return 1;
    }

    // Create display resources (texture, PBO, shaders)
    if (!glContext.createDisplayResources(shaderDir)) {
        std::cerr << "[Main] Failed to create display resources\n";
        return 1;
    }

    // Register PBO with CUDA
    if (!cudaInterop.registerPBO(glContext.getPBO(), glContext.getBufferSize())) {
        std::cerr << "[Main] Failed to register PBO with CUDA\n";
        return 1;
    }

    // Load OptiX pipeline
    if (!optixEngine.createPipeline(ptxDir)) {
        std::cerr << "[Main] Failed to create OptiX pipeline\n";
        return 1;
    }

    // Initialize managers
    GeometryManager geometryManager;
    TextureManager textureManager;
    MaterialManager materialManager;
    materialManager.setTextureManager(&textureManager);

    SceneManager sceneManager;
    sceneManager.setOptixEngine(&optixEngine);
    sceneManager.setGeometryManager(&geometryManager);
    sceneManager.setMaterialManager(&materialManager);

    // Initialize camera
    Camera camera;
    camera.setPosition(glm::vec3(0.0f, 1.0f, 5.0f));
    camera.setAspectRatio(static_cast<float>(glContext.getWidth()) / static_cast<float>(glContext.getHeight()));
    g_camera = &camera;

    // Load model if provided
    if (!modelPath.empty() && std::filesystem::exists(modelPath)) {
        ModelLoader loader;
        auto loadedModel = loader.load(modelPath);

        if (loadedModel) {
            std::cout << "[Main] Loading model: " << loadedModel->name << "\n";

            // Add materials
            for (const auto& matData : loadedModel->materials) {
                materialManager.addMaterial(matData);
            }

            // Add meshes and instances
            for (const auto& instance : loadedModel->instances) {
                if (instance.meshIndex < loadedModel->meshes.size()) {
                    const MeshData& mesh = loadedModel->meshes[instance.meshIndex];
                    uint32_t gasIndex = sceneManager.addMesh(mesh);
                    if (gasIndex != UINT32_MAX) {
                        sceneManager.addInstance(gasIndex, instance.transform);
                    }
                }
            }

            // Build BVH
            if (sceneManager.buildIAS()) {
                sceneManager.updateSBT();
                optixEngine.setSceneHandle(sceneManager.getSceneHandle());
                optixEngine.setGeometryBuffers(sceneManager.getVertexBuffers(),
                                               sceneManager.getIndexBuffers());
            }
        } else {
            std::cerr << "[Main] Failed to load model: " << loader.getLastError() << "\n";
        }
    } else if (!modelPath.empty()) {
        std::cerr << "[Main] Model file not found: " << modelPath << "\n";
    }

    // Set initial dimensions
    optixEngine.setDimensions(glContext.getWidth(), glContext.getHeight());

    // Set up callbacks
    glfwSetKeyCallback(glContext.getWindow(), keyCallback);
    glfwSetCursorPosCallback(glContext.getWindow(), cursorPosCallback);
    glfwSetScrollCallback(glContext.getWindow(), scrollCallback);

    // Set up resize callback
    glContext.setResizeCallback([&](uint32_t width, uint32_t height) {
        std::cout << "[Main] Resize: " << width << "x" << height << "\n";

        // Unregister old PBO
        cudaInterop.unregisterPBO();

        // Re-register new PBO
        if (!cudaInterop.registerPBO(glContext.getPBO(), glContext.getBufferSize())) {
            std::cerr << "[Main] Failed to re-register PBO after resize\n";
        }

        // Update OptiX dimensions
        optixEngine.setDimensions(width, height);

        // Update camera aspect ratio
        camera.setAspectRatio(static_cast<float>(width) / static_cast<float>(height));
    });

    std::cout << "\n[Main] Initialization complete!\n";
    std::cout << "[Main] Controls:\n";
    std::cout << "  ESC   - Quit (or release mouse)\n";
    std::cout << "  TAB   - Toggle mouse capture for camera\n";
    std::cout << "  WASD  - Move camera\n";
    std::cout << "  QE    - Move up/down\n";
    std::cout << "  Shift - Sprint (3x speed)\n";
    std::cout << "  Mouse - Look around (when captured)\n";
    std::cout << "  Scroll- Adjust FOV\n";
    std::cout << "  V     - Toggle VSync\n";
    std::cout << "  F     - Toggle Fullscreen\n";
    std::cout << "  1-4   - Resolution (720p/1080p/1440p/4K)\n";
    std::cout << "  T     - Print timing info\n";
    std::cout << "  G     - Print GPU info\n";
    std::cout << "  C     - Print camera info\n";
    std::cout << "\n";

    // Main render loop
    while (!glContext.shouldClose()) {
        g_timer.beginFrame();

        // Poll events
        glContext.pollEvents();

        // Update camera
        updateCamera(static_cast<float>(g_timer.deltaTime));

        // Update OptiX camera params
        optixEngine.setCamera(camera.getCameraParams());

        // Map PBO for CUDA access
        float4* devicePtr = reinterpret_cast<float4*>(cudaInterop.mapPBO());
        if (!devicePtr) {
            std::cerr << "[Main] Failed to map PBO\n";
            break;
        }

        // Render with OptiX
        optixEngine.render(devicePtr, cudaInterop.getStream());

        // Synchronize before unmapping
        cudaInterop.synchronize();

        // Unmap PBO
        cudaInterop.unmapPBO();

        // Update OpenGL texture from PBO
        glContext.updateTextureFromPBO();

        // Render fullscreen quad
        glContext.renderFullscreenQuad();

        // Swap buffers
        glContext.swapBuffers();

        g_timer.endFrame();
    }

    std::cout << "\n[Main] Shutting down...\n";
    std::cout << "[Main] Total frames rendered: " << g_timer.frameCount << "\n";

    // Cleanup
    g_cudaInterop = nullptr;
    g_camera = nullptr;

    // Clear scene before managers are destroyed
    sceneManager.clear();

    std::cout << "[Main] Goodbye!\n";

    return 0;
}
