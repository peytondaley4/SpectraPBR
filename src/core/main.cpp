#include "gl_context.h"
#include "cuda_interop.h"
#include "optix_engine.h"
#include "camera.h"
#include "model_loader.h"
#include "geometry_manager.h"
#include "texture_manager.h"
#include "material_manager.h"
#include "scene_manager.h"
#include "environment_map.h"
// Phase 4: UI System
#include "text/font_atlas.h"
#include "text/text_layout.h"
#include "ui/ui_manager.h"
#include "ui/ui_renderer.h"
#include "ui/input_handler.h"
#include "ui/property_panel.h"
#include "ui/texture_preview_cache.h"
#include "scene/selection_manager.h"
#include "scene/scene_serializer.h"
#include "scene/scene_hierarchy.h"
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

//------------------------------------------------------------------------------
// Light Manager - Manages CPU-side light data for editing
//------------------------------------------------------------------------------
struct LightManager {
    // CPU-side light data
    std::vector<GpuDirectionalLight> dirLights;
    std::vector<GpuAreaLight> areaLights;
    std::vector<GpuPointLight> pointLights;

    // GPU buffers
    CUdeviceptr d_dirLights = 0;
    CUdeviceptr d_areaLights = 0;
    CUdeviceptr d_pointLights = 0;

    void addDirectionalLight(const GpuDirectionalLight& light) {
        dirLights.push_back(light);
    }

    void addAreaLight(const GpuAreaLight& light) {
        areaLights.push_back(light);
    }

    void addPointLight(const GpuPointLight& light) {
        pointLights.push_back(light);
    }

    void updateDirectionalLight(uint32_t index, const ui::LightInfo& info) {
        if (index >= dirLights.size()) return;
        dirLights[index].direction = info.direction;
        dirLights[index].angularDiameter = info.angularDiameter;
        dirLights[index].irradiance = info.color;
    }

    void updateAreaLight(uint32_t index, const ui::LightInfo& info) {
        if (index >= areaLights.size()) return;
        areaLights[index].position = info.position;
        areaLights[index].emission = info.color;
        areaLights[index].size = info.size;
        areaLights[index].area = info.size.x * info.size.y;
    }

    void updatePointLight(uint32_t index, const ui::LightInfo& info) {
        if (index >= pointLights.size()) return;
        pointLights[index].position = info.position;
        pointLights[index].radius = info.radius;
        pointLights[index].intensity = info.color;
    }

    void syncToGpu(OptixEngine* engine) {
        // Directional lights
        if (!dirLights.empty()) {
            size_t size = dirLights.size() * sizeof(GpuDirectionalLight);
            if (!d_dirLights) {
                cudaMalloc(reinterpret_cast<void**>(&d_dirLights), size);
            }
            cudaMemcpy(reinterpret_cast<void*>(d_dirLights), dirLights.data(), size, cudaMemcpyHostToDevice);
            engine->setDirectionalLights(reinterpret_cast<GpuDirectionalLight*>(d_dirLights),
                                          static_cast<uint32_t>(dirLights.size()));
        }

        // Area lights
        if (!areaLights.empty()) {
            size_t size = areaLights.size() * sizeof(GpuAreaLight);
            if (!d_areaLights) {
                cudaMalloc(reinterpret_cast<void**>(&d_areaLights), size);
            }
            cudaMemcpy(reinterpret_cast<void*>(d_areaLights), areaLights.data(), size, cudaMemcpyHostToDevice);
            engine->setAreaLights(reinterpret_cast<GpuAreaLight*>(d_areaLights),
                                   static_cast<uint32_t>(areaLights.size()));
        }

        // Point lights
        if (!pointLights.empty()) {
            size_t size = pointLights.size() * sizeof(GpuPointLight);
            if (!d_pointLights) {
                cudaMalloc(reinterpret_cast<void**>(&d_pointLights), size);
            }
            cudaMemcpy(reinterpret_cast<void*>(d_pointLights), pointLights.data(), size, cudaMemcpyHostToDevice);
            engine->setPointLights(reinterpret_cast<GpuPointLight*>(d_pointLights),
                                    static_cast<uint32_t>(pointLights.size()));
        }
    }

    void cleanup() {
        if (d_dirLights) { cudaFree(reinterpret_cast<void*>(d_dirLights)); d_dirLights = 0; }
        if (d_areaLights) { cudaFree(reinterpret_cast<void*>(d_areaLights)); d_areaLights = 0; }
        if (d_pointLights) { cudaFree(reinterpret_cast<void*>(d_pointLights)); d_pointLights = 0; }
    }

    ui::LightInfo getDirectionalLightInfo(uint32_t index) const {
        ui::LightInfo info = {};
        if (index >= dirLights.size()) return info;
        info.type = SceneNodeType::DirectionalLight;
        info.index = index;
        info.direction = dirLights[index].direction;
        info.color = dirLights[index].irradiance;
        info.angularDiameter = dirLights[index].angularDiameter;
        // Calculate intensity from color magnitude
        info.intensity = std::sqrt(info.color.x * info.color.x +
                                    info.color.y * info.color.y +
                                    info.color.z * info.color.z);
        return info;
    }

    ui::LightInfo getAreaLightInfo(uint32_t index) const {
        ui::LightInfo info = {};
        if (index >= areaLights.size()) return info;
        info.type = SceneNodeType::AreaLight;
        info.index = index;
        info.position = areaLights[index].position;
        info.color = areaLights[index].emission;
        info.size = areaLights[index].size;
        info.intensity = std::sqrt(info.color.x * info.color.x +
                                    info.color.y * info.color.y +
                                    info.color.z * info.color.z);
        return info;
    }

    ui::LightInfo getPointLightInfo(uint32_t index) const {
        ui::LightInfo info = {};
        if (index >= pointLights.size()) return info;
        info.type = SceneNodeType::PointLight;
        info.index = index;
        info.position = pointLights[index].position;
        info.radius = pointLights[index].radius;
        info.color = pointLights[index].intensity;
        info.intensity = std::sqrt(info.color.x * info.color.x +
                                    info.color.y * info.color.y +
                                    info.color.z * info.color.z);
        return info;
    }
};

// Global state for callbacks
static FrameTimer g_timer;
static CudaInterop* g_cudaInterop = nullptr;
static Camera* g_camera = nullptr;
static OptixEngine* g_optixEngine = nullptr;
static bool g_mouseCaptured = false;
static double g_lastMouseX = 0.0, g_lastMouseY = 0.0;
static bool g_firstMouse = true;
static bool g_remoteMode = false;  // Disable cursor capture for remote desktop compatibility
static QualityMode g_qualityMode = QUALITY_BALANCED;

// Input state
static bool g_keyW = false, g_keyS = false, g_keyA = false, g_keyD = false;
static bool g_keyQ = false, g_keyE = false, g_keyShift = false;

// Phase 4: UI globals
static ui::UIManager* g_uiManager = nullptr;
static ui::InputHandler* g_inputHandler = nullptr;
static SelectionManager* g_selectionManager = nullptr;
static SceneSerializer* g_sceneSerializer = nullptr;
static SceneManager* g_sceneManager = nullptr;
static SceneHierarchy* g_sceneHierarchy = nullptr;
static LightManager* g_lightManager = nullptr;

// Quality mode names for display
static const char* getQualityModeName(QualityMode mode) {
    switch (mode) {
        case QUALITY_FAST: return "Fast";
        case QUALITY_BALANCED: return "Balanced";
        case QUALITY_HIGH: return "High";
        case QUALITY_ACCURATE: return "Accurate";
        default: return "Unknown";
    }
}

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
                if (!g_remoteMode) {
                    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
                }
                std::cout << "[Main] Mouse released\n";
            } else {
                glfwSetWindowShouldClose(window, GLFW_TRUE);
            }
            break;

        case GLFW_KEY_TAB:
            g_mouseCaptured = !g_mouseCaptured;
            if (g_mouseCaptured) {
                if (!g_remoteMode) {
                    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
                }
                g_firstMouse = true;
                std::cout << "[Main] Mouse captured\n";
            } else {
                if (!g_remoteMode) {
                    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
                }
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

        case GLFW_KEY_R:
            g_remoteMode = !g_remoteMode;
            std::cout << "[Main] Remote mode: " << (g_remoteMode ? "ON" : "OFF") 
                      << " (cursor capture " << (g_remoteMode ? "disabled" : "enabled") << ")\n";
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

        // Quality mode switching (F1-F4)
        case GLFW_KEY_F1:
            g_qualityMode = QUALITY_FAST;
            if (g_optixEngine) g_optixEngine->setQualityMode(g_qualityMode);
            std::cout << "[Main] Quality mode: Fast (Lambertian + basic specular)\n";
            break;

        case GLFW_KEY_F2:
            g_qualityMode = QUALITY_BALANCED;
            if (g_optixEngine) g_optixEngine->setQualityMode(g_qualityMode);
            std::cout << "[Main] Quality mode: Balanced (Full GGX)\n";
            break;

        case GLFW_KEY_F3:
            g_qualityMode = QUALITY_HIGH;
            if (g_optixEngine) g_optixEngine->setQualityMode(g_qualityMode);
            std::cout << "[Main] Quality mode: High (VNDF + clearcoat/sheen)\n";
            break;

        case GLFW_KEY_F4:
            g_qualityMode = QUALITY_ACCURATE;
            if (g_optixEngine) g_optixEngine->setQualityMode(g_qualityMode);
            std::cout << "[Main] Quality mode: Accurate (Full PBR + conductor Fresnel)\n";
            break;

        // SPP adjustment
        case GLFW_KEY_LEFT_BRACKET:
            if (g_optixEngine) {
                uint32_t spp = g_optixEngine->getSamplesPerPixel();
                spp = spp > 1 ? spp / 2 : 1;
                g_optixEngine->setSamplesPerPixel(spp);
                std::cout << "[Main] Samples per pixel: " << spp << "\n";
            }
            break;

        case GLFW_KEY_RIGHT_BRACKET:
            if (g_optixEngine) {
                uint32_t spp = g_optixEngine->getSamplesPerPixel();
                spp = spp < 64 ? spp * 2 : 64;
                g_optixEngine->setSamplesPerPixel(spp);
                std::cout << "[Main] Samples per pixel: " << spp << "\n";
            }
            break;

        // Phase 4: UI shortcuts
        case GLFW_KEY_H:
            if (g_uiManager) {
                g_uiManager->toggleScenePanel();
                std::cout << "[Main] Scene panel: " << (g_uiManager->isScenePanelVisible() ? "visible" : "hidden") << "\n";
            }
            break;

        case GLFW_KEY_L:
            if (g_uiManager && !(mods & GLFW_MOD_CONTROL)) {
                g_uiManager->toggleTheme();
                std::cout << "[Main] Theme: " << (g_uiManager->isDarkTheme() ? "dark" : "light") << "\n";
            }
            break;

        case GLFW_KEY_P:
            if (g_uiManager) {
                g_uiManager->togglePropertyPanel();
                std::cout << "[Main] Property panel: " << (g_uiManager->isPropertyPanelVisible() ? "visible" : "hidden") << "\n";
            }
            break;

        case GLFW_KEY_S:
            if (mods & GLFW_MOD_CONTROL) {
                // Ctrl+S: Save scene
                if (g_sceneSerializer && g_camera && g_sceneManager) {
                    std::string savePath = SceneSerializer::getAutoSavePath();
                    bool darkTheme = g_uiManager ? g_uiManager->isDarkTheme() : true;
                    if (g_sceneSerializer->saveScene(savePath, g_camera, g_sceneManager, g_qualityMode, darkTheme)) {
                        std::cout << "[Main] Scene saved to: " << savePath << "\n";
                    }
                }
            }
            break;

        case GLFW_KEY_O:
            if (mods & GLFW_MOD_CONTROL) {
                // Ctrl+O: Load scene (just load settings for now)
                if (g_sceneSerializer) {
                    std::string loadPath = SceneSerializer::getAutoSavePath();
                    if (g_sceneSerializer->loadScene(loadPath)) {
                        // Apply loaded camera settings
                        if (g_sceneSerializer->hasLoadedCamera() && g_camera) {
                            g_camera->setPosition(glm::vec3(
                                g_sceneSerializer->getCameraPositionX(),
                                g_sceneSerializer->getCameraPositionY(),
                                g_sceneSerializer->getCameraPositionZ()));
                            g_camera->setYaw(g_sceneSerializer->getCameraYaw());
                            g_camera->setPitch(g_sceneSerializer->getCameraPitch());
                            g_camera->setFOV(g_sceneSerializer->getCameraFov());
                        }
                        // Apply theme
                        if (g_uiManager) {
                            if (g_sceneSerializer->isDarkTheme()) {
                                g_uiManager->setTheme(&ui::THEME_DARK);
                            } else {
                                g_uiManager->setTheme(&ui::THEME_LIGHT);
                            }
                        }
                        // Apply quality mode
                        g_qualityMode = static_cast<QualityMode>(g_sceneSerializer->getQualityMode());
                        if (g_optixEngine) g_optixEngine->setQualityMode(g_qualityMode);

                        std::cout << "[Main] Scene loaded from: " << loadPath << "\n";
                    }
                }
            }
            break;

        default:
            break;
    }
}

void cursorPosCallback(GLFWwindow* window, double xpos, double ypos) {
    (void)window;

    // Always track mouse position for UI
    float2 mousePos = make_float2(static_cast<float>(xpos), static_cast<float>(ypos));
    
    // Forward to UI first (when not captured for camera)
    if (!g_mouseCaptured && g_uiManager) {
        g_uiManager->handleMouseMove(mousePos);
    }
    
    // Handle camera movement when captured
    if (g_mouseCaptured && g_camera) {
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
    } else {
        // Update tracking even when not captured so first movement isn't huge
        g_lastMouseX = xpos;
        g_lastMouseY = ypos;
    }
}

void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
    (void)mods;
    
    // Get current mouse position
    double xpos, ypos;
    glfwGetCursorPos(window, &xpos, &ypos);
    float2 mousePos = make_float2(static_cast<float>(xpos), static_cast<float>(ypos));
    
    // Forward to UI first (when not captured)
    if (!g_mouseCaptured && g_uiManager) {
        if (action == GLFW_PRESS) {
            if (g_uiManager->handleMouseDown(mousePos, button)) {
                return; // UI consumed the event
            }
        } else if (action == GLFW_RELEASE) {
            if (g_uiManager->handleMouseUp(mousePos, button)) {
                return; // UI consumed the event
            }
        }
    }
    
    // Handle left-click for picking (select model)
    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS && !g_mouseCaptured) {
        if (g_optixEngine && g_uiManager) {
            uint32_t pickedId = g_optixEngine->pickInstance(
                static_cast<uint32_t>(xpos),
                static_cast<uint32_t>(ypos)
            );
            
            // Update selection
            g_uiManager->setSelectedInstanceId(pickedId);
            g_optixEngine->setSelectedInstanceId(pickedId);
            g_optixEngine->resetAccumulation();
            
            if (pickedId != UINT32_MAX) {
                std::cout << "[Main] Selected instance: " << pickedId << "\n";
            } else {
                std::cout << "[Main] Selection cleared\n";
            }
        }
    }

    // Handle right-click for camera capture toggle
    if (button == GLFW_MOUSE_BUTTON_RIGHT) {
        if (action == GLFW_PRESS) {
            g_mouseCaptured = true;
            // In remote mode, don't capture/hide cursor - just track deltas
            if (!g_remoteMode) {
                glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
            }
            g_firstMouse = true;
        } else if (action == GLFW_RELEASE) {
            g_mouseCaptured = false;
            if (!g_remoteMode) {
                glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
            }
        }
    }
}

void scrollCallback(GLFWwindow* window, double xoffset, double yoffset) {
    (void)window; (void)xoffset;

    // Forward to UI first (when not captured)
    if (!g_mouseCaptured && g_uiManager) {
        double xpos, ypos;
        glfwGetCursorPos(window, &xpos, &ypos);
        float2 mousePos = make_float2(static_cast<float>(xpos), static_cast<float>(ypos));
        if (g_uiManager->handleMouseScroll(mousePos, static_cast<float>(yoffset))) {
            return; // UI consumed the event
        }
    }
    
    // Camera zoom
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

    // Parse command line arguments
    // Usage: SpectraPBR.exe [model.gltf] [environment.hdr] [--remote]
    std::filesystem::path modelPath;
    std::filesystem::path hdrPath;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--remote" || arg == "-r") {
            g_remoteMode = true;
            std::cout << "[Main] Remote mode enabled (cursor capture disabled)\n";
        } else if (modelPath.empty()) {
            modelPath = arg;
        } else if (hdrPath.empty()) {
            hdrPath = arg;
        }
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

    // If no model specified, try to find the default cube model from tinygltf
    if (modelPath.empty()) {
        // Search paths for the default cube model
        std::vector<std::filesystem::path> searchPaths = {
            exePath / "_deps" / "tinygltf-src" / "models" / "Cube" / "Cube.gltf",
            exePath.parent_path() / "_deps" / "tinygltf-src" / "models" / "Cube" / "Cube.gltf",
            std::filesystem::current_path() / "build" / "_deps" / "tinygltf-src" / "models" / "Cube" / "Cube.gltf",
            std::filesystem::current_path() / "_deps" / "tinygltf-src" / "models" / "Cube" / "Cube.gltf",
        };

        for (const auto& path : searchPaths) {
            if (std::filesystem::exists(path)) {
                modelPath = path;
                std::cout << "[Main] Using default cube model\n";
                break;
            }
        }
    }

    std::cout << "[Main] Shader directory: " << shaderDir << "\n";
    std::cout << "[Main] PTX directory: " << ptxDir << "\n";
    if (!modelPath.empty()) {
        std::cout << "[Main] Model path: " << modelPath << "\n";
    } else {
        std::cout << "[Main] No model specified, will show gradient background\n";
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
    g_optixEngine = &optixEngine;

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

    // Register UI PBO with CUDA
    if (!cudaInterop.registerUIPBO(glContext.getUIPBO(), glContext.getBufferSize())) {
        std::cerr << "[Main] Failed to register UI PBO with CUDA\n";
        return 1;
    }

    //--------------------------------------------------------------------------
    // Phase 4: Initialize UI System
    //--------------------------------------------------------------------------

    // Get fonts directory
    std::filesystem::path fontsDir = exePath / "assets" / "fonts";
    if (!std::filesystem::exists(fontsDir)) {
        fontsDir = std::filesystem::current_path() / "assets" / "fonts";
    }

    // Initialize Font Atlas
    text::FontAtlas fontAtlas;
    if (!fontAtlas.load((fontsDir / "DejaVuSans.ttf").string(), 32.0f, 512, 8)) {
        std::cerr << "[Main] Warning: Failed to load font atlas, UI text will not render\n";
    }

    // Initialize UI Manager
    ui::UIManager uiManager;
    g_uiManager = &uiManager;
    if (!uiManager.init(&fontAtlas, glContext.getWidth(), glContext.getHeight())) {
        std::cerr << "[Main] Warning: Failed to initialize UI manager\n";
    }

    // Initialize UI Renderer
    ui::UIRenderer uiRenderer;
    if (!uiRenderer.init(4096)) {
        std::cerr << "[Main] Warning: Failed to initialize UI renderer\n";
    }

    // Initialize Texture Preview Cache (for lazy texture preview rendering)
    ui::TexturePreviewCache texturePreviewCache;
    if (!texturePreviewCache.init()) {
        std::cerr << "[Main] Warning: Failed to initialize texture preview cache\n";
    }

    // Initialize Input Handler
    ui::InputHandler inputHandler;
    g_inputHandler = &inputHandler;
    inputHandler.init(glContext.getWindow(), &uiManager);

    // Initialize Selection Manager
    SelectionManager selectionManager;
    g_selectionManager = &selectionManager;

    // Initialize Scene Serializer
    SceneSerializer sceneSerializer;
    g_sceneSerializer = &sceneSerializer;

    // Wire selection manager to UI
    uiManager.setSelectionCallback([&](uint32_t instanceId) {
        selectionManager.setSelectedInstanceId(instanceId);
        optixEngine.setSelectedInstanceId(instanceId);
        optixEngine.resetAccumulation();
        std::cout << "[Main] Selected instance: " << instanceId << "\n";
    });

    // Enable UI compositing
    glContext.setUIEnabled(true);

    std::cout << "[Main] UI system initialized\n";

    // Initialize managers
    GeometryManager geometryManager;
    TextureManager textureManager;
    MaterialManager materialManager;
    materialManager.setTextureManager(&textureManager);

    SceneManager sceneManager;
    g_sceneManager = &sceneManager;
    sceneManager.setOptixEngine(&optixEngine);
    sceneManager.setGeometryManager(&geometryManager);
    sceneManager.setMaterialManager(&materialManager);

    // Initialize scene hierarchy for hierarchical tree view
    SceneHierarchy sceneHierarchy;
    g_sceneHierarchy = &sceneHierarchy;
    uiManager.setSceneHierarchy(&sceneHierarchy);
    uiManager.setMaterialManager(&materialManager);

    // Initialize light manager
    LightManager lightManager;
    g_lightManager = &lightManager;

    // Initialize camera
    Camera camera;
    camera.setPosition(glm::vec3(0.0f, 1.0f, 5.0f));
    camera.setAspectRatio(static_cast<float>(glContext.getWidth()) / static_cast<float>(glContext.getHeight()));
    g_camera = &camera;

    // Loaded model name for hierarchy
    std::string loadedModelName = "Model";

    // Load model if provided
    if (!modelPath.empty() && std::filesystem::exists(modelPath)) {
        ModelLoader loader;
        auto loadedModel = loader.load(modelPath);

        if (loadedModel) {
            std::cout << "[Main] Loading model: " << loadedModel->name << "\n";
            loadedModelName = loadedModel->name;

            // Add materials
            for (const auto& matData : loadedModel->materials) {
                materialManager.addMaterial(matData);
            }

            // Add model to hierarchy
            uint32_t modelNodeIdx = sceneHierarchy.addModel(loadedModel->name);

            // Add meshes and instances
            uint32_t instanceId = 0;
            for (const auto& instance : loadedModel->instances) {
                if (instance.meshIndex < loadedModel->meshes.size()) {
                    const MeshData& mesh = loadedModel->meshes[instance.meshIndex];
                    uint32_t gasIndex = sceneManager.addMesh(mesh);
                    if (gasIndex != UINT32_MAX) {
                        sceneManager.addInstance(gasIndex, instance.transform);

                        // Add instance to hierarchy
                        std::string instanceName = "Instance " + std::to_string(instanceId);
                        sceneHierarchy.addInstance(modelNodeIdx, instance.meshIndex, instanceId, instanceName);
                        instanceId++;
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

    //--------------------------------------------------------------------------
    // Set up default lighting using LightManager
    // glTF models may have emissive materials but rarely include explicit lights,
    // so we provide a default sun-like directional light for visibility.
    //--------------------------------------------------------------------------

    // Helper lambda to normalize float3 on host
    auto normalizeFloat3 = [](float3 v) -> float3 {
        float len = std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
        if (len > 0.0f) {
            return make_float3(v.x / len, v.y / len, v.z / len);
        }
        return v;
    };

    // Default directional light (sun)
    GpuDirectionalLight sunLight;
    sunLight.direction = make_float3(0.5f, -0.8f, 0.3f);  // Sun angle
    sunLight.angularDiameter = 0.2f;  // Sharp shadows
    sunLight.irradiance = make_float3(3.0f, 2.9f, 2.7f);  // Warm sunlight
    lightManager.addDirectionalLight(sunLight);
    sceneHierarchy.addDirectionalLight(0, "Sun");

    // Key light - large soft light above and to the right
    GpuAreaLight keyLight;
    keyLight.position = make_float3(3.0f, 4.0f, 2.0f);
    keyLight.normal = normalizeFloat3(make_float3(-0.3f, -0.8f, -0.2f));
    keyLight.tangent = make_float3(1.0f, 0.0f, 0.0f);
    keyLight.emission = make_float3(200.0f, 150.0f, 160.0f);
    keyLight.size = make_float2(2.0f, 2.0f);
    keyLight.area = 4.0f;
    lightManager.addAreaLight(keyLight);
    sceneHierarchy.addAreaLight(0, "Key Light");

    // Fill light - smaller, dimmer light to the left
    GpuAreaLight fillLight;
    fillLight.position = make_float3(-2.5f, 2.0f, 3.0f);
    fillLight.normal = normalizeFloat3(make_float3(0.4f, -0.5f, -0.6f));
    fillLight.tangent = make_float3(0.0f, 0.0f, 1.0f);
    fillLight.emission = make_float3(100.0f, 110.0f, 120.0f);
    fillLight.size = make_float2(1.5f, 1.5f);
    fillLight.area = 2.25f;
    lightManager.addAreaLight(fillLight);
    sceneHierarchy.addAreaLight(1, "Fill Light");

    // Sync lights to GPU
    lightManager.syncToGpu(&optixEngine);

    std::cout << "[Main] Default lights enabled (1 directional, 2 area)\n";

    //--------------------------------------------------------------------------
    // Load HDR Environment Map
    //--------------------------------------------------------------------------

    EnvironmentMap environmentMap;

    // Search for HDR file
    if (hdrPath.empty()) {
        // Search for default HDR in assets/hdri/
        std::vector<std::filesystem::path> hdrSearchPaths = {
            exePath / "assets" / "hdri" / "default.hdr",
            exePath.parent_path() / "assets" / "hdri" / "default.hdr",
            std::filesystem::current_path() / "assets" / "hdri" / "default.hdr",
            // Also check for any .hdr file in the hdri directory
        };

        for (const auto& path : hdrSearchPaths) {
            if (std::filesystem::exists(path)) {
                hdrPath = path;
                std::cout << "[Main] Found default HDR: " << hdrPath << "\n";
                break;
            }
        }
    }

    // Load environment map if path is specified
    if (!hdrPath.empty() && std::filesystem::exists(hdrPath)) {
        if (environmentMap.loadFromFile(hdrPath.string())) {
            // Set environment map texture
            optixEngine.setEnvironmentMap(environmentMap.getTexture(), 1.0f);
            
            // Set importance sampling CDFs
            optixEngine.setEnvironmentCDF(
                environmentMap.getConditionalCDF(),
                environmentMap.getMarginalCDF(),
                environmentMap.getWidth(),
                environmentMap.getHeight(),
                environmentMap.getTotalLuminance()
            );
            
            std::cout << "[Main] Environment map loaded successfully\n";
        } else {
            std::cerr << "[Main] Failed to load environment map: " << hdrPath << "\n";
        }
    } else if (!hdrPath.empty()) {
        std::cerr << "[Main] HDR file not found: " << hdrPath << "\n";
    } else {
        std::cout << "[Main] No environment map specified, using default lighting\n";
    }

    // Build hierarchical scene tree now that model and lights are loaded
    uiManager.buildHierarchicalSceneTree();

    // Wire up light edit callback
    uiManager.setOnLightEdit([&](SceneNodeType type, uint32_t index, const ui::LightInfo& info) {
        switch (type) {
            case SceneNodeType::DirectionalLight:
                lightManager.updateDirectionalLight(index, info);
                break;
            case SceneNodeType::AreaLight:
                lightManager.updateAreaLight(index, info);
                break;
            case SceneNodeType::PointLight:
                lightManager.updatePointLight(index, info);
                break;
            default:
                break;
        }
        lightManager.syncToGpu(&optixEngine);
        optixEngine.resetAccumulation();
    });

    // Wire up light info request callback (for double-click to show properties)
    uiManager.setLightInfoRequestCallback([&](SceneNodeType type, uint32_t index) -> ui::LightInfo {
        switch (type) {
            case SceneNodeType::DirectionalLight:
                return lightManager.getDirectionalLightInfo(index);
            case SceneNodeType::AreaLight:
                return lightManager.getAreaLightInfo(index);
            case SceneNodeType::PointLight:
                return lightManager.getPointLightInfo(index);
            default:
                return ui::LightInfo{};
        }
    });

    // Wire up instance info request callback (for showing material/texture properties)
    uiManager.setInstanceInfoRequestCallback([&](uint32_t instanceId) -> ui::InstanceInfo {
        ui::InstanceInfo info = {};
        info.instanceId = instanceId;
        
        // Collect preview textures for UI rendering
        std::vector<cudaTextureObject_t> previewTextures;
        uint32_t texIndex = 0;
        
        // Get material handle for this instance
        MaterialHandle matHandle = sceneManager.getMaterialHandle(instanceId);
        if (matHandle != INVALID_MATERIAL_HANDLE) {
            info.materialIndex = matHandle;
            
            // Get material data
            const GpuMaterial* mat = materialManager.get(matHandle);
            if (mat) {
                info.baseColor = mat->baseColor;
                info.metallic = mat->metallic;
                info.roughness = mat->roughness;
                info.emissive = mat->emissive;
                
                // Gather textures and assign indices
                if (mat->baseColorTex != 0) {
                    info.hasBaseColorTex = true;
                    info.baseColorTexIndex = texIndex++;
                    previewTextures.push_back(mat->baseColorTex);
                }
                
                if (mat->normalTex != 0) {
                    info.hasNormalTex = true;
                    info.normalTexIndex = texIndex++;
                    previewTextures.push_back(mat->normalTex);
                }
                
                if (mat->metallicRoughnessTex != 0) {
                    info.hasMetallicRoughnessTex = true;
                    info.metallicRoughnessTexIndex = texIndex++;
                    previewTextures.push_back(mat->metallicRoughnessTex);
                }
                
                if (mat->emissiveTex != 0) {
                    info.hasEmissiveTex = true;
                    info.emissiveTexIndex = texIndex++;
                    previewTextures.push_back(mat->emissiveTex);
                }
            }
        }
        
        // Store preview textures in UIManager for rendering
        uiManager.setPreviewTextures(previewTextures);
        
        // Get model name from hierarchy
        info.modelName = "Instance " + std::to_string(instanceId);
        
        return info;
    });

    // Set initial dimensions
    optixEngine.setDimensions(glContext.getWidth(), glContext.getHeight());

    // Allocate accumulation buffer for progressive anti-aliasing
    float4* d_accumulationBuffer = nullptr;
    size_t accumulationBufferSize = glContext.getWidth() * glContext.getHeight() * sizeof(float4);
    cudaMalloc(reinterpret_cast<void**>(&d_accumulationBuffer), accumulationBufferSize);
    optixEngine.setAccumulationBuffer(d_accumulationBuffer);
    std::cout << "[Main] Accumulation buffer allocated (" << accumulationBufferSize / 1024 << " KB)\n";

    // Track previous camera state for accumulation reset
    CameraParams prevCameraParams = camera.getCameraParams();

    // Set up callbacks
    glfwSetKeyCallback(glContext.getWindow(), keyCallback);
    glfwSetCursorPosCallback(glContext.getWindow(), cursorPosCallback);
    glfwSetMouseButtonCallback(glContext.getWindow(), mouseButtonCallback);
    glfwSetScrollCallback(glContext.getWindow(), scrollCallback);

    // Set up pre-resize callback to unregister CUDA resources BEFORE buffers are invalidated
    glContext.setPreResizeCallback([&]() {
        // Ensure all CUDA work is complete before unregistering
        cudaDeviceSynchronize();
        // Must unregister PBOs before OpenGL recreates the buffers
        cudaInterop.unregisterPBO();
        cudaInterop.unregisterUIPBO();
    });

    // Set up resize callback to re-register AFTER buffers are recreated
    glContext.setResizeCallback([&](uint32_t width, uint32_t height) {
        std::cout << "[Main] Resize: " << width << "x" << height << "\n";

        // Ensure OpenGL has finished with the buffer before CUDA registers it
        glFinish();

        // Re-register new PBOs (old ones were unregistered in pre-resize callback)
        if (!cudaInterop.registerPBO(glContext.getPBO(), glContext.getBufferSize())) {
            std::cerr << "[Main] Failed to re-register PBO after resize\n";
        }
        if (!cudaInterop.registerUIPBO(glContext.getUIPBO(), glContext.getBufferSize())) {
            std::cerr << "[Main] Failed to re-register UI PBO after resize\n";
        }

        // Reallocate accumulation buffer
        if (d_accumulationBuffer) {
            cudaFree(d_accumulationBuffer);
        }
        accumulationBufferSize = width * height * sizeof(float4);
        cudaMalloc(reinterpret_cast<void**>(&d_accumulationBuffer), accumulationBufferSize);
        optixEngine.setAccumulationBuffer(d_accumulationBuffer);
        optixEngine.resetAccumulation();

        // Update OptiX dimensions
        optixEngine.setDimensions(width, height);

        // Update camera aspect ratio
        camera.setAspectRatio(static_cast<float>(width) / static_cast<float>(height));

        // Update UI dimensions
        uiManager.setScreenSize(width, height);
        
        // Force UI re-render after resize
        uiRenderer.invalidate();
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
    std::cout << "  F1-F4 - Quality mode (Fast/Balanced/High/Accurate)\n";
    std::cout << "  T     - Print timing info\n";
    std::cout << "  G     - Print GPU info\n";
    std::cout << "  C     - Print camera info\n";
    std::cout << "\n";
    std::cout << "[Main] UI Controls:\n";
    std::cout << "  H     - Toggle scene hierarchy panel\n";
    std::cout << "  P     - Toggle property panel\n";
    std::cout << "  L     - Toggle light/dark theme\n";
    std::cout << "  Ctrl+S- Save scene\n";
    std::cout << "  Ctrl+O- Load scene\n";
    std::cout << "  Double-click on tree node to show properties\n";
    std::cout << "\n";
    std::cout << "[Main] Current quality mode: " << getQualityModeName(g_qualityMode) << "\n\n";

    // Main render loop
    while (!glContext.shouldClose()) {
        g_timer.beginFrame();

        // Poll events
        glContext.pollEvents();

        // Update UI
        uiManager.update(static_cast<float>(g_timer.deltaTime));

        // Update camera only if UI didn't consume input
        if (!inputHandler.wasMouseConsumed() && g_mouseCaptured) {
            updateCamera(static_cast<float>(g_timer.deltaTime));
        } else if (!g_mouseCaptured) {
            // Still allow keyboard movement when mouse not captured and UI not active
            updateCamera(static_cast<float>(g_timer.deltaTime));
        }

        // Get current camera params
        CameraParams currentCameraParams = camera.getCameraParams();

        // Check if camera has changed (reset accumulation for anti-aliasing)
        bool cameraChanged = 
            currentCameraParams.position.x != prevCameraParams.position.x ||
            currentCameraParams.position.y != prevCameraParams.position.y ||
            currentCameraParams.position.z != prevCameraParams.position.z ||
            currentCameraParams.forward.x != prevCameraParams.forward.x ||
            currentCameraParams.forward.y != prevCameraParams.forward.y ||
            currentCameraParams.forward.z != prevCameraParams.forward.z ||
            currentCameraParams.fovY != prevCameraParams.fovY;

        if (cameraChanged) {
            optixEngine.resetAccumulation();
            prevCameraParams = currentCameraParams;
        }

        // Update OptiX camera params
        optixEngine.setCamera(currentCameraParams);

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

        //----------------------------------------------------------------------
        // Phase 4: Render UI
        //----------------------------------------------------------------------

        // Collect UI geometry
        uiManager.collectGeometry();

        // Set preview textures for UI rendering (only when changed)
        // Use the cache to avoid sampling full-res textures every frame
        if (uiManager.texturesChanged()) {
            const auto& previewTextures = uiManager.getPreviewTextures();
            
            // Generate cached previews (renders full textures to small cache once)
            texturePreviewCache.generatePreviews(
                previewTextures.data(),
                static_cast<uint32_t>(previewTextures.size()),
                cudaInterop.getStream()
            );
            
            // Wait for cache generation to complete
            cudaInterop.synchronize();
            
            // Use cached textures for UI rendering (fast path)
            uiRenderer.setTextures(
                texturePreviewCache.getCachedTextures(),
                texturePreviewCache.getCachedTextureCount()
            );
            
            uiManager.clearTexturesChanged();
        }

        // Map UI PBO for CUDA access
        float4* uiDevicePtr = reinterpret_cast<float4*>(cudaInterop.mapUIPBO());
        if (uiDevicePtr) {
            // Render UI only if geometry changed (major performance optimization)
            // When UI hasn't changed, we skip render, sync, and texture update
            bool rendered = uiRenderer.renderIfChanged(
                uiManager.getQuads(),
                uiManager.getGeometryGeneration(),
                fontAtlas.getTexture(),
                uiDevicePtr,
                glContext.getWidth(), glContext.getHeight(),
                cudaInterop.getStream());

            // Sync before unmap if we rendered (ensures CUDA writes complete)
            if (rendered) {
                cudaInterop.synchronize();
            }

            // Unmap UI PBO
            cudaInterop.unmapUIPBO();

            // Update texture from PBO if we rendered new content
            if (rendered) {
                glContext.updateUITextureFromPBO();
            }
        }

        // Render fullscreen quad (composites scene + UI)
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
    g_optixEngine = nullptr;
    g_uiManager = nullptr;
    g_inputHandler = nullptr;
    g_selectionManager = nullptr;
    g_sceneSerializer = nullptr;
    g_sceneManager = nullptr;
    g_sceneHierarchy = nullptr;
    g_lightManager = nullptr;

    // Shutdown UI
    inputHandler.shutdown();
    uiRenderer.shutdown();
    uiManager.shutdown();
    fontAtlas.release();

    // Free lighting buffers
    lightManager.cleanup();

    // Free accumulation buffer
    if (d_accumulationBuffer) {
        cudaFree(d_accumulationBuffer);
    }

    // Clear scene before managers are destroyed
    sceneManager.clear();

    std::cout << "[Main] Goodbye!\n";

    return 0;
}
