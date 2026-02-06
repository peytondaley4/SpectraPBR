#include "application.h"
#include "model_loader.h"
#include <glm/glm.hpp>
#include <iostream>

namespace spectra {

Application* Application::s_instance = nullptr;

Application::~Application() {
    shutdown();
}

bool Application::parseArgs(int argc, char* argv[]) {
    m_exePath = std::filesystem::absolute(argv[0]).parent_path();

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--remote" || arg == "-r") {
            m_config.remoteMode = true;
            m_remoteMode = true;
        } else if (m_config.modelPath.empty()) {
            m_config.modelPath = arg;
        } else if (m_config.hdrPath.empty()) {
            m_config.hdrPath = arg;
        }
    }

    // Set up paths
    m_shaderDir = m_exePath / "shaders";
    m_ptxDir = m_exePath / "optix_programs";
    m_fontsDir = m_exePath / "assets" / "fonts";

    if (!std::filesystem::exists(m_shaderDir)) {
        m_shaderDir = std::filesystem::current_path() / "shaders";
    }
    if (!std::filesystem::exists(m_ptxDir)) {
        m_ptxDir = std::filesystem::current_path() / "optix_programs";
    }
    if (!std::filesystem::exists(m_fontsDir)) {
        m_fontsDir = std::filesystem::current_path() / "assets" / "fonts";
    }

    return true;
}

bool Application::init() {
    std::cout << "=== SpectraPBR ===\n\n";

    s_instance = this;

    if (!initGraphics()) return false;
    if (!initCuda()) return false;
    if (!initOptix()) return false;
    if (!initUI()) return false;
    if (!loadScene()) return false;

    setupCallbacks();

    // Allocate accumulation buffer
    size_t bufferSize = m_glContext->getWidth() * m_glContext->getHeight() * sizeof(float4);
    cudaMalloc(reinterpret_cast<void**>(&m_accumulationBuffer), bufferSize);
    cudaMemset(m_accumulationBuffer, 0, bufferSize);
    m_optixEngine->setAccumulationBuffer(m_accumulationBuffer);

    m_prevCameraParams = m_camera->getCameraParams();
    m_running = true;

    printControls();

    return true;
}

bool Application::initGraphics() {
    m_glContext = std::make_unique<GLContext>();
    if (!m_glContext->init(m_config.width, m_config.height, "SpectraPBR")) {
        std::cerr << "[App] Failed to initialize OpenGL\n";
        return false;
    }

    if (!m_glContext->createDisplayResources(m_shaderDir)) {
        std::cerr << "[App] Failed to create display resources\n";
        return false;
    }

    return true;
}

bool Application::initCuda() {
    m_cudaInterop = std::make_unique<CudaInterop>();
    if (!m_cudaInterop->init()) {
        std::cerr << "[App] Failed to initialize CUDA\n";
        return false;
    }

    if (!m_cudaInterop->registerPBOs(
            m_glContext->getPBO(0), m_glContext->getPBO(1), m_glContext->getPBO(2),
            m_glContext->getBufferSize())) {
        std::cerr << "[App] Failed to register PBOs\n";
        return false;
    }

    if (!m_cudaInterop->registerUIPBO(m_glContext->getUIPBO(), m_glContext->getBufferSize())) {
        std::cerr << "[App] Failed to register UI PBO\n";
        return false;
    }

    return true;
}

bool Application::initOptix() {
    m_optixEngine = std::make_unique<OptixEngine>();
    if (!m_optixEngine->init(m_cudaInterop->getCudaContext())) {
        std::cerr << "[App] Failed to initialize OptiX\n";
        return false;
    }

    if (!m_optixEngine->createPipeline(m_ptxDir)) {
        std::cerr << "[App] Failed to create OptiX pipeline\n";
        return false;
    }

    m_optixEngine->setDimensions(m_glContext->getWidth(), m_glContext->getHeight());
    m_optixEngine->setQualityMode(m_qualityMode);

    return true;
}

bool Application::initUI() {
    m_fontAtlas = std::make_unique<text::FontAtlas>();
    if (!m_fontAtlas->load((m_fontsDir / "DejaVuSans.ttf").string(), 32.0f, 512, 8)) {
        std::cerr << "[App] Warning: Failed to load font\n";
    }

    m_uiManager = std::make_unique<ui::UIManager>();
    if (!m_uiManager->init(m_fontAtlas.get(), m_glContext->getWidth(), m_glContext->getHeight())) {
        std::cerr << "[App] Warning: Failed to initialize UI manager\n";
    }

    m_uiRenderer = std::make_unique<ui::UIRenderer>();
    if (!m_uiRenderer->init(4096)) {
        std::cerr << "[App] Warning: Failed to initialize UI renderer\n";
    }

    m_texturePreviewCache = std::make_unique<ui::TexturePreviewCache>();
    m_texturePreviewCache->init();

    m_inputHandler = std::make_unique<ui::InputHandler>();
    m_inputHandler->init(m_glContext->getWindow(), m_uiManager.get());

    m_selectionManager = std::make_unique<SelectionManager>();
    m_sceneSerializer = std::make_unique<SceneSerializer>();

    m_glContext->setUIEnabled(true);

    return true;
}

bool Application::loadScene() {
    // Initialize managers
    m_geometryManager = std::make_unique<GeometryManager>();
    m_textureManager = std::make_unique<TextureManager>();
    m_materialManager = std::make_unique<MaterialManager>();
    m_materialManager->setTextureManager(m_textureManager.get());

    m_sceneManager = std::make_unique<SceneManager>();
    m_sceneManager->setOptixEngine(m_optixEngine.get());
    m_sceneManager->setGeometryManager(m_geometryManager.get());
    m_sceneManager->setMaterialManager(m_materialManager.get());

    m_sceneHierarchy = std::make_unique<SceneHierarchy>();
    m_uiManager->setSceneHierarchy(m_sceneHierarchy.get());
    m_uiManager->setMaterialManager(m_materialManager.get());

    m_lightManager = std::make_unique<LightManager>();
    m_environmentMap = std::make_unique<EnvironmentMap>();

    m_camera = std::make_unique<Camera>();
    m_camera->setPosition(glm::vec3(0.0f, 1.0f, 5.0f));
    m_camera->setAspectRatio(static_cast<float>(m_glContext->getWidth()) /
                              static_cast<float>(m_glContext->getHeight()));

    // Load model if specified
    if (!m_config.modelPath.empty() && std::filesystem::exists(m_config.modelPath)) {
        ModelLoader loader;
        auto model = loader.load(m_config.modelPath);
        if (model) {
            std::cout << "[App] Loaded model: " << model->name << "\n";

            for (const auto& matData : model->materials) {
                m_materialManager->addMaterial(matData);
            }

            uint32_t modelNodeIdx = m_sceneHierarchy->addModel(model->name);
            uint32_t instanceId = 0;

            for (const auto& instance : model->instances) {
                if (instance.meshIndex < model->meshes.size()) {
                    const MeshData& mesh = model->meshes[instance.meshIndex];
                    uint32_t gasIndex = m_sceneManager->addMesh(mesh);
                    if (gasIndex != UINT32_MAX) {
                        m_sceneManager->addInstance(gasIndex, instance.transform);
                        std::string name = "Instance " + std::to_string(instanceId);
                        m_sceneHierarchy->addInstance(modelNodeIdx, instance.meshIndex, instanceId, name);
                        instanceId++;
                    }
                }
            }

            if (m_sceneManager->buildIAS()) {
                m_sceneManager->updateSBT();
                m_optixEngine->setSceneHandle(m_sceneManager->getSceneHandle());
                m_optixEngine->setGeometryBuffers(m_sceneManager->getVertexBuffers(),
                                                   m_sceneManager->getIndexBuffers());
            }
        }
    }

    // Set up default lighting and environment
    setupDefaultScene();

    // Wire up UI callbacks
    wireUICallbacks();

    m_uiManager->buildHierarchicalSceneTree();

    return true;
}

void Application::setupDefaultScene() {
    // Create default lights
    m_lightManager->createDefaultLights();

    // Add lights to hierarchy
    m_sceneHierarchy->addDirectionalLight(0, "Sun");
    m_sceneHierarchy->addAreaLight(0, "Key Light");
    m_sceneHierarchy->addAreaLight(1, "Fill Light");

    m_lightManager->syncToGpu(m_optixEngine.get(), m_cudaInterop->getStream());

    // Load environment map
    std::filesystem::path hdrPath = m_config.hdrPath;
    if (hdrPath.empty()) {
        std::vector<std::filesystem::path> searchPaths = {
            m_exePath / "assets" / "hdri" / "default.hdr",
            std::filesystem::current_path() / "assets" / "hdri" / "default.hdr",
        };
        for (const auto& path : searchPaths) {
            if (std::filesystem::exists(path)) {
                hdrPath = path;
                break;
            }
        }
    }

    if (!hdrPath.empty() && std::filesystem::exists(hdrPath)) {
        if (m_environmentMap->loadFromFile(hdrPath.string())) {
            m_optixEngine->setEnvironmentMap(m_environmentMap->getTexture(), 1.0f);
            m_optixEngine->setEnvironmentCDF(
                m_environmentMap->getConditionalCDF(),
                m_environmentMap->getMarginalCDF(),
                m_environmentMap->getWidth(),
                m_environmentMap->getHeight(),
                m_environmentMap->getTotalLuminance()
            );
            std::cout << "[App] Environment map loaded\n";
        }
    }
}

void Application::wireUICallbacks() {
    m_uiManager->setSelectionCallback([this](uint32_t instanceId) {
        m_selectionManager->setSelectedInstanceId(instanceId);
        m_optixEngine->setSelectedInstanceId(instanceId);
        m_optixEngine->resetAccumulation();
    });

    m_uiManager->setOnLightEdit([this](SceneNodeType type, uint32_t index, const ui::LightInfo& info) {
        switch (type) {
            case SceneNodeType::DirectionalLight:
                m_lightManager->updateDirectionalLight(index, info);
                break;
            case SceneNodeType::AreaLight:
                m_lightManager->updateAreaLight(index, info);
                break;
            case SceneNodeType::PointLight:
                m_lightManager->updatePointLight(index, info);
                break;
            default: break;
        }
        m_lightManager->syncToGpu(m_optixEngine.get(), m_cudaInterop->getStream());
        m_optixEngine->resetAccumulation();
    });

    m_uiManager->setLightInfoRequestCallback([this](SceneNodeType type, uint32_t index) -> ui::LightInfo {
        switch (type) {
            case SceneNodeType::DirectionalLight:
                return m_lightManager->getDirectionalLightInfo(index);
            case SceneNodeType::AreaLight:
                return m_lightManager->getAreaLightInfo(index);
            case SceneNodeType::PointLight:
                return m_lightManager->getPointLightInfo(index);
            default: return ui::LightInfo{};
        }
    });

    m_uiManager->setInstanceInfoRequestCallback([this](uint32_t instanceId) -> ui::InstanceInfo {
        ui::InstanceInfo info = {};
        info.instanceId = instanceId;

        std::vector<cudaTextureObject_t> previewTextures;
        uint32_t texIndex = 0;

        MaterialHandle matHandle = m_sceneManager->getMaterialHandle(instanceId);
        if (matHandle != INVALID_MATERIAL_HANDLE) {
            info.materialIndex = matHandle;
            const GpuMaterial* mat = m_materialManager->get(matHandle);
            if (mat) {
                info.baseColor = mat->baseColor;
                info.metallic = mat->metallic;
                info.roughness = mat->roughness;
                info.emissive = mat->emissive;

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

        m_uiManager->setPreviewTextures(previewTextures);
        info.modelName = "Instance " + std::to_string(instanceId);
        return info;
    });
}

void Application::setupCallbacks() {
    glfwSetWindowUserPointer(m_glContext->getWindow(), this);
    glfwSetKeyCallback(m_glContext->getWindow(), keyCallback);
    glfwSetCursorPosCallback(m_glContext->getWindow(), cursorPosCallback);
    glfwSetMouseButtonCallback(m_glContext->getWindow(), mouseButtonCallback);
    glfwSetScrollCallback(m_glContext->getWindow(), scrollCallback);

    m_glContext->setPreResizeCallback([this]() {
        cudaDeviceSynchronize();
        m_cudaInterop->unregisterPBOs();
        m_cudaInterop->unregisterUIPBO();
    });

    m_glContext->setResizeCallback([this](uint32_t width, uint32_t height) {
        glFinish();

        m_cudaInterop->registerPBOs(
            m_glContext->getPBO(0), m_glContext->getPBO(1), m_glContext->getPBO(2),
            m_glContext->getBufferSize());
        m_cudaInterop->registerUIPBO(m_glContext->getUIPBO(), m_glContext->getBufferSize());

        if (m_accumulationBuffer) cudaFree(m_accumulationBuffer);
        size_t bufferSize = width * height * sizeof(float4);
        cudaMalloc(reinterpret_cast<void**>(&m_accumulationBuffer), bufferSize);
        cudaMemset(m_accumulationBuffer, 0, bufferSize);
        m_optixEngine->setAccumulationBuffer(m_accumulationBuffer);
        m_optixEngine->resetAccumulation();
        m_optixEngine->setDimensions(width, height);

        m_camera->setAspectRatio(static_cast<float>(width) / static_cast<float>(height));
        m_uiManager->setScreenSize(width, height);
        m_uiRenderer->invalidate();

        m_writeIdx = 0;
        m_displayIdx = 0;
        m_framesPipelined = 0;
    });
}

void Application::run() {
    while (m_running && !m_glContext->shouldClose()) {
        m_timer.beginFrame();
        m_glContext->pollEvents();

        m_uiManager->update(static_cast<float>(m_timer.deltaTime));

        if (m_mouseCaptured || !m_inputHandler->wasMouseConsumed()) {
            updateCamera(static_cast<float>(m_timer.deltaTime));
        }

        // Check camera change
        CameraParams currentParams = m_camera->getCameraParams();
        if (cameraChanged(currentParams, m_prevCameraParams)) {
            m_optixEngine->resetAccumulation();
            m_prevCameraParams = currentParams;
        }
        m_optixEngine->setCamera(currentParams);

        renderFrame();

        m_timer.endFrame();

        // Print timing every 60 frames if diagnostics enabled
        if (m_diagEnabled && m_timer.frameCount - m_lastTimingPrint >= 60) {
            std::cout << "[Timing] " << m_timer.fps << " FPS, "
                      << m_timer.frameTimeMs << " ms/frame\n";
            m_lastTimingPrint = m_timer.frameCount;
        }
    }
}

bool Application::cameraChanged(const CameraParams& a, const CameraParams& b) {
    return a.position.x != b.position.x || a.position.y != b.position.y || a.position.z != b.position.z ||
           a.forward.x != b.forward.x || a.forward.y != b.forward.y || a.forward.z != b.forward.z ||
           a.fovY != b.fovY;
}

void Application::renderFrame() {
    m_displayIdx = (m_writeIdx + 1) % 3;

    // Render scene
    float4* devicePtr = reinterpret_cast<float4*>(m_cudaInterop->mapBuffer(m_writeIdx));
    if (!devicePtr) return;

    m_optixEngine->render(devicePtr, m_cudaInterop->getStream());
    m_cudaInterop->recordRenderComplete(m_writeIdx);
    m_cudaInterop->unmapBuffer(m_writeIdx);

    if (m_framesPipelined >= 2) {
        if (!m_cudaInterop->isRenderComplete(m_displayIdx)) {
            m_cudaInterop->waitForRender(m_displayIdx);
        }
        m_glContext->updateTextureFromPBO(m_displayIdx);
    }

    m_writeIdx = (m_writeIdx + 1) % 3;
    m_framesPipelined++;

    // Render UI
    m_uiManager->collectGeometry();

    if (m_uiManager->texturesChanged()) {
        const auto& textures = m_uiManager->getPreviewTextures();
        m_texturePreviewCache->generatePreviews(
            textures.data(), static_cast<uint32_t>(textures.size()),
            m_cudaInterop->getUIStream());
        m_cudaInterop->synchronizeUI();
        m_uiRenderer->setTextures(
            m_texturePreviewCache->getCachedTextures(),
            m_texturePreviewCache->getCachedTextureCount());
        m_uiManager->clearTexturesChanged();
    }

    float4* uiPtr = reinterpret_cast<float4*>(m_cudaInterop->mapUIPBO());
    if (uiPtr) {
        bool rendered = m_uiRenderer->renderIfChanged(
            m_uiManager->getQuads(), m_uiManager->getGeometryGeneration(),
            m_fontAtlas->getTexture(), uiPtr,
            m_glContext->getWidth(), m_glContext->getHeight(),
            m_cudaInterop->getUIStream());

        if (rendered) m_cudaInterop->synchronizeUI();
        m_cudaInterop->unmapUIPBO();
        if (rendered) m_glContext->updateUITextureFromPBO();
    }

    // Display
    if (m_framesPipelined >= 2) {
        m_glContext->renderFullscreenQuad(m_displayIdx);
    } else {
        m_cudaInterop->synchronize();
        m_glContext->updateTextureFromPBO((m_writeIdx + 2) % 3);
        m_glContext->renderFullscreenQuad((m_writeIdx + 2) % 3);
    }

    m_glContext->swapBuffers();
}

void Application::updateCamera(float deltaTime) {
    float forward = 0.0f, right = 0.0f, up = 0.0f;
    if (m_keyW) forward += 1.0f;
    if (m_keyS) forward -= 1.0f;
    if (m_keyD) right += 1.0f;
    if (m_keyA) right -= 1.0f;
    if (m_keyE) up += 1.0f;
    if (m_keyQ) up -= 1.0f;
    m_camera->processKeyboard(forward, right, up, deltaTime, m_keyShift);
}

void Application::shutdown() {
    m_running = false;
    s_instance = nullptr;

    if (m_inputHandler) m_inputHandler->shutdown();
    if (m_uiRenderer) m_uiRenderer->shutdown();
    if (m_uiManager) m_uiManager->shutdown();
    if (m_fontAtlas) m_fontAtlas->release();
    if (m_sceneManager) m_sceneManager->clear();

    if (m_accumulationBuffer) {
        cudaFree(m_accumulationBuffer);
        m_accumulationBuffer = nullptr;
    }
}

void Application::printControls() {
    std::cout << "\n[Controls]\n";
    std::cout << "  ESC      - Quit\n";
    std::cout << "  TAB      - Toggle mouse capture\n";
    std::cout << "  WASD/QE  - Move camera\n";
    std::cout << "  Shift    - Sprint\n";
    std::cout << "  V        - Toggle VSync\n";
    std::cout << "  F        - Toggle Fullscreen\n";
    std::cout << "  T        - Print frame timing\n";
    std::cout << "  F5       - Toggle continuous timing display\n";
    std::cout << "  1-4      - Resolution presets\n";
    std::cout << "  F1-F4    - Quality modes\n";
    std::cout << "  [ ]      - Decrease/Increase SPP\n";
    std::cout << "  H        - Toggle hierarchy panel\n";
    std::cout << "  P        - Toggle property panel\n";
    std::cout << "\n";
}

// Static callbacks
void Application::keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    (void)scancode;
    auto* app = static_cast<Application*>(glfwGetWindowUserPointer(window));
    if (!app) return;

    if (action == GLFW_PRESS || action == GLFW_RELEASE) {
        bool pressed = (action == GLFW_PRESS);
        switch (key) {
            case GLFW_KEY_W: app->m_keyW = pressed; break;
            case GLFW_KEY_S: app->m_keyS = pressed; break;
            case GLFW_KEY_A: app->m_keyA = pressed; break;
            case GLFW_KEY_D: app->m_keyD = pressed; break;
            case GLFW_KEY_Q: app->m_keyQ = pressed; break;
            case GLFW_KEY_E: app->m_keyE = pressed; break;
            case GLFW_KEY_LEFT_SHIFT:
            case GLFW_KEY_RIGHT_SHIFT:
                app->m_keyShift = pressed; break;
            default: break;
        }
    }

    if (action != GLFW_PRESS) return;

    switch (key) {
        case GLFW_KEY_ESCAPE:
            if (app->m_mouseCaptured) {
                app->m_mouseCaptured = false;
                if (!app->m_remoteMode) {
                    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
                }
            } else {
                glfwSetWindowShouldClose(window, GLFW_TRUE);
            }
            break;

        case GLFW_KEY_TAB:
            app->m_mouseCaptured = !app->m_mouseCaptured;
            if (!app->m_remoteMode) {
                glfwSetInputMode(window, GLFW_CURSOR,
                    app->m_mouseCaptured ? GLFW_CURSOR_DISABLED : GLFW_CURSOR_NORMAL);
            }
            app->m_firstMouse = true;
            break;

        case GLFW_KEY_V:
            app->m_glContext->setVSync(!app->m_glContext->isVSyncEnabled());
            break;

        case GLFW_KEY_F:
            app->m_glContext->toggleFullscreen();
            break;

        case GLFW_KEY_1:
            app->m_glContext->setResolution(1280, 720);
            app->m_camera->setAspectRatio(1280.0f / 720.0f);
            break;
        case GLFW_KEY_2:
            app->m_glContext->setResolution(1920, 1080);
            app->m_camera->setAspectRatio(1920.0f / 1080.0f);
            break;
        case GLFW_KEY_3:
            app->m_glContext->setResolution(2560, 1440);
            app->m_camera->setAspectRatio(2560.0f / 1440.0f);
            break;
        case GLFW_KEY_4:
            app->m_glContext->setResolution(3840, 2160);
            app->m_camera->setAspectRatio(3840.0f / 2160.0f);
            break;

        case GLFW_KEY_F1:
            app->m_qualityMode = QUALITY_FAST;
            app->m_optixEngine->setQualityMode(QUALITY_FAST);
            break;
        case GLFW_KEY_F2:
            app->m_qualityMode = QUALITY_BALANCED;
            app->m_optixEngine->setQualityMode(QUALITY_BALANCED);
            break;
        case GLFW_KEY_F3:
            app->m_qualityMode = QUALITY_HIGH;
            app->m_optixEngine->setQualityMode(QUALITY_HIGH);
            break;
        case GLFW_KEY_F4:
            app->m_qualityMode = QUALITY_ACCURATE;
            app->m_optixEngine->setQualityMode(QUALITY_ACCURATE);
            break;

        case GLFW_KEY_LEFT_BRACKET:
            if (app->m_optixEngine) {
                uint32_t spp = app->m_optixEngine->getSamplesPerPixel();
                app->m_optixEngine->setSamplesPerPixel(spp > 1 ? spp / 2 : 1);
            }
            break;
        case GLFW_KEY_RIGHT_BRACKET:
            if (app->m_optixEngine) {
                uint32_t spp = app->m_optixEngine->getSamplesPerPixel();
                app->m_optixEngine->setSamplesPerPixel(spp < 64 ? spp * 2 : 64);
            }
            break;

        case GLFW_KEY_T:
            std::cout << "[Timing] Frame: " << app->m_timer.frameTimeMs << " ms, "
                      << "FPS: " << app->m_timer.fps << " (avg over 60 frames)\n";
            break;

        case GLFW_KEY_F5:
            app->m_diagEnabled = !app->m_diagEnabled;
            std::cout << "[Timing] Continuous display: " << (app->m_diagEnabled ? "ON" : "OFF") << "\n";
            break;

        case GLFW_KEY_H:
            app->m_uiManager->toggleScenePanel();
            break;
        case GLFW_KEY_P:
            app->m_uiManager->togglePropertyPanel();
            break;
        case GLFW_KEY_L:
            if (!(mods & GLFW_MOD_CONTROL)) {
                app->m_uiManager->toggleTheme();
            }
            break;

        case GLFW_KEY_S:
            if (mods & GLFW_MOD_CONTROL) {
                std::string path = SceneSerializer::getAutoSavePath();
                bool dark = app->m_uiManager->isDarkTheme();
                app->m_sceneSerializer->saveScene(path, app->m_camera.get(),
                    app->m_sceneManager.get(), app->m_qualityMode, dark);
            }
            break;

        default: break;
    }
}

void Application::cursorPosCallback(GLFWwindow* window, double xpos, double ypos) {
    auto* app = static_cast<Application*>(glfwGetWindowUserPointer(window));
    if (!app) return;

    float2 pos = make_float2(static_cast<float>(xpos), static_cast<float>(ypos));

    if (!app->m_mouseCaptured && app->m_uiManager) {
        app->m_uiManager->handleMouseMove(pos);
    }

    if (app->m_mouseCaptured && app->m_camera) {
        if (app->m_firstMouse) {
            app->m_lastMouseX = xpos;
            app->m_lastMouseY = ypos;
            app->m_firstMouse = false;
            return;
        }

        float dx = static_cast<float>(xpos - app->m_lastMouseX);
        float dy = static_cast<float>(ypos - app->m_lastMouseY);
        app->m_lastMouseX = xpos;
        app->m_lastMouseY = ypos;
        app->m_camera->processMouseMovement(dx, dy);
    } else {
        app->m_lastMouseX = xpos;
        app->m_lastMouseY = ypos;
    }
}

void Application::mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
    (void)mods;
    auto* app = static_cast<Application*>(glfwGetWindowUserPointer(window));
    if (!app) return;

    double xpos, ypos;
    glfwGetCursorPos(window, &xpos, &ypos);
    float2 pos = make_float2(static_cast<float>(xpos), static_cast<float>(ypos));

    if (!app->m_mouseCaptured && app->m_uiManager) {
        if (action == GLFW_PRESS && app->m_uiManager->handleMouseDown(pos, button)) return;
        if (action == GLFW_RELEASE && app->m_uiManager->handleMouseUp(pos, button)) return;
    }

    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS && !app->m_mouseCaptured) {
        uint32_t picked = app->m_optixEngine->pickInstance(
            static_cast<uint32_t>(xpos), static_cast<uint32_t>(ypos));
        app->m_uiManager->setSelectedInstanceId(picked);
        app->m_optixEngine->setSelectedInstanceId(picked);
        app->m_optixEngine->resetAccumulation();
    }

    if (button == GLFW_MOUSE_BUTTON_RIGHT) {
        if (action == GLFW_PRESS) {
            app->m_mouseCaptured = true;
            if (!app->m_remoteMode) {
                glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
            }
            app->m_firstMouse = true;
        } else if (action == GLFW_RELEASE) {
            app->m_mouseCaptured = false;
            if (!app->m_remoteMode) {
                glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
            }
        }
    }
}

void Application::scrollCallback(GLFWwindow* window, double xoffset, double yoffset) {
    (void)xoffset;
    auto* app = static_cast<Application*>(glfwGetWindowUserPointer(window));
    if (!app) return;

    if (!app->m_mouseCaptured && app->m_uiManager) {
        double xpos, ypos;
        glfwGetCursorPos(window, &xpos, &ypos);
        float2 pos = make_float2(static_cast<float>(xpos), static_cast<float>(ypos));
        if (app->m_uiManager->handleMouseScroll(pos, static_cast<float>(yoffset))) return;
    }

    if (app->m_camera) {
        app->m_camera->processMouseScroll(static_cast<float>(yoffset));
    }
}

} // namespace spectra
