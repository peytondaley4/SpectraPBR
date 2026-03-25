#include "application.h"
#include "hemisphere_vis.h"
#include "model_loader.h"
#include <glm/glm.hpp>
#include <algorithm>
#include <iostream>
#include <chrono>
#include <cstring>
#include <cmath>

namespace spectra {

Application* Application::s_instance = nullptr;

Application::Application() = default;

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

    // Sparse multi-res path guide grid (collision-free)
    m_pathGuideGrid = std::make_unique<PathGuideGrid>();
    PathGuideGridConfig gridConfig;
    gridConfig.num_levels = 8;
    gridConfig.base_resolution = 16;
    gridConfig.per_level_scale = 2.0f;
    gridConfig.entry_stride = PATH_GUIDE_ENTRY_STRIDE_DEFAULT;

    // Use actual scene bounds (with small padding) so the grid covers
    // the entire scene.  Default [-10,10] is far too small for most models.
    glm::vec3 extent = m_sceneMax - m_sceneMin;
    float pad = glm::length(extent) * 0.05f;  // 5% padding
    if (pad > 0.0f) {
        gridConfig.bounds_min[0] = m_sceneMin.x - pad;
        gridConfig.bounds_min[1] = m_sceneMin.y - pad;
        gridConfig.bounds_min[2] = m_sceneMin.z - pad;
        gridConfig.bounds_max[0] = m_sceneMax.x + pad;
        gridConfig.bounds_max[1] = m_sceneMax.y + pad;
        gridConfig.bounds_max[2] = m_sceneMax.z + pad;
    }

    if (!m_pathGuideGrid->init(gridConfig)) {
        std::cerr << "[App] Path guide grid init failed (non-fatal)\n";
        m_pathGuideGrid.reset();
    } else {
        // Initialize async readback pipeline (double-buffered, non-blocking builds)
        if (!m_pathGuideGrid->initAsync()) {
            std::cerr << "[App] Path guide async init failed (falling back to sync)\n";
        }
    }

    // Initialize wireframe renderer for grid debug visualization
    m_wireframeRenderer = std::make_unique<WireframeRenderer>();
    if (!m_wireframeRenderer->init(m_shaderDir)) {
        std::cerr << "[App] Wireframe renderer init failed (non-fatal)\n";
        m_wireframeRenderer.reset();
    }
    // Always add the debug panel when we have UI (so G key shows it); use grid data if initialized
    if (m_uiManager) {
        uint32_t numLevels = 8;
        uint32_t totalCells = 0;
        uint32_t entryStride = PATH_GUIDE_ENTRY_STRIDE_DEFAULT;
        uint32_t baseResolution = 16;
        float perLevelScale = 2.0f;
        float boundsMin[3] = { -10.0f, -10.0f, -10.0f };
        float boundsMax[3] = {  10.0f,  10.0f,  10.0f };
        if (m_pathGuideGrid && m_pathGuideGrid->isInitialized()) {
            SparsePathGuideDescriptor d = m_pathGuideGrid->getDescriptor();
            numLevels = d.num_levels;
            totalCells = m_pathGuideGrid->getTotalCells();
            entryStride = d.entry_stride;
            baseResolution = d.base_resolution;
            perLevelScale = d.per_level_scale;
            boundsMin[0] = d.bounds_min[0]; boundsMin[1] = d.bounds_min[1]; boundsMin[2] = d.bounds_min[2];
            boundsMax[0] = d.bounds_max[0]; boundsMax[1] = d.bounds_max[1]; boundsMax[2] = d.bounds_max[2];
        }
        m_uiManager->addPathGuideGridDebugPanel(
            numLevels, totalCells, entryStride,
            baseResolution, perLevelScale,
            boundsMin, boundsMax,
            m_debugGridVisualize,
            [this](bool v) { m_debugGridVisualize = v; },
            [this](uint32_t l) {
                m_debugGridLevel = l;
                if (m_wireframeRenderer && m_wireframeRenderer->isInitialized() &&
                    m_pathGuideGrid && m_pathGuideGrid->hasSparseData()) {
                    auto vertices = m_pathGuideGrid->generateEdgeVertices(l);
                    m_wireframeRenderer->updateVertices(vertices);
                }
            },
            nullptr,  // No manual build button (auto-build handles it)
            [this](bool enabled) {
                // "Enable Guiding" toggle -> Running / Disabled
                if (m_optixEngine) {
                    if (enabled) {
                        m_pathGuideMode = PathGuideMode::Running;
                        m_pathGuideTrainingFrameCount = 0;
                        m_optixEngine->setPathGuideEnabled(true);
                        m_optixEngine->setPathGuideDebugEnabled(true);
                    } else {
                        m_pathGuideMode = PathGuideMode::Disabled;
                        m_optixEngine->setPathGuideEnabled(false);
                        m_optixEngine->setPathGuideDebugEnabled(false);
                    }
                    // Lighting model changed — reset accumulation to avoid blending
                    // guided and non-guided frames
                    m_optixEngine->resetAccumulation();
                }
            },
            [this]() {
                // "Pause" callback
                if (m_pathGuideMode == PathGuideMode::Running) {
                    m_pathGuideMode = PathGuideMode::Paused;
                    // Mode change reflected in UI status panel
                }
            },
            [this]() {
                // "Build & Step" callback
                if (m_pathGuideGrid && m_pathGuideGrid->isInitialized()) {
                    m_pathGuideMode = PathGuideMode::StepOnce;
                    if (m_optixEngine) {
                        m_optixEngine->setPathGuideEnabled(true);
                        m_optixEngine->setPathGuideDebugEnabled(true);
                    }
                    // Mode change reflected in UI status panel
                }
            });
    }

    // Cell inspector panel
    if (m_uiManager) {
        m_uiManager->addCellInspectorPanel();
    }

    // Hemisphere visualization
    m_hemisphereVis = std::make_unique<HemisphereVis>();
    if (!m_hemisphereVis->init(m_shaderDir)) {
        std::cerr << "[App] Hemisphere vis init failed (non-fatal)\n";
        m_hemisphereVis.reset();
    }

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
        std::vector<std::filesystem::path> filesToLoad;

        if (std::filesystem::is_directory(m_config.modelPath)) {
            // Directory mode: scan for all .obj files
            std::cout << "[App] Scanning directory for OBJ files: " << m_config.modelPath << "\n";
            for (const auto& entry : std::filesystem::directory_iterator(m_config.modelPath)) {
                if (!entry.is_regular_file()) continue;
                std::string ext = entry.path().extension().string();
                if (ext == ".obj" || ext == ".OBJ") {
                    filesToLoad.push_back(entry.path());
                }
            }
            std::sort(filesToLoad.begin(), filesToLoad.end());
            std::cout << "[App] Found " << filesToLoad.size() << " OBJ files\n";
        } else {
            filesToLoad.push_back(m_config.modelPath);
        }

        ModelLoader loader;
        uint32_t globalInstanceId = 0;
        bool anyLoaded = false;

        // Track scene bounding box for camera auto-fit
        glm::vec3 sceneMin(std::numeric_limits<float>::max());
        glm::vec3 sceneMax(std::numeric_limits<float>::lowest());

        for (const auto& filePath : filesToLoad) {
            auto model = loader.load(filePath);
            if (!model) continue;

            std::cout << "[App] Loaded model: " << model->name << "\n";

            // Track material index offset for this model
            uint32_t materialOffset = static_cast<uint32_t>(m_materialManager->getMaterialCount());

            for (const auto& matData : model->materials) {
                m_materialManager->addMaterial(matData);
            }

            // Compute per-mesh local AABBs, then transform by instance transforms
            // to get correct world-space scene bounds (handles scale/rotation/translation).
            std::vector<glm::vec3> meshMins(model->meshes.size(), glm::vec3(std::numeric_limits<float>::max()));
            std::vector<glm::vec3> meshMaxs(model->meshes.size(), glm::vec3(std::numeric_limits<float>::lowest()));
            for (size_t mi = 0; mi < model->meshes.size(); mi++) {
                for (const auto& vert : model->meshes[mi].vertices) {
                    glm::vec3 p(vert.position.x, vert.position.y, vert.position.z);
                    meshMins[mi] = glm::min(meshMins[mi], p);
                    meshMaxs[mi] = glm::max(meshMaxs[mi], p);
                }
            }
            for (const auto& instance : model->instances) {
                if (instance.meshIndex >= model->meshes.size()) continue;
                const glm::vec3& mMin = meshMins[instance.meshIndex];
                const glm::vec3& mMax = meshMaxs[instance.meshIndex];
                const float* t = instance.transform;
                for (int c = 0; c < 8; c++) {
                    float cx = (c & 1) ? mMax.x : mMin.x;
                    float cy = (c & 2) ? mMax.y : mMin.y;
                    float cz = (c & 4) ? mMax.z : mMin.z;
                    glm::vec3 w(t[0]*cx + t[1]*cy + t[2]*cz + t[3],
                                t[4]*cx + t[5]*cy + t[6]*cz + t[7],
                                t[8]*cx + t[9]*cy + t[10]*cz + t[11]);
                    sceneMin = glm::min(sceneMin, w);
                    sceneMax = glm::max(sceneMax, w);
                }
            }

            uint32_t modelNodeIdx = m_sceneHierarchy->addModel(model->name);

            for (const auto& instance : model->instances) {
                if (instance.meshIndex < model->meshes.size()) {
                    // Adjust material index to account for previously loaded materials
                    MeshData mesh = model->meshes[instance.meshIndex];
                    mesh.materialIndex += materialOffset;

                    uint32_t gasIndex = m_sceneManager->addMesh(mesh);
                    if (gasIndex != UINT32_MAX) {
                        m_sceneManager->addInstance(gasIndex, instance.transform);
                        std::string name = "Instance " + std::to_string(globalInstanceId);
                        m_sceneHierarchy->addInstance(modelNodeIdx, instance.meshIndex, globalInstanceId, name);
                        globalInstanceId++;

                        // Extract area light from emissive meshes
                        const auto& srcMesh = model->meshes[instance.meshIndex];
                        uint32_t origMatIdx = srcMesh.materialIndex;
                        if (origMatIdx < model->materials.size()) {
                            const float3& em = model->materials[origMatIdx].emissive;
                            if (em.x + em.y + em.z > 0.01f) {
                                const float* t = instance.transform;
                                const auto& verts = srcMesh.vertices;
                                const auto& idxs  = srcMesh.indices;

                                // Accumulate centroid, area-weighted normal, and total area
                                glm::vec3 centroid(0.0f);
                                glm::vec3 weightedNormal(0.0f);
                                float totalArea = 0.0f;

                                for (size_t ti = 0; ti + 2 < idxs.size(); ti += 3) {
                                    glm::vec3 lp[3];
                                    for (int vi = 0; vi < 3; vi++) {
                                        const float3& p = verts[idxs[ti + vi]].position;
                                        // Transform to world space (3x4 row-major)
                                        lp[vi] = glm::vec3(
                                            t[0]*p.x + t[1]*p.y + t[2]*p.z  + t[3],
                                            t[4]*p.x + t[5]*p.y + t[6]*p.z  + t[7],
                                            t[8]*p.x + t[9]*p.y + t[10]*p.z + t[11]);
                                    }
                                    glm::vec3 e1 = lp[1] - lp[0];
                                    glm::vec3 e2 = lp[2] - lp[0];
                                    glm::vec3 cross = glm::cross(e1, e2);
                                    float triArea = glm::length(cross) * 0.5f;
                                    totalArea += triArea;
                                    weightedNormal += cross; // length = 2*area, acts as area weight
                                    centroid += (lp[0] + lp[1] + lp[2]) * triArea;
                                }

                                if (totalArea > 1e-8f) {
                                    centroid /= (totalArea * 3.0f);
                                    glm::vec3 avgNormal = glm::normalize(weightedNormal);

                                    // Build tangent perpendicular to normal
                                    glm::vec3 up = (std::abs(avgNormal.y) < 0.99f)
                                        ? glm::vec3(0, 1, 0) : glm::vec3(1, 0, 0);
                                    glm::vec3 tangent = glm::normalize(glm::cross(up, avgNormal));

                                    float sideLen = std::sqrt(totalArea);

                                    GpuAreaLight light;
                                    light.position = make_float3(centroid.x, centroid.y, centroid.z);
                                    light.normal   = make_float3(avgNormal.x, avgNormal.y, avgNormal.z);
                                    light.tangent  = make_float3(tangent.x, tangent.y, tangent.z);
                                    light.emission = em;
                                    light.area     = totalArea;
                                    light.size     = make_float2(sideLen, sideLen);

                                    uint32_t lightIdx = static_cast<uint32_t>(m_lightManager->getAreaLightCount());
                                    m_lightManager->addAreaLight(light);
                                    std::string lightName = "Emissive " + std::to_string(lightIdx);
                                    m_sceneHierarchy->addAreaLight(lightIdx, lightName);

                                    std::cout << "[App] Emissive mesh -> area light (emission: ["
                                              << em.x << ", " << em.y << ", " << em.z
                                              << "], area: " << totalArea << ")\n";
                                }
                            }
                        }
                    }
                }
            }

            anyLoaded = true;
        }

        if (anyLoaded && m_sceneManager->buildIAS()) {
            m_sceneManager->updateSBT();
            m_optixEngine->setSceneHandle(m_sceneManager->getSceneHandle());
            m_optixEngine->setGeometryBuffers(m_sceneManager->getVertexBuffers(),
                                               m_sceneManager->getIndexBuffers());

            // Auto-fit camera to scene bounding box
            glm::vec3 center = (sceneMin + sceneMax) * 0.5f;
            glm::vec3 extent = sceneMax - sceneMin;
            float maxExtent = std::max({extent.x, extent.y, extent.z});

            if (maxExtent > 0.0f) {
                // Pull back along +Z from center so camera looks into the scene
                float halfFovRad = glm::radians(m_camera->getFOV() * 0.5f);
                float distance = (maxExtent * 0.5f) / std::tan(halfFovRad) * 1.2f;

                // Find the thinnest axis — for a Cornell Box that's typically the
                // depth axis (the open face). Place the camera on the side of the
                // thinnest extent so it looks *into* the box.
                glm::vec3 camPos = center;
                float yaw = -90.0f; // default: looking along -Z

                if (extent.x <= extent.y && extent.x <= extent.z) {
                    // Thinnest along X — place camera on +X looking -X
                    camPos.x = sceneMax.x + distance * 0.3f;
                    yaw = 180.0f;
                } else if (extent.z <= extent.x && extent.z <= extent.y) {
                    // Thinnest along Z — place camera on +Z looking -Z
                    camPos.z = sceneMax.z + distance * 0.3f;
                    yaw = -90.0f;
                } else {
                    // Thinnest along Y (unusual) — fall back to +Z
                    camPos.z = sceneMax.z + distance * 0.3f;
                    yaw = -90.0f;
                }

                m_camera->setPosition(camPos);
                m_camera->setYawPitch(yaw, 0.0f);

                // Scale move speed and clip planes to scene size
                m_camera->setMoveSpeed(maxExtent * 0.5f);
                float sceneRadius = glm::length(extent) * 0.5f;
                float nearPlane = std::max(0.001f, sceneRadius * 0.0001f);
                float farPlane = std::max(1000.0f, sceneRadius * 20.0f);
                m_camera->setClipPlanes(nearPlane, farPlane);

                // Store scene bounds for path guide grid initialization
                m_sceneMin = sceneMin;
                m_sceneMax = sceneMax;

                std::cout << "[App] Scene bounds: ("
                          << sceneMin.x << ", " << sceneMin.y << ", " << sceneMin.z << ") -> ("
                          << sceneMax.x << ", " << sceneMax.y << ", " << sceneMax.z << ")\n";
                std::cout << "[App] Camera auto-fit to: ("
                          << camPos.x << ", " << camPos.y << ", " << camPos.z << ")\n";
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
        // Only reset accumulation when guiding is off — avoid disrupting converged image
        if (m_pathGuideMode != PathGuideMode::Running && m_pathGuideMode != PathGuideMode::Paused) {
            m_optixEngine->resetAccumulation();
        }
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
        resetPathGuideTraining();
    });

    m_uiManager->setOnMaterialEdit([this](uint32_t instanceId, const GpuMaterial& material) {
        // Find which material this instance uses
        MaterialHandle matHandle = m_sceneManager->getMaterialHandle(instanceId);
        if (matHandle == INVALID_MATERIAL_HANDLE) return;

        // Update the material's scalar properties (preserves texture handles)
        m_materialManager->updateMaterial(matHandle, material);

        // Synchronize render stream before SBT rebuild — updateSBT frees and
        // reallocates GPU memory, which is unsafe while optixLaunch is in flight.
        m_cudaInterop->synchronize();

        // Rebuild SBT so the GPU sees the new material values
        m_sceneManager->updateSBT();

        // Reset accumulation so the new material is immediately visible
        m_optixEngine->resetAccumulation();
        resetPathGuideTraining();
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
        // Skip resize if window is minimized (0x0) — no valid buffers to create,
        // and division by zero would occur in aspect ratio / pixel_world_size.
        if (width == 0 || height == 0) return;

        glFinish();

        m_cudaInterop->registerPBOs(
            m_glContext->getPBO(0), m_glContext->getPBO(1), m_glContext->getPBO(2),
            m_glContext->getBufferSize());
        m_cudaInterop->registerUIPBO(m_glContext->getUIPBO(), m_glContext->getBufferSize());

        if (m_accumulationBuffer) cudaFree(m_accumulationBuffer);
        size_t bufferSize = static_cast<size_t>(width) * height * sizeof(float4);
        cudaMalloc(reinterpret_cast<void**>(&m_accumulationBuffer), bufferSize);
        cudaMemset(m_accumulationBuffer, 0, bufferSize);
        m_optixEngine->setAccumulationBuffer(m_accumulationBuffer);
        m_optixEngine->resetAccumulation();
        m_optixEngine->setDimensions(width, height);
        resetPathGuideTraining();

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

            // Reset training frame count on camera move (fresh samples for new view)
            if (m_pathGuideMode == PathGuideMode::Running && m_pathGuideGrid && m_pathGuideGrid->isInitialized()) {
                m_pathGuideTrainingFrameCount = 0;
            }
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

void Application::resetPathGuideTraining() {
    if (!m_pathGuideGrid || !m_pathGuideGrid->isInitialized()) return;
    if (m_pathGuideMode == PathGuideMode::Disabled) return;

    cudaStream_t stream = m_cudaInterop ? m_cudaInterop->getStream() : nullptr;
    m_pathGuideGrid->clear(stream);  // Zero vMF lobes + stats (structure preserved)
    m_pathGuideTrainingFrameCount = 0;
}

bool Application::cameraChanged(const CameraParams& a, const CameraParams& b) {
    return a.position.x != b.position.x || a.position.y != b.position.y || a.position.z != b.position.z ||
           a.forward.x != b.forward.x || a.forward.y != b.forward.y || a.forward.z != b.forward.z ||
           a.fovY != b.fovY;
}

void Application::renderFrame() {
    // ── Frame timing diagnostic (prints every 120 frames) ──
    static uint64_t diagFrameCount = 0;
    static double diagAccum[10] = {};
    static const char* diagNames[10] = {
        "buildChk", "syncRdr", "uiRender", "glDisp", "swapBuf",
        "pgSetup", "mapBuf", "render", "unmapEvt", "asyncIO"
    };
    auto diagT = []{
        return std::chrono::high_resolution_clock::now();
    };
    auto diagMs = [](auto a, auto b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };
    auto t0 = diagT();
    auto tPrev = t0;

    m_displayIdx = (m_writeIdx + 1) % 3;

    // ═══ PHASE 1: BUILD THREAD CHECK (render stream is idle, safe for sync) ═══
    // finishBuildFromReadback does hashing, sorting, vMF fitting (10-300ms CPU).
    // Running it on a background thread lets the GPU render continuously.

    // Step 1: Check if background build thread completed
    if (m_buildThreadActive) {
        if (m_buildFuture.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
            bool success = m_buildFuture.get();
            m_buildThreadActive = false;

            if (success) {
                cudaStream_t renderStream = m_cudaInterop ? m_cudaInterop->getStream() : nullptr;
                m_pathGuideGrid->swapGrids(renderStream);
                m_pathGuideBuildInFlight = false;
                m_pathGuideTotalBuilds++;
                m_pathGuideTrainingFrameCount = 0;

                if (m_debugGridVisualize && m_wireframeRenderer && m_wireframeRenderer->isInitialized()) {
                    auto vertices = m_pathGuideGrid->generateEdgeVerticesAllLevels();
                    m_wireframeRenderer->updateVertices(vertices);
                }
                if (m_uiManager) {
                    auto d = m_pathGuideGrid->getDescriptor();
                    m_uiManager->updatePathGuideGridStats(d.num_levels, m_pathGuideGrid->getTotalCells(), d.entry_stride);
                }

                // Run adaptive refinement every 5 builds
                uint32_t currentFrame = m_optixEngine ? m_optixEngine->getFrameIndex() : 0;
                if (m_pathGuideTotalBuilds > 0 && m_pathGuideTotalBuilds % 5 == 0) {
                    cudaStream_t stream = m_cudaInterop ? m_cudaInterop->getStream() : nullptr;
                    if (m_pathGuideGrid->runRefinementPass(currentFrame, stream)) {
                        if (m_debugGridVisualize && m_wireframeRenderer && m_wireframeRenderer->isInitialized()) {
                            auto vertices = m_pathGuideGrid->generateEdgeVerticesAllLevels();
                            m_wireframeRenderer->updateVertices(vertices);
                        }
                        if (m_uiManager) {
                            auto d = m_pathGuideGrid->getDescriptor();
                            m_uiManager->updatePathGuideGridStats(d.num_levels, m_pathGuideGrid->getTotalCells(), d.entry_stride);
                        }
                    }
                }

                // If StepOnce, transition to Paused after build
                if (m_pathGuideMode == PathGuideMode::StepOnce) {
                    m_pathGuideMode = PathGuideMode::Paused;
                }
            } else {
                m_pathGuideBuildInFlight = false;
            }
        }
    }

    // Step 2: Poll readback and launch build thread (only if no thread running)
    if (!m_buildThreadActive && m_pathGuideBuildInFlight &&
        m_pathGuideGrid && m_pathGuideGrid->pollAsyncReadback()) {
        m_buildFuture = std::async(std::launch::async, [this]() {
            return m_pathGuideGrid->finishBuildFromReadback();
        });
        m_buildThreadActive = true;
    }

    // Update UI status and optionally print stats when running
    if (m_pathGuideMode != PathGuideMode::Disabled && m_optixEngine) {
        m_pathGuideStatsFrame++;
        if (m_pathGuideStatsFrame >= 60) {
            m_pathGuideStatsFrame = 0;
            m_optixEngine->resetPathGuideStats(m_cudaInterop ? m_cudaInterop->getStream() : nullptr);

            // Update automation status in UI
            if (m_uiManager) {
                const char* modeStr = "Disabled";
                switch (m_pathGuideMode) {
                    case PathGuideMode::Running:  modeStr = "Running"; break;
                    case PathGuideMode::Paused:   modeStr = "Paused"; break;
                    case PathGuideMode::StepOnce: modeStr = "StepOnce"; break;
                    default: break;
                }
                m_uiManager->updatePathGuideAutomationStatus(modeStr, m_pathGuideTrainingFrameCount, m_pathGuideTotalBuilds);
            }
        }
    }

    // [DIAG] buildChk
    { auto tNow = diagT(); diagAccum[0] += diagMs(tPrev, tNow); tPrev = tNow; }

    // ═══ PHASE 2: DISPLAY PREVIOUS FRAME ═══
    // On Windows WDDM, glTexSubImage2D on a CUDA-registered PBO triggers an
    // implicit full-device sync. cudaStreamSynchronize is much faster because it
    // returns as soon as the GPU finishes, without WDDM driver overhead.
    // This wait IS the GPU render time — it's the minimum possible wait.
    m_cudaInterop->synchronize();

    // [DIAG] syncRdr (time spent waiting for previous frame's GPU render)
    { auto tNow = diagT(); diagAccum[1] += diagMs(tPrev, tNow); tPrev = tNow; }

    if (m_framesPipelined >= 2) {
        m_glContext->updateTextureFromPBO(m_displayIdx);
    } else if (m_framesPipelined == 1) {
        int warmupIdx = (m_writeIdx + 2) % 3;
        m_glContext->updateTextureFromPBO(warmupIdx);
        m_displayIdx = warmupIdx;
    }

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

        // unmapUIPBO provides GPU-GPU sync via cudaGraphicsUnmapResources —
        // OpenGL safely reads PBO after unmap, no explicit CPU sync needed
        m_cudaInterop->unmapUIPBO();
        if (rendered) m_glContext->updateUITextureFromPBO();
    }

    // [DIAG] uiRender
    { auto tNow = diagT(); diagAccum[2] += diagMs(tPrev, tNow); tPrev = tNow; }

    // GL display
    if (m_framesPipelined >= 1) {
        m_glContext->renderFullscreenQuad(m_displayIdx);
    }

    // Render grid wireframe overlay (same viewport and view/proj as scene)
    // Drawn AFTER scene but BEFORE UI overlay so it doesn't obscure panels
    if (m_debugGridVisualize && m_wireframeRenderer && m_wireframeRenderer->isInitialized() &&
        m_wireframeRenderer->getVertexCount() > 0) {
        uint32_t w = m_glContext->getWidth();
        uint32_t h = m_glContext->getHeight();
        glViewport(0, 0, static_cast<GLsizei>(w), static_cast<GLsizei>(h));
        glm::mat4 view = m_camera->getViewMatrix();
        const float wireframeNear = 0.0001f;
        glm::mat4 proj = glm::perspective(
            glm::radians(m_camera->getFOV()),
            m_camera->getAspectRatio(),
            wireframeNear,
            m_camera->getFarPlane());
        glm::mat4 viewProj = proj * view;
        m_wireframeRenderer->render(viewProj, glm::vec3(0.2f, 0.8f, 1.0f));
    }

    // Hemisphere visualization inset (top-right corner)
    // Shows 3 hemispheres: combined mixture (large) + lobe 0 and lobe 1 (smaller, below)
    if (m_hemisphereVis && m_inspectedCell.valid) {
        uint32_t w = m_glContext->getWidth();
        uint32_t h = m_glContext->getHeight();
        float insetSize = 160.0f;
        float insetX = static_cast<float>(w) - insetSize - 10.0f;
        float insetY = static_cast<float>(h) - insetSize - 10.0f;
        m_hemisphereVis->render(insetX, insetY, insetSize, w, h);
    }

    // UI overlay drawn last so it's on top of wireframe and other overlays
    m_glContext->renderUIOverlay();

    // [DIAG] glDisp
    { auto tNow = diagT(); diagAccum[3] += diagMs(tPrev, tNow); tPrev = tNow; }

    m_glContext->swapBuffers();

    // [DIAG] swapBuf
    { auto tNow = diagT(); diagAccum[4] += diagMs(tPrev, tNow); tPrev = tNow; }

    // ═══ PHASE 3: PATH GUIDE SETUP ═══
    m_buildThisFrame = false;
    uint32_t savedSpp = 0;
    if (m_pathGuideGrid && m_pathGuideGrid->isInitialized()) {
        // Determine if we should build this frame (only if no build in flight)
        if (!m_pathGuideBuildInFlight) {
            if (m_pathGuideMode == PathGuideMode::Running &&
                m_pathGuideTrainingFrameCount >= m_pathGuideAutoBuildInterval) {
                m_buildThisFrame = true;
            } else if (m_pathGuideMode == PathGuideMode::StepOnce) {
                m_buildThisFrame = true;
            }
        }

        // Use render descriptor (stable, not being mutated by build)
        SparsePathGuideDescriptor sparseDesc = m_pathGuideGrid->getRenderDescriptor();
        PathGuideStagingDescriptor stagingDesc = m_pathGuideGrid->getStagingDescriptor();
        m_optixEngine->setPathGuideGridDescriptor(&sparseDesc, &stagingDesc);
        m_optixEngine->setPathGuideGridDebug(m_debugGridVisualize, m_debugGridLevel);
        const auto& config = m_pathGuideGrid->getConfig();
        m_optixEngine->setPathGuideLevelConfig(config.start_level, config.min_level, config.max_level);

        m_optixEngine->setPathGuideNoJitter(false);
        if (m_buildThisFrame) {
            savedSpp = m_optixEngine->getSamplesPerPixel();
            m_optixEngine->setSamplesPerPixel(1);
        }

        // Increment training frame counter when actively training
        if (m_pathGuideMode == PathGuideMode::Running || m_pathGuideMode == PathGuideMode::StepOnce) {
            m_pathGuideTrainingFrameCount++;
        }
    } else {
        m_optixEngine->setPathGuideGridDescriptor(nullptr, nullptr);
    }

    // [DIAG] pgSetup
    { auto tNow = diagT(); diagAccum[5] += diagMs(tPrev, tNow); tPrev = tNow; }

    // ═══ PHASE 4: RENDER SUBMISSION (async on GPU) ═══
    // Map PBO first — render stream is idle from GL's perspective, near-instant
    float4* devicePtr = reinterpret_cast<float4*>(m_cudaInterop->mapBuffer(m_writeIdx));
    if (!devicePtr) return;

    // [DIAG] mapBuf
    { auto tNow = diagT(); diagAccum[6] += diagMs(tPrev, tNow); tPrev = tNow; }

    m_optixEngine->render(devicePtr, m_cudaInterop->getStream());

    // Restore SPP if we changed it for the build
    if (savedSpp > 0) {
        m_optixEngine->setSamplesPerPixel(savedSpp);
    }

    // [DIAG] render
    { auto tNow = diagT(); diagAccum[7] += diagMs(tPrev, tNow); tPrev = tNow; }

    // Record event + unmap: both stream-ordered, no CPU block
    m_cudaInterop->recordRenderComplete(m_writeIdx);
    m_cudaInterop->unmapBuffer(m_writeIdx);

    // [DIAG] unmapEvt
    { auto tNow = diagT(); diagAccum[8] += diagMs(tPrev, tNow); tPrev = tNow; }

    // ═══ PHASE 5: ASYNC I/O + ADVANCE ═══
    // Kick off async readback after this frame's trace has written to staging
    if (m_buildThisFrame && m_pathGuideGrid && m_pathGuideGrid->isInitialized() && !m_pathGuideBuildInFlight) {
        cudaStream_t stream = m_cudaInterop ? m_cudaInterop->getStream() : nullptr;
        uint32_t currentFrame = m_optixEngine ? m_optixEngine->getFrameIndex() : 0;
        m_pathGuideGrid->beginAsyncReadback(stream, currentFrame);
        m_pathGuideBuildInFlight = true;
    }

    m_writeIdx = (m_writeIdx + 1) % 3;
    m_framesPipelined++;

    // [DIAG] asyncIO
    { auto tNow = diagT(); diagAccum[9] += diagMs(tPrev, tNow); tPrev = tNow; }

    // Print timing every 120 frames
    diagFrameCount++;
    if (diagFrameCount % 120 == 0) {
        double total = 0;
        for (int i = 0; i < 10; i++) total += diagAccum[i];
        std::cout << "[DIAG] avg ms/frame over 120 frames (total="
                  << (total / 120.0) << "ms):\n";
        for (int i = 0; i < 10; i++) {
            std::cout << "  " << diagNames[i] << ": "
                      << (diagAccum[i] / 120.0) << " ms ("
                      << (100.0 * diagAccum[i] / total) << "%)\n";
            diagAccum[i] = 0;
        }
    }
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

    // Wait for any in-flight build thread before destroying resources
    if (m_buildThreadActive && m_buildFuture.valid()) {
        m_buildFuture.wait();
        m_buildThreadActive = false;
    }

    if (m_hemisphereVis) m_hemisphereVis->shutdown();
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
    std::cout << "  G        - Toggle grid debug panel\n";
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
        case GLFW_KEY_G:
            app->m_uiManager->toggleGridDebugPanel();
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
        PickResultBuffer pickResult = app->m_optixEngine->pickInstanceAndPosition(
            static_cast<uint32_t>(xpos), static_cast<uint32_t>(ypos));

        // Only update selection highlight and reset accumulation when guiding is off.
        // When guiding is active, clicks inspect cells without disrupting the scene.
        bool guidingActive = (app->m_pathGuideMode == PathGuideMode::Running ||
                              app->m_pathGuideMode == PathGuideMode::Paused);
        if (!guidingActive) {
            app->m_uiManager->setSelectedInstanceId(pickResult.instanceId);
            app->m_optixEngine->setSelectedInstanceId(pickResult.instanceId);
            app->m_optixEngine->resetAccumulation();
        }

        // Cell inspector: look up grid cell at hit position
        if (pickResult.instanceId != UINT32_MAX && app->m_pathGuideGrid &&
            app->m_pathGuideGrid->hasSparseData()) {
            const auto& cfg = app->m_pathGuideGrid->getConfig();
            std::cout << "[CellInspect] Hit pos: (" << pickResult.hitX << ", "
                      << pickResult.hitY << ", " << pickResult.hitZ
                      << ") Bounds: [" << cfg.bounds_min[0] << ".." << cfg.bounds_max[0]
                      << ", " << cfg.bounds_min[1] << ".." << cfg.bounds_max[1]
                      << ", " << cfg.bounds_min[2] << ".." << cfg.bounds_max[2]
                      << "] Cells: " << app->m_pathGuideGrid->getTotalCells() << "\n";
            auto cellResult = app->m_pathGuideGrid->inspectCellAtPosition(
                pickResult.hitX, pickResult.hitY, pickResult.hitZ);

            InspectedCellInfo info = {};
            info.worldPos[0] = pickResult.hitX;
            info.worldPos[1] = pickResult.hitY;
            info.worldPos[2] = pickResult.hitZ;

            if (cellResult.found) {
                info.valid = true;
                info.level = cellResult.level;
                info.ix = cellResult.ix;
                info.iy = cellResult.iy;
                info.iz = cellResult.iz;
                std::memcpy(info.cellAABBMin, cellResult.aabbMin, sizeof(float) * 3);
                std::memcpy(info.cellAABBMax, cellResult.aabbMax, sizeof(float) * 3);
                // vMF lobes
                info.theta0 = cellResult.data[0];
                info.phi0 = cellResult.data[1];
                info.kappa0 = cellResult.data[2];
                info.theta1 = cellResult.data[3];
                info.phi1 = cellResult.data[4];
                info.kappa1 = cellResult.data[5];
                // Stats (offsets 6-11)
                info.sumW = cellResult.data[9];
                float sumX = cellResult.data[6], sumY = cellResult.data[7], sumZ = cellResult.data[8];
                float meanLen = (info.sumW > 1e-9f)
                    ? std::sqrt(sumX*sumX + sumY*sumY + sumZ*sumZ) / info.sumW
                    : 0.0f;
                info.variance = 1.0f - meanLen;
                info.pi0 = cellResult.data[10];
                info.lastFrame = cellResult.data[11];

                // Determine subdivision/coarsening status
                const auto& config = app->m_pathGuideGrid->getConfig();
                info.wouldSubdivide = (cellResult.level < config.max_level &&
                    info.sumW >= config.subdivide_sample_threshold &&
                    info.variance > config.subdivide_variance_threshold);
                info.wouldCoarsen = (cellResult.level > config.min_level &&
                    app->m_optixEngine->getFrameIndex() > static_cast<uint32_t>(info.lastFrame) + config.coarsen_frames_threshold);
            }

            app->m_inspectedCell = info;
            app->m_uiManager->updateCellInspectorData(info);

            // Update hemisphere vis if available
            if (app->m_hemisphereVis && info.valid) {
                app->m_hemisphereVis->update(
                    info.theta0, info.phi0, info.kappa0,
                    info.theta1, info.phi1, info.kappa1,
                    info.pi0);
            }
        }
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
