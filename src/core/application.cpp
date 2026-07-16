#include "application.h"
#include "hemisphere_vis.h"
#include "model_loader.h"
#include "log.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <algorithm>
#include <iostream>
#include <chrono>
#include <cstring>
#include <cmath>

namespace spectra {

//------------------------------------------------------------------------------
// Transform helpers: compose/decompose a 3x4 row-major matrix from TRS
//------------------------------------------------------------------------------

// Compose: T = Translate * RotZ * RotY * RotX * Scale (row-major 3x4)
static void composeTransform(const float3& scale, const float3& translation,
                             const float3& rotDeg, float out[12]) {
    constexpr float DEG2RAD = 3.14159265358979f / 180.0f;
    float cx = std::cos(rotDeg.x * DEG2RAD), sx = std::sin(rotDeg.x * DEG2RAD);
    float cy = std::cos(rotDeg.y * DEG2RAD), sy = std::sin(rotDeg.y * DEG2RAD);
    float cz = std::cos(rotDeg.z * DEG2RAD), sz = std::sin(rotDeg.z * DEG2RAD);

    // Rotation = Rz * Ry * Rx (intrinsic XYZ)
    float r00 = cy * cz;
    float r01 = cz * sx * sy - cx * sz;
    float r02 = sx * sz + cx * cz * sy;
    float r10 = cy * sz;
    float r11 = cx * cz + sx * sy * sz;
    float r12 = cx * sy * sz - cz * sx;
    float r20 = -sy;
    float r21 = cy * sx;
    float r22 = cx * cy;

    // Row-major 3x4: each row is [Rx*S, Ry*S, Rz*S, T]
    out[0]  = r00 * scale.x;  out[1]  = r01 * scale.y;  out[2]  = r02 * scale.z;  out[3]  = translation.x;
    out[4]  = r10 * scale.x;  out[5]  = r11 * scale.y;  out[6]  = r12 * scale.z;  out[7]  = translation.y;
    out[8]  = r20 * scale.x;  out[9]  = r21 * scale.y;  out[10] = r22 * scale.z;  out[11] = translation.z;
}

// Decompose a 3x4 row-major matrix into scale, translation, euler (degrees)
static void decomposeTransform(const float transform[12], float3& scale,
                               float3& translation, float3& rotDeg) {
    // Translation is column 3
    translation = make_float3(transform[3], transform[7], transform[11]);

    // Scale = length of each column of the 3x3 sub-matrix
    float sx = std::sqrt(transform[0]*transform[0] + transform[4]*transform[4] + transform[8]*transform[8]);
    float sy = std::sqrt(transform[1]*transform[1] + transform[5]*transform[5] + transform[9]*transform[9]);
    float sz = std::sqrt(transform[2]*transform[2] + transform[6]*transform[6] + transform[10]*transform[10]);
    scale = make_float3(sx, sy, sz);

    // Rotation matrix (columns normalized)
    float r00 = transform[0] / sx, r01 = transform[1] / sy, r02 = transform[2] / sz;
    float r10 = transform[4] / sx, r11 = transform[5] / sy, r12 = transform[6] / sz;
    float r20 = transform[8] / sx, r21 = transform[9] / sy, r22 = transform[10] / sz;

    constexpr float RAD2DEG = 180.0f / 3.14159265358979f;
    // Extract euler XYZ (intrinsic) from rotation matrix
    float sy_val = -r20;
    if (sy_val >= 1.0f) {
        rotDeg.y = 90.0f;
        rotDeg.x = std::atan2(r01, r11) * RAD2DEG;
        rotDeg.z = 0.0f;
    } else if (sy_val <= -1.0f) {
        rotDeg.y = -90.0f;
        rotDeg.x = std::atan2(-r01, r11) * RAD2DEG;
        rotDeg.z = 0.0f;
    } else {
        rotDeg.y = std::asin(sy_val) * RAD2DEG;
        rotDeg.x = std::atan2(r21, r22) * RAD2DEG;
        rotDeg.z = std::atan2(r10, r00) * RAD2DEG;
    }
}

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
        } else if (arg == "--guide" || arg == "-g") {
            m_config.enableGuiding = true;
        } else if (arg == "--test-move-light" && i + 1 < argc) {
            // frame,dx,dy,dz — self-test: translate the first mesh-light
            // instance at the given frame via the UI edit path
            float v[4];
            if (sscanf(argv[++i], "%f,%f,%f,%f", &v[0], &v[1], &v[2], &v[3]) == 4) {
                m_config.testMoveFrame = static_cast<int>(v[0]);
                m_config.testMoveDelta[0] = v[1];
                m_config.testMoveDelta[1] = v[2];
                m_config.testMoveDelta[2] = v[3];
            }
        } else if (arg == "--cam" && i + 1 < argc) {
            // --cam px,py,pz[,tx,ty,tz]
            float v[6];
            int n = sscanf(argv[++i], "%f,%f,%f,%f,%f,%f",
                           &v[0], &v[1], &v[2], &v[3], &v[4], &v[5]);
            if (n >= 3) {
                m_config.hasCamera = true;
                m_config.camPos[0] = v[0]; m_config.camPos[1] = v[1]; m_config.camPos[2] = v[2];
                if (n >= 6) {
                    m_config.hasTarget = true;
                    m_config.camTarget[0] = v[3]; m_config.camTarget[1] = v[4]; m_config.camTarget[2] = v[5];
                }
            }
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

    // Denoiser AOV buffers
    cudaMalloc(reinterpret_cast<void**>(&m_aovAlbedoBuffer), bufferSize);
    cudaMalloc(reinterpret_cast<void**>(&m_aovNormalBuffer), bufferSize);
    cudaMemset(m_aovAlbedoBuffer, 0, bufferSize);
    cudaMemset(m_aovNormalBuffer, 0, bufferSize);
    m_optixEngine->setAOVBuffers(m_aovAlbedoBuffer, m_aovNormalBuffer);

    // OptiX AI denoiser (HDR, albedo+normal guides)
    m_denoiser = std::make_unique<OptixDenoiserWrapper>();
    if (!m_denoiser->init(m_optixEngine->getContext())) {
        std::cerr << "[App] Denoiser init failed, denoising disabled\n";
        m_denoiser.reset();
    } else {
        m_denoiser->resize(m_glContext->getWidth(), m_glContext->getHeight());
    }

    // Sparse multi-res path guide grid (device-resident cell table:
    // shaders allocate cells on first touch, no CPU build pipeline)
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
    }

    // --guide: enable guiding from frame 0 (same effect as the UI toggle) —
    // lets unattended/scripted runs exercise the guided path.
    if (m_config.enableGuiding && m_pathGuideGrid && m_pathGuideGrid->isInitialized()) {
        m_pathGuideMode = PathGuideMode::Running;
        m_pathGuideTrainingFrameCount = 0;
        m_optixEngine->setPathGuideEnabled(true);
        m_optixEngine->setPathGuideDebugEnabled(true);
        std::cout << "[App] Path guiding enabled from startup (--guide)\n";
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
                    m_pathGuideGrid && m_pathGuideGrid->refreshHostMirror()) {
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

        // Mesh-light extraction: world-space triangles of emissive meshes
        // (3 float4 per triangle; tri[0].w = per-light cumulative area CDF,
        // tri[1].w = triangle area) and instance -> light index map.
        std::vector<float4> lightTris;
        std::vector<std::pair<uint32_t, uint32_t>> instanceLightPairs; // (instanceId, lightIdx)

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
                        uint32_t thisInstanceId = globalInstanceId;
                        m_sceneManager->addInstance(gasIndex, instance.transform);
                        std::string name = "Instance " + std::to_string(thisInstanceId);
                        m_sceneHierarchy->addInstance(modelNodeIdx, instance.meshIndex, thisInstanceId, name);
                        globalInstanceId++;

                        // Mesh light from emissive meshes: keep the ACTUAL
                        // triangles so NEE samples the real surface (the old
                        // centroid+rectangle proxy sampled points that are not
                        // on the mesh) and so the emission can be MIS-paired
                        // with BSDF hits on the same geometry. Meshes with an
                        // emissive TEXTURE are skipped: NEE would evaluate the
                        // wrong (constant) emission, so those emit via path
                        // hits only — unbiased, just noisier.
                        const auto& srcMesh = model->meshes[instance.meshIndex];
                        uint32_t origMatIdx = srcMesh.materialIndex;
                        if (origMatIdx < model->materials.size() &&
                            model->materials[origMatIdx].emissiveTexPath.empty()) {
                            const float3& em = model->materials[origMatIdx].emissive;
                            if (em.x + em.y + em.z > 0.01f) {
                                const float* t = instance.transform;
                                const auto& verts = srcMesh.vertices;
                                const auto& idxs  = srcMesh.indices;

                                uint32_t triOffset = static_cast<uint32_t>(lightTris.size() / 3);
                                size_t firstTri = lightTris.size();

                                glm::vec3 centroid(0.0f);
                                glm::vec3 weightedNormal(0.0f);
                                float totalArea = 0.0f;
                                // Object-space copies of the pushed triangles,
                                // kept so transform edits can re-bake the
                                // world-space NEE geometry (see
                                // refreshMeshLightGeometry).
                                std::vector<float3> objVerts;

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
                                    glm::vec3 cr = glm::cross(e1, e2);
                                    float triArea = glm::length(cr) * 0.5f;
                                    if (triArea <= 1e-12f) continue;
                                    totalArea += triArea;
                                    weightedNormal += cr;
                                    centroid += (lp[0] + lp[1] + lp[2]) * triArea;

                                    lightTris.push_back(make_float4(lp[0].x, lp[0].y, lp[0].z, 0.0f)); // w = CDF (fixed up below)
                                    lightTris.push_back(make_float4(lp[1].x, lp[1].y, lp[1].z, triArea));
                                    lightTris.push_back(make_float4(lp[2].x, lp[2].y, lp[2].z, 0.0f));
                                    for (int vi = 0; vi < 3; vi++) {
                                        objVerts.push_back(verts[idxs[ti + vi]].position);
                                    }
                                }

                                uint32_t triCount = static_cast<uint32_t>((lightTris.size() - firstTri) / 3);
                                if (totalArea > 1e-8f && triCount > 0) {
                                    // Per-light cumulative area CDF in tri[0].w
                                    float cumulative = 0.0f;
                                    for (uint32_t ti = 0; ti < triCount; ti++) {
                                        cumulative += lightTris[firstTri + ti * 3 + 1].w / totalArea;
                                        lightTris[firstTri + ti * 3].w = cumulative;
                                    }
                                    lightTris[firstTri + (triCount - 1) * 3].w = 1.0f;

                                    centroid /= (totalArea * 3.0f);
                                    glm::vec3 avgNormal = glm::normalize(weightedNormal);
                                    glm::vec3 up = (std::abs(avgNormal.y) < 0.99f)
                                        ? glm::vec3(0, 1, 0) : glm::vec3(1, 0, 0);
                                    glm::vec3 tangent = glm::normalize(glm::cross(up, avgNormal));
                                    float sideLen = std::sqrt(totalArea);

                                    GpuAreaLight light = {};
                                    light.position = make_float3(centroid.x, centroid.y, centroid.z);
                                    light.normal   = make_float3(avgNormal.x, avgNormal.y, avgNormal.z);
                                    light.tangent  = make_float3(tangent.x, tangent.y, tangent.z);
                                    light.emission = em;
                                    light.area     = totalArea;
                                    light.size     = make_float2(sideLen, sideLen);
                                    light.triOffset = triOffset;
                                    light.triCount  = triCount;
                                    light.instanceId = thisInstanceId;

                                    uint32_t lightIdx = static_cast<uint32_t>(m_lightManager->getAreaLightCount());
                                    m_lightManager->addAreaLight(light);
                                    instanceLightPairs.push_back({ thisInstanceId, lightIdx });
                                    m_meshLightSources[thisInstanceId] =
                                        MeshLightSource{ lightIdx, triOffset, std::move(objVerts) };
                                    std::string lightName = "Emissive " + std::to_string(lightIdx);
                                    m_sceneHierarchy->addAreaLight(lightIdx, lightName);

                                    std::cout << "[App] Emissive mesh -> mesh light (emission: ["
                                              << em.x << ", " << em.y << ", " << em.z
                                              << "], area: " << totalArea
                                              << ", tris: " << triCount << ")\n";
                                } else {
                                    // No usable triangles — drop the partial data
                                    lightTris.resize(firstTri);
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
            // Per-instance transforms + material slots for raygen-side shading
            m_optixEngine->setInstanceData(
                m_sceneManager->getInstanceTransforms(),
                m_sceneManager->getInstanceNormalTransforms(),
                m_sceneManager->getInstanceMaterialIndices());

            // Mesh-light triangle buffer + instance -> light index map (for
            // NEE <-> BSDF MIS on emissive geometry)
            if (!lightTris.empty()) {
                cudaMalloc(reinterpret_cast<void**>(&m_areaLightTris),
                           lightTris.size() * sizeof(float4));
                cudaMemcpy(m_areaLightTris, lightTris.data(),
                           lightTris.size() * sizeof(float4), cudaMemcpyHostToDevice);
                m_optixEngine->setAreaLightTriangles(m_areaLightTris);
            }
            {
                std::vector<uint32_t> instanceLightIndices(
                    m_sceneManager->getInstanceCount(), UINT32_MAX);
                for (const auto& pair : instanceLightPairs) {
                    if (pair.first < instanceLightIndices.size()) {
                        instanceLightIndices[pair.first] = pair.second;
                    }
                }
                if (!instanceLightIndices.empty()) {
                    cudaMalloc(reinterpret_cast<void**>(&m_instanceLightIndices),
                               instanceLightIndices.size() * sizeof(uint32_t));
                    cudaMemcpy(m_instanceLightIndices, instanceLightIndices.data(),
                               instanceLightIndices.size() * sizeof(uint32_t),
                               cudaMemcpyHostToDevice);
                    m_optixEngine->setInstanceLightIndices(m_instanceLightIndices);
                }
            }

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

                if (m_config.hasCamera) {
                    // --cam override: fixed start pose for scripted runs
                    camPos = glm::vec3(m_config.camPos[0], m_config.camPos[1], m_config.camPos[2]);
                    if (m_config.hasTarget) {
                        glm::vec3 dir = glm::normalize(glm::vec3(
                            m_config.camTarget[0], m_config.camTarget[1], m_config.camTarget[2]) - camPos);
                        yaw = glm::degrees(std::atan2(dir.z, dir.x));
                        float pitch = glm::degrees(std::asin(glm::clamp(dir.y, -1.0f, 1.0f)));
                        m_camera->setPosition(camPos);
                        m_camera->setYawPitch(yaw, pitch);
                    } else {
                        m_camera->setPosition(camPos);
                        m_camera->setYawPitch(yaw, 0.0f);
                    }
                } else {
                    m_camera->setPosition(camPos);
                    m_camera->setYawPitch(yaw, 0.0f);
                }

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
    // The default Sun/Key/Fill rig exists ONLY for the empty no-model
    // startup (so the app isn't a black window). Any LOADED model renders
    // under its own lighting: its emissive materials (mesh lights,
    // registered in loadScene) and/or the environment map — never the
    // default rig, which would wash out the transport (caustics, hard
    // shadows) real scenes exist to exercise.
    if (m_sceneManager->getInstanceCount() == 0) {
        m_lightManager->createDefaultLights();
        m_sceneHierarchy->addDirectionalLight(0, "Sun");
        m_sceneHierarchy->addAreaLight(0, "Key Light");
        m_sceneHierarchy->addAreaLight(1, "Fill Light");
    } else {
        std::cout << "[App] Model loaded - skipping default Sun/Key/Fill ("
                  << m_lightManager->getAreaLightCount()
                  << " emissive mesh light(s) from the scene)\n";
    }

    m_lightManager->syncToGpu(m_optixEngine.get(), m_cudaInterop->getStream());

    // Load environment map. The DEFAULT fallback HDRI is skipped when the
    // scene brought its own emissive lighting — an ambient env wash would
    // drown exactly the transport (caustics, hard shadows) such scenes
    // exercise. An explicitly passed hdrPath always loads.
    std::filesystem::path hdrPath = m_config.hdrPath;
    if (hdrPath.empty()) {
        if (m_lightManager->getAreaLightCount() > 0 &&
            m_lightManager->getDirectionalLightCount() == 0) {
            std::cout << "[App] Scene has its own lights - skipping default HDRI\n";
        } else {
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
    }

    if (!hdrPath.empty() && std::filesystem::exists(hdrPath)) {
        if (m_environmentMap->loadFromFile(hdrPath.string())) {
            const float envIntensity = 1.0f;
            m_optixEngine->setEnvironmentMap(m_environmentMap->getTexture(), envIntensity);
            m_optixEngine->setEnvironmentImportance(
                m_environmentMap->getAliasProb(),
                m_environmentMap->getAliasIdx(),
                m_environmentMap->getPmf(),
                m_environmentMap->getWidth(),
                m_environmentMap->getHeight(),
                m_environmentMap->getTotalLuminance()
            );
            // Selection weight of the environment in the NEE light pick:
            // total incident radiance integral. The CDF total is
            // sum(lum * sin(theta)) over texels; the solid-angle integral is
            // that times 2*pi^2 / (W*H) (equirectangular texel solid angle).
            float w = m_environmentMap->getWidth() > 0 && m_environmentMap->getHeight() > 0
                ? envIntensity * m_environmentMap->getTotalLuminance() *
                  (2.0f * 3.14159265f * 3.14159265f) /
                  (static_cast<float>(m_environmentMap->getWidth()) *
                   static_cast<float>(m_environmentMap->getHeight()))
                : 0.0f;
            m_optixEngine->setEnvSelectionWeight(w);
            std::cout << "[App] Environment map loaded (selection weight " << w << ")\n";
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
            case SceneNodeType::AreaLight: {
                // Mesh lights: the panel's position sliders must move the
                // OWNING INSTANCE — NEE samples the light's world-space
                // triangles, and the GpuAreaLight.position field is only the
                // (derived) centroid, which mesh-light NEE never reads.
                // Writing it did nothing, which is why dragging a mesh
                // light's position looked dead. Translate the instance
                // through the canonical transform path instead (which
                // re-bakes the triangles and the centroid).
                {
                    const GpuAreaLight* ml = m_lightManager->getAreaLight(index);
                    if (ml && ml->triCount > 0 && ml->instanceId != UINT32_MAX) {
                        float3 delta = make_float3(info.position.x - ml->position.x,
                                                   info.position.y - ml->position.y,
                                                   info.position.z - ml->position.z);
                        if (std::fabs(delta.x) + std::fabs(delta.y) + std::fabs(delta.z) > 1e-6f) {
                            const auto& instances = m_sceneManager->getInstances();
                            if (ml->instanceId < instances.size()) {
                                float t[12];
                                std::memcpy(t, instances[ml->instanceId].transform, sizeof(t));
                                t[3] += delta.x; t[7] += delta.y; t[11] += delta.z;
                                applyInstanceTransform(ml->instanceId, t);
                            }
                        }
                    }
                }
                m_lightManager->updateAreaLight(index, info);
                // Mesh lights: mirror the new emission into the owning
                // instance's MATERIAL so path-hit emission agrees with NEE
                // (the two MIS estimators must see the same emitter).
                const GpuAreaLight* l = m_lightManager->getAreaLight(index);
                if (l && l->instanceId != UINT32_MAX) {
                    MaterialHandle mh = m_sceneManager->getMaterialHandle(l->instanceId);
                    const GpuMaterial* mat = (mh != INVALID_MATERIAL_HANDLE)
                        ? m_materialManager->get(mh) : nullptr;
                    if (mat) {
                        GpuMaterial updated = *mat;
                        updated.emissive = l->emission;
                        m_materialManager->updateMaterial(mh, updated);
                        if (!m_sceneManager->updateMaterialRecords(mh, m_cudaInterop->getStream())) {
                            m_cudaInterop->synchronize();
                            m_sceneManager->updateSBT();
                        }
                    }
                }
                break;
            }
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

        // Fast path: patch the affected SBT records in place (stream-ordered,
        // no pipeline stall — slider drags no longer hitch). Falls back to a
        // full SBT rebuild when hit-group selection changes (alphaMode).
        if (!m_sceneManager->updateMaterialRecords(matHandle, m_cudaInterop->getStream())) {
            // Synchronize render stream before SBT rebuild — updateSBT frees
            // and reallocates GPU memory, unsafe while optixLaunch is in flight.
            m_cudaInterop->synchronize();
            m_sceneManager->updateSBT();
        }

        // Mirror the emissive into the instance's mesh light: the SBT
        // material drives what paths see when they HIT the lamp, but the
        // GpuAreaLight drives NEE — every directly-lit surface. Without the
        // mirror, editing a lamp's material brightened the lamp but not the
        // scene (and left the two MIS estimators disagreeing on emission).
        uint32_t lightIdx = m_lightManager->findAreaLightByInstance(instanceId);
        if (lightIdx != UINT32_MAX) {
            m_lightManager->setAreaLightEmission(lightIdx, material.emissive);
            m_lightManager->syncToGpu(m_optixEngine.get(), m_cudaInterop->getStream());
        }

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
                info.transmission = mat->transmission;
                info.ior = mat->ior;

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

        // Decompose the instance's current transform for the UI sliders
        const auto& instances = m_sceneManager->getInstances();
        if (instanceId < instances.size()) {
            decomposeTransform(instances[instanceId].transform,
                               info.scale, info.translation, info.rotation);
        }

        return info;
    });

    // Transform edit: recompose matrix, apply through the canonical path
    // (IAS rebuild + mesh-light re-bake + launch-param refresh + resets).
    // No updateSBT — a transform edit changes no material/geometry SBT
    // record, and the full SBT rebuild was a large part of the per-drag-event
    // cost. buildIAS itself reuses persistent buffers in the steady state.
    m_uiManager->setOnTransformEdit([this](uint32_t instanceId, float3 scale, float3 translation, float3 rotation) {
        float transform[12];
        composeTransform(scale, translation, rotation, transform);
        applyInstanceTransform(instanceId, transform);
    });
}

void Application::setupCallbacks() {
    // The Application is the SINGLE owner of the GLFW user pointer and all
    // GLFW callbacks. (Previously GLContext and InputHandler also registered
    // callbacks; GLFW keeps only the last registration and one user pointer,
    // so GLContext's framebuffer callback ended up casting the Application
    // pointer to GLContext* — the window-resize crash.)
    glfwSetWindowUserPointer(m_glContext->getWindow(), this);
    glfwSetKeyCallback(m_glContext->getWindow(), keyCallback);
    glfwSetCursorPosCallback(m_glContext->getWindow(), cursorPosCallback);
    glfwSetMouseButtonCallback(m_glContext->getWindow(), mouseButtonCallback);
    glfwSetScrollCallback(m_glContext->getWindow(), scrollCallback);
    glfwSetFramebufferSizeCallback(m_glContext->getWindow(), framebufferSizeCallback);

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

        if (m_aovAlbedoBuffer) cudaFree(m_aovAlbedoBuffer);
        if (m_aovNormalBuffer) cudaFree(m_aovNormalBuffer);
        cudaMalloc(reinterpret_cast<void**>(&m_aovAlbedoBuffer), bufferSize);
        cudaMalloc(reinterpret_cast<void**>(&m_aovNormalBuffer), bufferSize);
        cudaMemset(m_aovAlbedoBuffer, 0, bufferSize);
        cudaMemset(m_aovNormalBuffer, 0, bufferSize);
        m_optixEngine->setAOVBuffers(m_aovAlbedoBuffer, m_aovNormalBuffer);
        if (m_denoiser) m_denoiser->resize(width, height);

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
            // Note: the path-guide training counter deliberately keeps running
            // across camera moves. The grid is world-space, and resetting the
            // counter pinned the refit gate (0 % N == 0) true every frame,
            // EMA-decaying the whole grid's training away during navigation.
        }
        m_optixEngine->setCamera(currentParams);

        // Self-test hook (--test-move-light): translate the first mesh-light
        // instance at the configured frame, through the exact UI edit path.
        if (m_config.testMoveFrame > 0 &&
            m_optixEngine->getFrameIndex() >= static_cast<uint32_t>(m_config.testMoveFrame) &&
            !m_meshLightSources.empty()) {
            uint32_t instanceId = m_meshLightSources.begin()->first;
            const auto& instances = m_sceneManager->getInstances();
            if (instanceId < instances.size()) {
                float t[12];
                std::memcpy(t, instances[instanceId].transform, sizeof(t));
                t[3] += m_config.testMoveDelta[0];
                t[7] += m_config.testMoveDelta[1];
                t[11] += m_config.testMoveDelta[2];
                std::cout << "[App] TEST: moving mesh-light instance " << instanceId
                          << " by (" << m_config.testMoveDelta[0] << ", "
                          << m_config.testMoveDelta[1] << ", "
                          << m_config.testMoveDelta[2] << ")\n";
                applyInstanceTransform(instanceId, t);
            }
            m_config.testMoveFrame = 0;   // fire once
        }

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

void Application::applyInstanceTransform(uint32_t instanceId, const float transform12[12]) {
    m_sceneManager->updateInstanceTransform(instanceId, transform12);

    // Render stream must be idle before the IAS rebuild touches device
    // buffers optixLaunch may still be reading.
    m_cudaInterop->synchronize();
    m_sceneManager->buildIAS();

    // Mesh lights bake world-space NEE geometry — re-extract it for the
    // moved instance (no-op for instances without a light).
    refreshMeshLightGeometry(instanceId);

    m_optixEngine->setSceneHandle(m_sceneManager->getSceneHandle());
    m_optixEngine->setGeometryBuffers(m_sceneManager->getVertexBuffers(),
                                       m_sceneManager->getIndexBuffers());
    m_optixEngine->setInstanceData(
        m_sceneManager->getInstanceTransforms(),
        m_sceneManager->getInstanceNormalTransforms(),
        m_sceneManager->getInstanceMaterialIndices());

    m_optixEngine->resetAccumulation();
    resetPathGuideTraining();
}

void Application::refreshMeshLightGeometry(uint32_t instanceId) {
    auto it = m_meshLightSources.find(instanceId);
    if (it == m_meshLightSources.end() || !m_areaLightTris) return;
    const MeshLightSource& src = it->second;
    const auto& instances = m_sceneManager->getInstances();
    if (instanceId >= instances.size()) return;
    const float* t = instances[instanceId].transform;

    // Re-bake world-space triangles + per-light area CDF from the saved
    // object-space source, mirroring the extraction in loadScene.
    const size_t triCount = src.objVerts.size() / 3;
    std::vector<float4> tris;
    tris.reserve(triCount * 3);
    glm::vec3 centroid(0.0f);
    glm::vec3 weightedNormal(0.0f);
    float totalArea = 0.0f;
    for (size_t ti = 0; ti < triCount; ti++) {
        glm::vec3 lp[3];
        for (int vi = 0; vi < 3; vi++) {
            const float3& p = src.objVerts[ti * 3 + vi];
            lp[vi] = glm::vec3(
                t[0]*p.x + t[1]*p.y + t[2]*p.z  + t[3],
                t[4]*p.x + t[5]*p.y + t[6]*p.z  + t[7],
                t[8]*p.x + t[9]*p.y + t[10]*p.z + t[11]);
        }
        glm::vec3 cr = glm::cross(lp[1] - lp[0], lp[2] - lp[0]);
        float triArea = glm::length(cr) * 0.5f;
        totalArea += triArea;
        weightedNormal += cr;
        centroid += (lp[0] + lp[1] + lp[2]) * triArea;
        tris.push_back(make_float4(lp[0].x, lp[0].y, lp[0].z, 0.0f));
        tris.push_back(make_float4(lp[1].x, lp[1].y, lp[1].z, triArea));
        tris.push_back(make_float4(lp[2].x, lp[2].y, lp[2].z, 0.0f));
    }
    if (totalArea <= 1e-12f || tris.empty()) return;  // degenerate transform

    float cumulative = 0.0f;
    for (size_t ti = 0; ti < triCount; ti++) {
        cumulative += tris[ti * 3 + 1].w / totalArea;
        tris[ti * 3].w = cumulative;
    }
    tris[(triCount - 1) * 3].w = 1.0f;

    // In-place update of this light's slice of the device triangle buffer.
    // The transform-edit path synchronized the render stream before the IAS
    // rebuild, so no launch is reading the buffer here.
    cudaMemcpy(m_areaLightTris + (size_t)src.triOffset * 3, tris.data(),
               tris.size() * sizeof(float4), cudaMemcpyHostToDevice);

    centroid /= (totalArea * 3.0f);
    glm::vec3 avgNormal = glm::normalize(weightedNormal);
    float sideLen = std::sqrt(totalArea);
    m_lightManager->setAreaLightGeometry(src.lightIdx,
        make_float3(centroid.x, centroid.y, centroid.z),
        make_float3(avgNormal.x, avgNormal.y, avgNormal.z),
        totalArea, make_float2(sideLen, sideLen));
    // Area feeds the selection weights + MIS pdfs — resync (also rebuilds
    // the light alias table).
    m_lightManager->syncToGpu(m_optixEngine.get(), m_cudaInterop->getStream());
    std::cout << "[App] Mesh light " << src.lightIdx << " re-baked: centroid ("
              << centroid.x << ", " << centroid.y << ", " << centroid.z
              << "), area " << totalArea << "\n";
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

    // ═══ PHASE 1: PATH GUIDE UI/DEBUG UPKEEP ═══
    // The cell table is device-resident (shaders allocate cells on first
    // touch), so there is no build pipeline to drive — only periodic UI
    // bookkeeping: a non-blocking 4-byte cell-count poll, the debug stat
    // reset, and (debug-viz only) a synchronous host-mirror refresh for the
    // wireframe overlay.
    if (m_pathGuideMode != PathGuideMode::Disabled && m_optixEngine) {
        m_pathGuideStatsFrame++;
        if (m_pathGuideStatsFrame >= 60) {
            m_pathGuideStatsFrame = 0;
            cudaStream_t stream = m_cudaInterop ? m_cudaInterop->getStream() : nullptr;
            m_optixEngine->resetPathGuideStats(stream);

            if (m_pathGuideGrid && m_pathGuideGrid->isInitialized()) {
                m_pathGuideGrid->requestCellCountAsync(stream);

                if (m_debugGridVisualize && m_wireframeRenderer &&
                    m_wireframeRenderer->isInitialized() &&
                    m_pathGuideGrid->refreshHostMirror()) {
                    auto vertices = m_pathGuideGrid->generateEdgeVerticesAllLevels();
                    m_wireframeRenderer->updateVertices(vertices);
                }

                if (m_uiManager) {
                    m_uiManager->updatePathGuideGridStats(
                        m_pathGuideGrid->getNumLevels(),
                        m_pathGuideGrid->getTotalCells(),
                        m_pathGuideGrid->getEntryStride());
                }
            }

            // Update automation status in UI
            if (m_uiManager) {
                const char* modeStr = "Disabled";
                switch (m_pathGuideMode) {
                    case PathGuideMode::Running:  modeStr = "Running"; break;
                    case PathGuideMode::Paused:   modeStr = "Paused"; break;
                    case PathGuideMode::StepOnce: modeStr = "StepOnce"; break;
                    default: break;
                }
                m_uiManager->updatePathGuideAutomationStatus(modeStr, m_pathGuideTrainingFrameCount, m_pathGuideSubdivPasses);
            }
        }
    }

    // [DIAG] buildChk
    { auto tNow = diagT(); diagAccum[0] += diagMs(tPrev, tNow); tPrev = tNow; }

    // ═══ PHASE 2: DISPLAY PREVIOUS FRAME ═══
    // Wait only for the buffer being DISPLAYED (event recorded after its
    // unmap), not the whole render stream. The displayed buffer is 2 frames
    // old, so this wait is usually zero and the triple buffer can actually
    // keep more than one frame in flight. (cudaEventSynchronize on a
    // never-recorded event returns immediately, so warmup frames are safe.)
    m_cudaInterop->waitForRender(m_displayIdx);

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
    if (m_pathGuideGrid && m_pathGuideGrid->isInitialized()) {
        bool training = (m_pathGuideMode == PathGuideMode::Running ||
                         m_pathGuideMode == PathGuideMode::StepOnce);
        cudaStream_t stream = m_cudaInterop ? m_cudaInterop->getStream() : nullptr;
        uint32_t currentFrame = m_optixEngine->getFrameIndex();
        bool stepOnce = (m_pathGuideMode == PathGuideMode::StepOnce);

        // Device-side lobe refit: fold interval sums into cumulative sums and
        // refit vMF lobes in place — a tiny kernel on the render stream.
        // Lobes only change between launches, so sampling PDFs stay
        // consistent within each frame.
        if (training &&
            ((m_pathGuideTrainingFrameCount % m_pathGuideRefitInterval) == 0 || stepOnce)) {
            m_pathGuideGrid->refitLobes(currentFrame, stream);
        }

        // Device-side subdivision: insert children of cells whose EMA deposit
        // count crossed the threshold. Idempotent and cheap; the new cells
        // are usable as soon as the next refit gives them their own fit
        // (until then they sample with the parent's warm-started lobe).
        if (training && (stepOnce ||
            (m_pathGuideTrainingFrameCount > 0 &&
             (m_pathGuideTrainingFrameCount % m_pathGuideGrid->getConfig().refine_interval_frames) == 0))) {
            m_pathGuideSubdivPasses++;
            m_pathGuideGrid->runSubdivisionPass(currentFrame, m_pathGuideSubdivPasses, stream);
        }

        // Harvest completed subdivision-pass stats (async, at most one
        // readback in flight — a pass launched while one is pending runs
        // without stats and is not reported; with the 30-frame cadence that
        // requires a badly stalled stream). Every reported pass prints,
        // tagged with ITS OWN pass id — a silent console means "no pass
        // ran", never "pass ran and found nothing" (that ambiguity cost a
        // debugging round). The level histogram is the ground truth for
        // "where is the grid refining".
        {
            uint32_t s[PG_SUBDIV_STATS_SIZE];
            uint32_t passId = 0;
            if (m_pathGuideGrid->pollSubdivStats(s, &passId)) {
                std::cout << "[PathGuide] subdiv pass " << passId
                          << ": split=" << s[PG_SUBDIV_STAT_SPLIT]
                          << " eligible=" << s[PG_SUBDIV_STAT_ELIGIBLE]
                          << " no-structure=" << s[PG_SUBDIV_STAT_NOSTRUCT]
                          << " children=" << s[PG_SUBDIV_STAT_CHILDREN]
                          << " | levels:";
                for (uint32_t l = 0; l < 16; l++) {
                    if (s[PG_SUBDIV_STAT_LEVEL0 + l])
                        std::cout << " L" << l << "=" << s[PG_SUBDIV_STAT_LEVEL0 + l];
                }
                std::cout << "\n";
            }
        }

        // StepOnce: one refit + subdivision step, then freeze
        if (stepOnce) {
            m_pathGuideMode = PathGuideMode::Paused;
        }

        SparsePathGuideDescriptor sparseDesc = m_pathGuideGrid->getDescriptor();
        m_optixEngine->setPathGuideGridDescriptor(&sparseDesc);
        const auto& config = m_pathGuideGrid->getConfig();
        m_optixEngine->setPathGuideLevelConfig(config.start_level, config.min_level, config.max_level);

        // Deposits only while refits are running (Paused keeps sampling the
        // guide but must not grow the un-folded interval sums)
        m_optixEngine->setPathGuideTraining(training);

        // Increment training frame counter when actively training
        if (training) {
            m_pathGuideTrainingFrameCount++;
        }
    } else {
        m_optixEngine->setPathGuideGridDescriptor(nullptr);
        m_optixEngine->setPathGuideTraining(false);
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

    // Denoise the display copy (PBO); accumulation buffer stays untouched
    if (m_denoiserEnabled && m_denoiser) {
        m_denoiser->denoise(
            devicePtr, devicePtr,
            m_aovAlbedoBuffer, m_aovNormalBuffer,
            m_glContext->getWidth(), m_glContext->getHeight(),
            m_denoiserBlend,
            m_cudaInterop->getStream());
    }

    // [DIAG] render
    { auto tNow = diagT(); diagAccum[7] += diagMs(tPrev, tNow); tPrev = tNow; }

    // Unmap, THEN record the event: the display path waits on this event
    // before glTexSubImage2D, so it must cover the unmap (GL may not touch a
    // PBO that is still CUDA-mapped). Both calls are stream-ordered.
    m_cudaInterop->unmapBuffer(m_writeIdx);
    m_cudaInterop->recordRenderComplete(m_writeIdx);

    // [DIAG] unmapEvt
    { auto tNow = diagT(); diagAccum[8] += diagMs(tPrev, tNow); tPrev = tNow; }

    // ═══ PHASE 5: ADVANCE ═══
    m_writeIdx = (m_writeIdx + 1) % 3;
    m_framesPipelined++;

    // [DIAG] asyncIO
    { auto tNow = diagT(); diagAccum[9] += diagMs(tPrev, tNow); tPrev = tNow; }

    // Phase timing report every 120 frames — only when verbose logging is on
    // (F6). The accumulators reset either way so the window stays aligned.
    diagFrameCount++;
    if (diagFrameCount % 120 == 0) {
        if (verboseLogging()) {
            double total = 0;
            for (int i = 0; i < 10; i++) total += diagAccum[i];
            std::cout << "[DIAG] avg ms/frame over 120 frames (total="
                      << (total / 120.0) << "ms):\n";
            for (int i = 0; i < 10; i++) {
                std::cout << "  " << diagNames[i] << ": "
                          << (diagAccum[i] / 120.0) << " ms ("
                          << (100.0 * diagAccum[i] / total) << "%)\n";
            }
        }
        for (int i = 0; i < 10; i++) diagAccum[i] = 0;
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

    if (m_hemisphereVis) m_hemisphereVis->shutdown();
    if (m_inputHandler) m_inputHandler->shutdown();
    if (m_uiRenderer) m_uiRenderer->shutdown();
    if (m_uiManager) m_uiManager->shutdown();
    if (m_fontAtlas) m_fontAtlas->release();
    if (m_sceneManager) m_sceneManager->clear();

    if (m_denoiser) { m_denoiser->shutdown(); m_denoiser.reset(); }
    if (m_accumulationBuffer) {
        cudaFree(m_accumulationBuffer);
        m_accumulationBuffer = nullptr;
    }
    if (m_aovAlbedoBuffer) { cudaFree(m_aovAlbedoBuffer); m_aovAlbedoBuffer = nullptr; }
    if (m_aovNormalBuffer) { cudaFree(m_aovNormalBuffer); m_aovNormalBuffer = nullptr; }
    if (m_areaLightTris) {
        cudaFree(m_areaLightTris);
        m_areaLightTris = nullptr;
    }
    if (m_instanceLightIndices) {
        cudaFree(m_instanceLightIndices);
        m_instanceLightIndices = nullptr;
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
    std::cout << "  F6       - Toggle verbose logging (default off)\n";
    std::cout << "  1-4      - Resolution presets\n";
    std::cout << "  F1-F4    - Quality modes\n";
    std::cout << "  [ ]      - Decrease/Increase SPP\n";
    std::cout << "  H        - Toggle hierarchy panel\n";
    std::cout << "  P        - Toggle property panel\n";
    std::cout << "  G        - Toggle grid debug panel\n";
    std::cout << "  DblClick - Open material panel for clicked surface\n";
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

        case GLFW_KEY_F6: {
            bool verbose = !verboseLogging();
            g_verboseLogging.store(verbose, std::memory_order_relaxed);
            std::cout << "[Log] Verbose logging: " << (verbose ? "ON" : "OFF") << "\n";
            break;
        }

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

        case GLFW_KEY_N:
            if (mods & GLFW_MOD_CONTROL) {
                app->m_denoiserBlend += 0.25f;
                if (app->m_denoiserBlend > 0.99f) app->m_denoiserBlend = 0.0f;
                std::cout << "[Denoise] Blend: " << app->m_denoiserBlend << "\n";
            } else {
                app->m_denoiserEnabled = !app->m_denoiserEnabled;
                std::cout << "[Denoise] " << (app->m_denoiserEnabled ? "ON" : "OFF") << "\n";
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
        bool consumed = app->m_uiManager->handleMouseMove(pos);
        // Feed the UI-consumption state to the input handler so run() can
        // gate keyboard camera movement (the handler no longer has its own
        // GLFW callbacks — they were being silently overridden anyway).
        if (app->m_inputHandler) {
            app->m_inputHandler->setMouseConsumed(consumed);
        }
    } else if (app->m_inputHandler) {
        app->m_inputHandler->setMouseConsumed(false);
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
        // Pick on the render stream: the legacy default stream would act as a
        // barrier against ALL streams and its final sync would drain the whole
        // multi-frame pipeline — a visible hitch on every viewport click.
        PickResultBuffer pickResult = app->m_optixEngine->pickInstanceAndPosition(
            static_cast<uint32_t>(xpos), static_cast<uint32_t>(ypos),
            app->m_cudaInterop ? app->m_cudaInterop->getStream() : 0);

        // Viewport double-click: open the picked surface's material panel.
        // Works in EVERY mode — during path guiding this is the only way to
        // reach a material without disturbing the render (single clicks
        // deliberately leave selection and accumulation untouched while
        // guiding, and opening the panel disturbs neither).
        double now = glfwGetTime();
        bool isDoubleClick =
            pickResult.instanceId != UINT32_MAX &&
            pickResult.instanceId == app->m_lastViewportClickInstance &&
            (now - app->m_lastViewportClickTime) < 0.35 &&
            std::abs(xpos - app->m_lastViewportClickX) < 8.0 &&
            std::abs(ypos - app->m_lastViewportClickY) < 8.0;
        if (isDoubleClick) {
            app->m_uiManager->showInstancePropertiesFor(pickResult.instanceId);
            // Require a fresh pair of clicks for the next double-click
            app->m_lastViewportClickTime = -1.0;
            app->m_lastViewportClickInstance = UINT32_MAX;
        } else {
            app->m_lastViewportClickTime = now;
            app->m_lastViewportClickX = xpos;
            app->m_lastViewportClickY = ypos;
            app->m_lastViewportClickInstance = pickResult.instanceId;
        }

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
        // (inspectCellAtPosition refreshes the host mirror itself, so gate on
        // initialization only — the polled cell count can lag by ~60 frames)
        if (pickResult.instanceId != UINT32_MAX && app->m_pathGuideGrid &&
            app->m_pathGuideGrid->isInitialized()) {
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
                // vMF mixture: show the top-2 lobes by mixture weight
                // (layout: see path_guide_cell_layout.h)
                int top0 = 0, top1 = -1;
                for (int k = 1; k < PG_NUM_LOBES; k++) {
                    float w = cellResult.data[k * PG_LOBE_STRIDE + PG_L_WEIGHT];
                    if (w > cellResult.data[top0 * PG_LOBE_STRIDE + PG_L_WEIGHT]) {
                        top1 = top0;
                        top0 = k;
                    } else if (top1 < 0 ||
                               w > cellResult.data[top1 * PG_LOBE_STRIDE + PG_L_WEIGHT]) {
                        top1 = k;
                    }
                }
                auto lobeAngles = [&cellResult](int k, float& theta, float& phi) {
                    const float* l = cellResult.data + k * PG_LOBE_STRIDE;
                    float mx = l[PG_L_MU_X], my = l[PG_L_MU_Y], mz = l[PG_L_MU_Z];
                    float mlen = std::sqrt(mx*mx + my*my + mz*mz);
                    if (mlen > 1e-8f) {
                        theta = std::acos(std::max(-1.0f, std::min(1.0f, my / mlen)));
                        phi = std::atan2(mz, mx);
                        if (phi < 0.0f) phi += 6.28318530718f;
                    } else {
                        theta = 0.0f;
                        phi = 0.0f;
                    }
                };
                lobeAngles(top0, info.theta0, info.phi0);
                info.kappa0 = cellResult.data[top0 * PG_LOBE_STRIDE + PG_L_KAPPA];
                lobeAngles(top1, info.theta1, info.phi1);
                info.kappa1 = cellResult.data[top1 * PG_LOBE_STRIDE + PG_L_KAPPA];
                float w0 = cellResult.data[top0 * PG_LOBE_STRIDE + PG_L_WEIGHT];
                float w1 = cellResult.data[top1 * PG_LOBE_STRIDE + PG_L_WEIGHT];
                info.pi0 = (w0 + w1 > 1e-9f) ? w0 / (w0 + w1) : 1.0f;
                // Cumulative (EMA lifetime) stats, summed over all lobes
                float sumX = 0.0f, sumY = 0.0f, sumZ = 0.0f, sumW = 0.0f;
                for (int k = 0; k < PG_NUM_LOBES; k++) {
                    const float* cs = cellResult.data + PG_CUMS_BASE + k * PG_SUM_STRIDE;
                    sumX += cs[0]; sumY += cs[1]; sumZ += cs[2]; sumW += cs[3];
                }
                info.sumW = sumW;
                float meanLen = (sumW > 1e-9f)
                    ? std::sqrt(sumX*sumX + sumY*sumY + sumZ*sumZ) / sumW
                    : 0.0f;
                info.variance = 1.0f - meanLen;
                info.lastFrame = cellResult.data[PG_LAST_HIT_FRAME];

                // Determine subdivision/coarsening status (display heuristic).
                // Mirrors subdivideCellsKernel: (1) level-normalized VISIT
                // gate (traffic, no radiance gate), (2) per-axis half-cell
                // conditional mean log1p(radiance) ratio above the threshold
                // with a per-half visit floor. Coarsening is retired (cell
                // indices are stable).
                const auto& config = app->m_pathGuideGrid->getConfig();
                float gateVisits = config.subdivide_count_threshold;
                if (cellResult.level > config.start_level) {
                    gateVisits = std::max(
                        gateVisits * std::exp2(-2.0f * static_cast<float>(
                            cellResult.level - config.start_level)),
                        std::min(256.0f, config.subdivide_count_threshold));
                }
                float visits = cellResult.data[PG_CUM_VISITS];
                float sl = cellResult.data[PG_CUM_SL];
                bool structure = false;
                for (int a = 0; a < 3; a++) {
                    float posC = cellResult.data[PG_CUM_HC_X + a];
                    float negC = visits - posC;
                    if (posC < 32.0f || negC < 32.0f) continue;
                    float posMean = cellResult.data[PG_CUM_HL_X + a] / posC;
                    float negMean = std::max(sl - cellResult.data[PG_CUM_HL_X + a], 0.0f) / negC;
                    float ratio = std::fabs(std::log((posMean + 0.02f) / (negMean + 0.02f)));
                    float diff = std::fabs(posMean - negMean);
                    if (ratio > config.subdivide_contrast_threshold ||
                        diff > 1.7f * config.subdivide_contrast_threshold) {
                        structure = true;
                        break;
                    }
                }
                // Resolution-limited test (mirrors subdivideCellsKernel
                // gate 2b): a well-evidenced lobe whose fitted kappa demand
                // exceeds the spread-based cap severalfold splits even
                // without a radiance edge.
                if (!structure) {
                    float smw = cellResult.data[PG_CUM_SMW];
                    if (smw > 1e-6f) {
                        float invSmw = 1.0f / smw;
                        float cwx = cellResult.data[PG_CUM_SR_X] * invSmw;
                        float cwy = cellResult.data[PG_CUM_SR_Y] * invSmw;
                        float cwz = cellResult.data[PG_CUM_SR_Z] * invSmw;
                        float spread2 = cellResult.data[PG_CUM_SRR] * invSmw -
                                        (cwx * cwx + cwy * cwy + cwz * cwz);
                        float spreadRel = std::max(std::sqrt(std::max(spread2, 0.0f)), 0.25f);
                        float baseCellSize =
                            ((config.bounds_max[0] - config.bounds_min[0]) +
                             (config.bounds_max[1] - config.bounds_min[1]) +
                             (config.bounds_max[2] - config.bounds_min[2])) /
                            (3.0f * static_cast<float>(config.base_resolution));
                        float cellSize = baseCellSize * std::exp2(-static_cast<float>(cellResult.level));
                        float sigmaPos = std::max(spreadRel * 0.5f * cellSize, 1e-6f);
                        for (int k = 0; k < PG_NUM_LOBES; k++) {
                            const float* cs = cellResult.data + PG_CUMS_BASE + k * PG_SUM_STRIDE;
                            float w = cs[3];
                            if (w < 32.0f) continue;
                            float len = std::sqrt(cs[0]*cs[0] + cs[1]*cs[1] + cs[2]*cs[2]);
                            float rbar = std::min(len / w, 0.99999f);
                            float implied = rbar * (3.0f - rbar * rbar) /
                                            std::max(1.0f - rbar * rbar, 1e-4f);
                            float meanDist = cs[PG_S_DIST] / w;
                            if (meanDist < 1e-4f) continue;
                            float cap = (meanDist / sigmaPos) * (meanDist / sigmaPos);
                            if (cap < 500.0f && implied > 4.0f * cap) {
                                structure = true;
                                break;
                            }
                        }
                    }
                }
                info.wouldSubdivide = (cellResult.level < config.max_level &&
                    config.subdivide_count_threshold > 0.0f &&
                    visits >= gateVisits && structure);
                info.wouldCoarsen = false;
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

void Application::framebufferSizeCallback(GLFWwindow* window, int width, int height) {
    auto* app = static_cast<Application*>(glfwGetWindowUserPointer(window));
    if (app && app->m_glContext) {
        app->m_glContext->onFramebufferResized(width, height);
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
