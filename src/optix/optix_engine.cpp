#include "optix_engine.h"
#include <optix_stubs.h>
#include <optix_function_table_definition.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cstring>

namespace spectra {

// OptiX logging callback
static void optixLogCallback(unsigned int level, const char* tag, const char* message, void* /*cbdata*/) {
    std::cerr << "[OptiX][" << level << "][" << tag << "]: " << message << "\n";
}

OptixEngine::~OptixEngine() {
    shutdown();
}

bool OptixEngine::init(CUcontext cudaContext) {
    // Initialize OptiX
    OPTIX_CHECK(optixInit());

    // Create OptiX device context
    OptixDeviceContextOptions options = {};
    options.logCallbackFunction = &optixLogCallback;
    options.logCallbackData = nullptr;
#ifdef _DEBUG
    options.logCallbackLevel = 4;  // All messages in debug
    options.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;
#else
    options.logCallbackLevel = 2;  // Errors and warnings only
    options.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_OFF;
#endif

    // Use CUcontext = 0 to use current CUDA context, or pass explicit context
    CUcontext ctx = cudaContext ? cudaContext : 0;
    OPTIX_CHECK(optixDeviceContextCreate(ctx, &options, &m_context));

    std::cout << "[OptiX] Context created successfully\n";

    // Allocate launch params buffer
    cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&m_launchParamsBuffer), sizeof(LaunchParams));
    if (err != cudaSuccess) {
        std::cerr << "[OptiX] Failed to allocate launch params buffer: " << cudaGetErrorString(err) << "\n";
        return false;
    }

    // Initialize launch params with defaults
    m_launchParams = {};
    m_launchParams.output_buffer = nullptr;
    m_launchParams.accumulation_buffer = nullptr;
    m_launchParams.accumulated_frames = 0;
    m_launchParams.scene_handle = 0;
    m_launchParams.vertex_buffers = nullptr;
    m_launchParams.index_buffers = nullptr;
    m_launchParams.instance_material_indices = nullptr;

    // Set default camera
    m_launchParams.camera.position = make_float3(0.0f, 0.0f, 5.0f);
    m_launchParams.camera.forward = make_float3(0.0f, 0.0f, -1.0f);
    m_launchParams.camera.right = make_float3(1.0f, 0.0f, 0.0f);
    m_launchParams.camera.up = make_float3(0.0f, 1.0f, 0.0f);
    m_launchParams.camera.fovY = 1.0472f;  // 60 degrees in radians
    m_launchParams.camera.aspectRatio = 16.0f / 9.0f;
    m_launchParams.camera.nearPlane = 0.01f;
    m_launchParams.camera.farPlane = 1000.0f;

    // Initialize lighting to empty
    m_launchParams.point_lights = nullptr;
    m_launchParams.point_light_count = 0;
    m_launchParams.directional_lights = nullptr;
    m_launchParams.directional_light_count = 0;
    m_launchParams.area_lights = nullptr;
    m_launchParams.area_light_count = 0;

    // Initialize environment map
    m_launchParams.environment_map = 0;
    m_launchParams.environment_intensity = 1.0f;

    // Initialize environment CDF (for importance sampling)
    m_launchParams.env_conditional_cdf = 0;
    m_launchParams.env_marginal_cdf = 0;
    m_launchParams.env_width = 0;
    m_launchParams.env_height = 0;
    m_launchParams.env_total_luminance = 0.0f;

    // Initialize quality settings
    m_launchParams.quality_mode = QUALITY_BALANCED;
    m_launchParams.samples_per_pixel = 4;  // Default 4 SPP for reasonable first-frame quality
    m_launchParams.random_seed = 0;

    // Initialize selection (UINT32_MAX = no selection)
    m_launchParams.selected_instance_id = UINT32_MAX;

    // Initialize pick mode
    m_launchParams.pick_mode = 0;
    m_launchParams.pick_x = 0;
    m_launchParams.pick_y = 0;

    // Allocate pick result buffer (single uint32_t)
    err = cudaMalloc(reinterpret_cast<void**>(&m_pickBuffer), sizeof(uint32_t));
    if (err != cudaSuccess) {
        std::cerr << "[OptiX] Failed to allocate pick buffer: " << cudaGetErrorString(err) << "\n";
        return false;
    }
    m_launchParams.pick_result = reinterpret_cast<uint32_t*>(m_pickBuffer);

    return true;
}

bool OptixEngine::createPipeline(const std::filesystem::path& ptxDir) {
    // Set pipeline compile options for Phase 2
    m_pipelineCompileOptions = {};
    m_pipelineCompileOptions.usesMotionBlur = false;
    m_pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
    m_pipelineCompileOptions.numPayloadValues = 5;      // color (3) + hitDistance (1) + instanceId (1)
    m_pipelineCompileOptions.numAttributeValues = 2;    // barycentrics (u, v)
    m_pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    m_pipelineCompileOptions.pipelineLaunchParamsVariableName = "params";

    // Load raygen module
    auto raygenPtx = ptxDir / "raygen.ptx";
    if (!createModule(raygenPtx, &m_raygenModule)) {
        return false;
    }

    // Load miss module
    auto missPtx = ptxDir / "miss.ptx";
    if (!createModule(missPtx, &m_missModule)) {
        return false;
    }

    // Load closesthit module
    auto closesthitPtx = ptxDir / "closesthit.ptx";
    if (!createModule(closesthitPtx, &m_closesthitModule)) {
        return false;
    }

    // Load anyhit module (for alpha testing)
    auto anyhitPtx = ptxDir / "anyhit.ptx";
    if (!createModule(anyhitPtx, &m_anyhitModule)) {
        return false;
    }

    // Create program groups
    if (!createProgramGroups()) {
        return false;
    }

    // Link pipeline
    OptixPipelineLinkOptions linkOptions = {};
    linkOptions.maxTraceDepth = 2;  // Primary ray + shadow ray

    OptixProgramGroup programGroups[] = { 
        m_raygenPG, m_missPG, m_missShadowPG, 
        m_hitgroupPG, m_hitgroupShadowPG,
        m_hitgroupAlphaPG, m_hitgroupShadowAlphaPG
    };

    char log[2048];
    size_t logSize = sizeof(log);

    OPTIX_CHECK_LOG(optixPipelineCreate(
        m_context,
        &m_pipelineCompileOptions,
        &linkOptions,
        programGroups,
        sizeof(programGroups) / sizeof(programGroups[0]),
        log,
        &logSize,
        &m_pipeline
    ));

    std::cout << "[OptiX] Pipeline created\n";

    // Set stack sizes
    OPTIX_CHECK(optixPipelineSetStackSize(
        m_pipeline,
        2 * 1024,   // Direct callable stack size
        2 * 1024,   // Continuation stack size
        2 * 1024,   // Continuation stack size from traversal
        2           // Max traversable depth
    ));

    // Create default SBT (will be updated with materials later)
    if (!createDefaultSBT()) {
        return false;
    }

    return true;
}

bool OptixEngine::createModule(const std::filesystem::path& ptxPath, OptixModule* module) {
    // Read PTX file
    std::ifstream file(ptxPath, std::ios::binary);
    if (!file) {
        std::cerr << "[OptiX] Failed to open PTX file: " << ptxPath << "\n";
        return false;
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string ptxSource = buffer.str();

    if (ptxSource.empty()) {
        std::cerr << "[OptiX] PTX file is empty: " << ptxPath << "\n";
        return false;
    }

    // Module compile options
    OptixModuleCompileOptions moduleOptions = {};
#ifdef _DEBUG
    moduleOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
    moduleOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
#else
    moduleOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    moduleOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;
#endif

    char log[2048];
    size_t logSize = sizeof(log);

    OPTIX_CHECK_LOG(optixModuleCreate(
        m_context,
        &moduleOptions,
        &m_pipelineCompileOptions,
        ptxSource.c_str(),
        ptxSource.size(),
        log,
        &logSize,
        module
    ));

    std::cout << "[OptiX] Module loaded: " << ptxPath.filename() << "\n";

    return true;
}

bool OptixEngine::createProgramGroups() {
    char log[2048];
    size_t logSize;

    // Raygen program group
    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroupDesc raygenDesc = {};
    raygenDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    raygenDesc.raygen.module = m_raygenModule;
    raygenDesc.raygen.entryFunctionName = "__raygen__simple";

    logSize = sizeof(log);
    OPTIX_CHECK_LOG(optixProgramGroupCreate(
        m_context,
        &raygenDesc,
        1,
        &pgOptions,
        log,
        &logSize,
        &m_raygenPG
    ));

    // Miss program group (radiance - background)
    OptixProgramGroupDesc missDesc = {};
    missDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    missDesc.miss.module = m_missModule;
    missDesc.miss.entryFunctionName = "__miss__background";

    logSize = sizeof(log);
    OPTIX_CHECK_LOG(optixProgramGroupCreate(
        m_context,
        &missDesc,
        1,
        &pgOptions,
        log,
        &logSize,
        &m_missPG
    ));

    // Miss program group (shadow - visibility)
    OptixProgramGroupDesc missShadowDesc = {};
    missShadowDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    missShadowDesc.miss.module = m_missModule;
    missShadowDesc.miss.entryFunctionName = "__miss__shadow";

    logSize = sizeof(log);
    OPTIX_CHECK_LOG(optixProgramGroupCreate(
        m_context,
        &missShadowDesc,
        1,
        &pgOptions,
        log,
        &logSize,
        &m_missShadowPG
    ));

    // Hit group program group (radiance - closesthit)
    OptixProgramGroupDesc hitgroupDesc = {};
    hitgroupDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hitgroupDesc.hitgroup.moduleCH = m_closesthitModule;
    hitgroupDesc.hitgroup.entryFunctionNameCH = "__closesthit__radiance";
    hitgroupDesc.hitgroup.moduleAH = nullptr;  // No anyhit for opaque
    hitgroupDesc.hitgroup.entryFunctionNameAH = nullptr;
    hitgroupDesc.hitgroup.moduleIS = nullptr;  // Use built-in triangle intersection
    hitgroupDesc.hitgroup.entryFunctionNameIS = nullptr;

    logSize = sizeof(log);
    OPTIX_CHECK_LOG(optixProgramGroupCreate(
        m_context,
        &hitgroupDesc,
        1,
        &pgOptions,
        log,
        &logSize,
        &m_hitgroupPG
    ));

    // Hit group program group (shadow - closesthit)
    OptixProgramGroupDesc hitgroupShadowDesc = {};
    hitgroupShadowDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hitgroupShadowDesc.hitgroup.moduleCH = m_closesthitModule;
    hitgroupShadowDesc.hitgroup.entryFunctionNameCH = "__closesthit__shadow";
    hitgroupShadowDesc.hitgroup.moduleAH = nullptr;  // No anyhit for opaque shadows
    hitgroupShadowDesc.hitgroup.entryFunctionNameAH = nullptr;
    hitgroupShadowDesc.hitgroup.moduleIS = nullptr;  // Use built-in triangle intersection
    hitgroupShadowDesc.hitgroup.entryFunctionNameIS = nullptr;

    logSize = sizeof(log);
    OPTIX_CHECK_LOG(optixProgramGroupCreate(
        m_context,
        &hitgroupShadowDesc,
        1,
        &pgOptions,
        log,
        &logSize,
        &m_hitgroupShadowPG
    ));

    // Hit group program group (radiance - closesthit + anyhit for alpha testing)
    OptixProgramGroupDesc hitgroupAlphaDesc = {};
    hitgroupAlphaDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hitgroupAlphaDesc.hitgroup.moduleCH = m_closesthitModule;
    hitgroupAlphaDesc.hitgroup.entryFunctionNameCH = "__closesthit__radiance";
    hitgroupAlphaDesc.hitgroup.moduleAH = m_anyhitModule;
    hitgroupAlphaDesc.hitgroup.entryFunctionNameAH = "__anyhit__alpha";
    hitgroupAlphaDesc.hitgroup.moduleIS = nullptr;
    hitgroupAlphaDesc.hitgroup.entryFunctionNameIS = nullptr;

    logSize = sizeof(log);
    OPTIX_CHECK_LOG(optixProgramGroupCreate(
        m_context,
        &hitgroupAlphaDesc,
        1,
        &pgOptions,
        log,
        &logSize,
        &m_hitgroupAlphaPG
    ));

    // Hit group program group (shadow - closesthit + anyhit for alpha testing)
    OptixProgramGroupDesc hitgroupShadowAlphaDesc = {};
    hitgroupShadowAlphaDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hitgroupShadowAlphaDesc.hitgroup.moduleCH = m_closesthitModule;
    hitgroupShadowAlphaDesc.hitgroup.entryFunctionNameCH = "__closesthit__shadow";
    hitgroupShadowAlphaDesc.hitgroup.moduleAH = m_anyhitModule;
    hitgroupShadowAlphaDesc.hitgroup.entryFunctionNameAH = "__anyhit__shadow_alpha";
    hitgroupShadowAlphaDesc.hitgroup.moduleIS = nullptr;
    hitgroupShadowAlphaDesc.hitgroup.entryFunctionNameIS = nullptr;

    logSize = sizeof(log);
    OPTIX_CHECK_LOG(optixProgramGroupCreate(
        m_context,
        &hitgroupShadowAlphaDesc,
        1,
        &pgOptions,
        log,
        &logSize,
        &m_hitgroupShadowAlphaPG
    ));

    std::cout << "[OptiX] Program groups created (radiance + shadow + alpha)\n";

    return true;
}

bool OptixEngine::createDefaultSBT() {
    // Raygen record
    RaygenRecord raygenRecord;
    OPTIX_CHECK(optixSbtRecordPackHeader(m_raygenPG, &raygenRecord));

    cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&m_raygenRecord), sizeof(RaygenRecord));
    if (err != cudaSuccess) {
        std::cerr << "[OptiX] Failed to allocate raygen record: " << cudaGetErrorString(err) << "\n";
        return false;
    }
    err = cudaMemcpy(reinterpret_cast<void*>(m_raygenRecord), &raygenRecord,
                     sizeof(RaygenRecord), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "[OptiX] Failed to copy raygen record: " << cudaGetErrorString(err) << "\n";
        return false;
    }

    // Miss records: [0] = radiance (background), [1] = shadow (visibility)
    MissRecord missRecords[RAY_TYPE_COUNT];

    // Radiance miss - background (black)
    OPTIX_CHECK(optixSbtRecordPackHeader(m_missPG, &missRecords[RAY_TYPE_RADIANCE]));
    missRecords[RAY_TYPE_RADIANCE].backgroundColor = make_float3(0.0f, 0.0f, 0.0f);

    // Shadow miss - visibility
    OPTIX_CHECK(optixSbtRecordPackHeader(m_missShadowPG, &missRecords[RAY_TYPE_SHADOW]));
    missRecords[RAY_TYPE_SHADOW].backgroundColor = make_float3(0.0f, 0.0f, 0.0f);  // Not used

    err = cudaMalloc(reinterpret_cast<void**>(&m_missRecord), sizeof(MissRecord) * RAY_TYPE_COUNT);
    if (err != cudaSuccess) {
        std::cerr << "[OptiX] Failed to allocate miss records: " << cudaGetErrorString(err) << "\n";
        return false;
    }
    err = cudaMemcpy(reinterpret_cast<void*>(m_missRecord), missRecords,
                     sizeof(MissRecord) * RAY_TYPE_COUNT, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "[OptiX] Failed to copy miss records: " << cudaGetErrorString(err) << "\n";
        return false;
    }

    // Create default hitgroup records for both ray types
    // Layout: [radiance_0, shadow_0] per material
    HitGroupRecord hitgroupRecords[RAY_TYPE_COUNT];

    // Default material (gray diffuse)
    GpuMaterial defaultMat = {};
    defaultMat.baseColor = make_float4(0.8f, 0.8f, 0.8f, 1.0f);
    defaultMat.metallic = 0.0f;
    defaultMat.roughness = 0.5f;
    defaultMat.emissive = make_float3(0.0f, 0.0f, 0.0f);
    defaultMat.baseColorTex = 0;
    defaultMat.normalTex = 0;
    defaultMat.metallicRoughnessTex = 0;
    defaultMat.emissiveTex = 0;
    defaultMat.transmission = 0.0f;
    defaultMat.ior = 1.5f;
    defaultMat.transmissionTex = 0;
    defaultMat.attenuationColor = make_float3(1.0f, 1.0f, 1.0f);
    defaultMat.attenuationDistance = 0.0f;
    defaultMat.thickness = 0.0f;
    defaultMat.clearcoat = 0.0f;
    defaultMat.clearcoatRoughness = 0.0f;
    defaultMat.clearcoatTex = 0;
    defaultMat.clearcoatRoughnessTex = 0;
    defaultMat.clearcoatNormalTex = 0;
    defaultMat.sheenColor = make_float3(0.0f, 0.0f, 0.0f);
    defaultMat.sheenRoughness = 0.0f;
    defaultMat.sheenColorTex = 0;
    defaultMat.sheenRoughnessTex = 0;
    defaultMat.specularFactor = 1.0f;
    defaultMat.specularColorFactor = make_float3(1.0f, 1.0f, 1.0f);
    defaultMat.specularTex = 0;
    defaultMat.specularColorTex = 0;
    defaultMat.occlusionTex = 0;
    defaultMat.occlusionStrength = 1.0f;
    defaultMat.alphaMode = ALPHA_MODE_OPAQUE;
    defaultMat.alphaCutoff = 0.5f;
    defaultMat.doubleSided = 0;

    // Radiance hit group
    OPTIX_CHECK(optixSbtRecordPackHeader(m_hitgroupPG, &hitgroupRecords[RAY_TYPE_RADIANCE]));
    hitgroupRecords[RAY_TYPE_RADIANCE].material = defaultMat;
    hitgroupRecords[RAY_TYPE_RADIANCE].geometryIndex = 0;

    // Shadow hit group
    OPTIX_CHECK(optixSbtRecordPackHeader(m_hitgroupShadowPG, &hitgroupRecords[RAY_TYPE_SHADOW]));
    hitgroupRecords[RAY_TYPE_SHADOW].material = defaultMat;
    hitgroupRecords[RAY_TYPE_SHADOW].geometryIndex = 0;

    err = cudaMalloc(reinterpret_cast<void**>(&m_hitgroupRecords), sizeof(HitGroupRecord) * RAY_TYPE_COUNT);
    if (err != cudaSuccess) {
        std::cerr << "[OptiX] Failed to allocate hitgroup records: " << cudaGetErrorString(err) << "\n";
        return false;
    }
    err = cudaMemcpy(reinterpret_cast<void*>(m_hitgroupRecords), hitgroupRecords,
                     sizeof(HitGroupRecord) * RAY_TYPE_COUNT, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "[OptiX] Failed to copy hitgroup records: " << cudaGetErrorString(err) << "\n";
        return false;
    }
    m_hitgroupRecordCount = RAY_TYPE_COUNT;

    // Set up SBT
    m_sbt = {};
    m_sbt.raygenRecord = m_raygenRecord;
    m_sbt.missRecordBase = m_missRecord;
    m_sbt.missRecordStrideInBytes = sizeof(MissRecord);
    m_sbt.missRecordCount = RAY_TYPE_COUNT;
    m_sbt.hitgroupRecordBase = m_hitgroupRecords;
    m_sbt.hitgroupRecordStrideInBytes = sizeof(HitGroupRecord);
    m_sbt.hitgroupRecordCount = RAY_TYPE_COUNT;  // Will be updated by buildSBT

    std::cout << "[OptiX] Shader Binding Table created (radiance + shadow)\n";

    return true;
}

bool OptixEngine::buildSBT(const std::vector<GpuMaterial>& materials,
                            const std::vector<uint32_t>& geometryIndices) {
    if (materials.empty()) {
        std::cerr << "[OptiX] No materials provided for SBT\n";
        return false;
    }

    if (materials.size() != geometryIndices.size()) {
        std::cerr << "[OptiX] Material and geometry index count mismatch\n";
        return false;
    }

    // Free old hitgroup records
    if (m_hitgroupRecords) {
        cudaFree(reinterpret_cast<void*>(m_hitgroupRecords));
        m_hitgroupRecords = 0;
    }

    // Create hitgroup records for each material AND ray type
    // Layout: [mat0_radiance, mat0_shadow, mat1_radiance, mat1_shadow, ...]
    // Total records = materials.size() * RAY_TYPE_COUNT
    size_t numRecords = materials.size() * RAY_TYPE_COUNT;
    std::vector<HitGroupRecord> records(numRecords);

    for (size_t i = 0; i < materials.size(); ++i) {
        // Radiance hit group for this material
        size_t radianceIdx = i * RAY_TYPE_COUNT + RAY_TYPE_RADIANCE;
        OPTIX_CHECK(optixSbtRecordPackHeader(m_hitgroupPG, &records[radianceIdx]));
        records[radianceIdx].material = materials[i];
        records[radianceIdx].geometryIndex = geometryIndices[i];

        // Shadow hit group for this material
        size_t shadowIdx = i * RAY_TYPE_COUNT + RAY_TYPE_SHADOW;
        OPTIX_CHECK(optixSbtRecordPackHeader(m_hitgroupShadowPG, &records[shadowIdx]));
        records[shadowIdx].material = materials[i];
        records[shadowIdx].geometryIndex = geometryIndices[i];
    }

    // Allocate and copy
    size_t recordsSize = sizeof(HitGroupRecord) * records.size();
    cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&m_hitgroupRecords), recordsSize);
    if (err != cudaSuccess) {
        std::cerr << "[OptiX] Failed to allocate hitgroup records: " << cudaGetErrorString(err) << "\n";
        return false;
    }
    err = cudaMemcpy(reinterpret_cast<void*>(m_hitgroupRecords), records.data(),
                     recordsSize, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "[OptiX] Failed to copy hitgroup records: " << cudaGetErrorString(err) << "\n";
        return false;
    }

    m_hitgroupRecordCount = records.size();

    // Update SBT
    m_sbt.hitgroupRecordBase = m_hitgroupRecords;
    m_sbt.hitgroupRecordStrideInBytes = sizeof(HitGroupRecord);
    m_sbt.hitgroupRecordCount = static_cast<unsigned int>(m_hitgroupRecordCount);

    std::cout << "[OptiX] SBT updated with " << materials.size() << " materials (" << numRecords << " records)\n";

    return true;
}

void OptixEngine::shutdown() {
    if (m_pickBuffer) {
        cudaFree(reinterpret_cast<void*>(m_pickBuffer));
        m_pickBuffer = 0;
    }
    if (m_launchParamsBuffer) {
        cudaFree(reinterpret_cast<void*>(m_launchParamsBuffer));
        m_launchParamsBuffer = 0;
    }
    if (m_raygenRecord) {
        cudaFree(reinterpret_cast<void*>(m_raygenRecord));
        m_raygenRecord = 0;
    }
    if (m_missRecord) {
        cudaFree(reinterpret_cast<void*>(m_missRecord));
        m_missRecord = 0;
    }
    if (m_hitgroupRecords) {
        cudaFree(reinterpret_cast<void*>(m_hitgroupRecords));
        m_hitgroupRecords = 0;
    }
    if (m_pipeline) {
        optixPipelineDestroy(m_pipeline);
        m_pipeline = nullptr;
    }
    if (m_raygenPG) {
        optixProgramGroupDestroy(m_raygenPG);
        m_raygenPG = nullptr;
    }
    if (m_missPG) {
        optixProgramGroupDestroy(m_missPG);
        m_missPG = nullptr;
    }
    if (m_missShadowPG) {
        optixProgramGroupDestroy(m_missShadowPG);
        m_missShadowPG = nullptr;
    }
    if (m_hitgroupPG) {
        optixProgramGroupDestroy(m_hitgroupPG);
        m_hitgroupPG = nullptr;
    }
    if (m_hitgroupShadowPG) {
        optixProgramGroupDestroy(m_hitgroupShadowPG);
        m_hitgroupShadowPG = nullptr;
    }
    if (m_hitgroupAlphaPG) {
        optixProgramGroupDestroy(m_hitgroupAlphaPG);
        m_hitgroupAlphaPG = nullptr;
    }
    if (m_hitgroupShadowAlphaPG) {
        optixProgramGroupDestroy(m_hitgroupShadowAlphaPG);
        m_hitgroupShadowAlphaPG = nullptr;
    }
    if (m_raygenModule) {
        optixModuleDestroy(m_raygenModule);
        m_raygenModule = nullptr;
    }
    if (m_missModule) {
        optixModuleDestroy(m_missModule);
        m_missModule = nullptr;
    }
    if (m_closesthitModule) {
        optixModuleDestroy(m_closesthitModule);
        m_closesthitModule = nullptr;
    }
    if (m_anyhitModule) {
        optixModuleDestroy(m_anyhitModule);
        m_anyhitModule = nullptr;
    }
    if (m_context) {
        optixDeviceContextDestroy(m_context);
        m_context = nullptr;
    }

    std::cout << "[OptiX] Shutdown complete\n";
}

void OptixEngine::setDimensions(uint32_t width, uint32_t height) {
    m_width = width;
    m_height = height;
}

void OptixEngine::setCamera(const CameraParams& camera) {
    m_launchParams.camera = camera;
}

void OptixEngine::setSceneHandle(OptixTraversableHandle handle) {
    m_launchParams.scene_handle = handle;
}

void OptixEngine::setGeometryBuffers(CUdeviceptr* vertexBuffers, CUdeviceptr* indexBuffers) {
    m_launchParams.vertex_buffers = vertexBuffers;
    m_launchParams.index_buffers = indexBuffers;
}

void OptixEngine::render(float4* outputBuffer, cudaStream_t stream) {
    // Update launch params
    m_launchParams.output_buffer = outputBuffer;
    m_launchParams.width = m_width;
    m_launchParams.height = m_height;
    m_launchParams.frame_index = m_frameIndex;
    m_launchParams.random_seed = m_frameIndex * 17 + 31;  // Simple per-frame seed
    // Note: accumulated_frames is managed by caller via resetAccumulation()

    // Copy params to device
    cudaMemcpyAsync(
        reinterpret_cast<void*>(m_launchParamsBuffer),
        &m_launchParams,
        sizeof(LaunchParams),
        cudaMemcpyHostToDevice,
        stream
    );

    // Launch OptiX
    OptixResult result = optixLaunch(
        m_pipeline,
        stream,
        m_launchParamsBuffer,
        sizeof(LaunchParams),
        &m_sbt,
        m_width,
        m_height,
        1  // depth
    );

    if (result != OPTIX_SUCCESS) {
        std::cerr << "[OptiX] Launch failed: " << optixGetErrorName(result) << "\n";
    }

    m_frameIndex++;
    m_launchParams.accumulated_frames++;
}

void OptixEngine::setPointLights(GpuPointLight* lights, uint32_t count) {
    m_launchParams.point_lights = lights;
    m_launchParams.point_light_count = count;
}

void OptixEngine::setDirectionalLights(GpuDirectionalLight* lights, uint32_t count) {
    m_launchParams.directional_lights = lights;
    m_launchParams.directional_light_count = count;
}

void OptixEngine::setAreaLights(GpuAreaLight* lights, uint32_t count) {
    m_launchParams.area_lights = lights;
    m_launchParams.area_light_count = count;
}

void OptixEngine::setEnvironmentMap(cudaTextureObject_t envMap, float intensity) {
    m_launchParams.environment_map = envMap;
    m_launchParams.environment_intensity = intensity;
}

void OptixEngine::setEnvironmentCDF(cudaTextureObject_t conditionalCDF,
                                    cudaTextureObject_t marginalCDF,
                                    uint32_t width, uint32_t height,
                                    float totalLuminance) {
    m_launchParams.env_conditional_cdf = conditionalCDF;
    m_launchParams.env_marginal_cdf = marginalCDF;
    m_launchParams.env_width = width;
    m_launchParams.env_height = height;
    m_launchParams.env_total_luminance = totalLuminance;
}

void OptixEngine::setQualityMode(QualityMode mode) {
    m_launchParams.quality_mode = mode;
}

void OptixEngine::setSamplesPerPixel(uint32_t spp) {
    m_launchParams.samples_per_pixel = spp > 0 ? spp : 1;
}

uint32_t OptixEngine::getSamplesPerPixel() const {
    return m_launchParams.samples_per_pixel;
}

void OptixEngine::setSelectedInstanceId(uint32_t instanceId) {
    m_launchParams.selected_instance_id = instanceId;
}

void OptixEngine::setAccumulationBuffer(float4* buffer) {
    m_launchParams.accumulation_buffer = buffer;
}

void OptixEngine::resetAccumulation() {
    m_launchParams.accumulated_frames = 0;
}

uint32_t OptixEngine::getAccumulatedFrames() const {
    return m_launchParams.accumulated_frames;
}

uint32_t OptixEngine::pickInstance(uint32_t screenX, uint32_t screenY, cudaStream_t stream) {
    if (!m_pipeline || !m_pickBuffer || screenX >= m_width || screenY >= m_height) {
        return UINT32_MAX;
    }

    // Initialize pick result to "no hit"
    uint32_t noHit = UINT32_MAX;
    cudaMemcpyAsync(reinterpret_cast<void*>(m_pickBuffer), &noHit, sizeof(uint32_t),
                    cudaMemcpyHostToDevice, stream);

    // Set up pick mode
    m_launchParams.pick_mode = 1;
    m_launchParams.pick_x = screenX;
    m_launchParams.pick_y = screenY;

    // Copy launch params to device
    cudaMemcpyAsync(reinterpret_cast<void*>(m_launchParamsBuffer),
                    &m_launchParams, sizeof(LaunchParams),
                    cudaMemcpyHostToDevice, stream);

    // Launch with 1x1 dimensions (single ray)
    OPTIX_CHECK(optixLaunch(
        m_pipeline,
        stream,
        m_launchParamsBuffer,
        sizeof(LaunchParams),
        &m_sbt,
        1, 1, 1  // Single pixel launch
    ));

    // Restore normal mode
    m_launchParams.pick_mode = 0;

    // Read back result
    uint32_t result;
    cudaMemcpyAsync(&result, reinterpret_cast<void*>(m_pickBuffer), sizeof(uint32_t),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    return result;
}

} // namespace spectra
