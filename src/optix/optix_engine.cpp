#include "optix_engine.h"
#include "path_guide_grid.h"
#include <optix_stubs.h>
#include <optix_function_table_definition.h>
#ifdef _WIN32
#undef min
#undef max
#endif
#include <optix_stack_size.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cstring>
#include <cstddef>
#include <cmath>
#include <cfloat>

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

    // Allocate double-buffered pinned launch params for async H2D upload
    for (int i = 0; i < 2; i++) {
        err = cudaMallocHost(&m_pinnedLaunchParams[i], sizeof(LaunchParams));
        if (err != cudaSuccess) {
            std::cerr << "[OptiX] Failed to allocate pinned launch params[" << i << "]: " << cudaGetErrorString(err) << "\n";
            return false;
        }
    }
    m_pinnedLaunchIdx = 0;

    // Initialize launch params with defaults
    m_launchParams = {};
    m_launchParams.output_buffer = nullptr;
    m_launchParams.accumulation_buffer = nullptr;
    m_launchParams.aov_albedo_buffer = nullptr;
    m_launchParams.aov_normal_buffer = nullptr;
    m_launchParams.accumulated_frames = 0;
    m_launchParams.scene_handle = 0;
    m_launchParams.vertex_buffers = nullptr;
    m_launchParams.index_buffers = nullptr;
    m_launchParams.materials = nullptr;
    m_launchParams.instance_material_indices = nullptr;
    m_launchParams.instance_transforms = nullptr;
    m_launchParams.instance_normal_transforms = nullptr;
    m_launchParams.instance_light_indices = nullptr;
    m_launchParams.area_light_tris = nullptr;
    m_launchParams.env_selection_weight = 0.0f;

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
    m_launchParams.total_light_luminance = 0.0f;

    // Initialize environment map
    m_launchParams.environment_map = 0;
    m_launchParams.environment_intensity = 1.0f;

    // Initialize environment CDF (for importance sampling)
    m_launchParams.env_alias_prob = nullptr;
    m_launchParams.env_alias_idx = nullptr;
    m_launchParams.env_pmf = nullptr;
    m_launchParams.env_width = 0;
    m_launchParams.env_height = 0;
    m_launchParams.env_total_luminance = 0.0f;

    // Initialize quality settings (firefly clamp set via setQualityMode)
    m_launchParams.quality_mode = QUALITY_BALANCED;
    m_launchParams.samples_per_pixel = 1;  // 1 SPP for real-time; use ] to increase
    m_launchParams.firefly_clamp = 100.0f;
    m_launchParams.max_bounce_depth = 8;

    // Initialize selection (UINT32_MAX = no selection)
    m_launchParams.selected_instance_id = UINT32_MAX;

    // Initialize pick mode
    m_launchParams.pick_mode = 0;
    m_launchParams.pick_x = 0;
    m_launchParams.pick_y = 0;

    // Path guide grid (device-resident cell table, cleared until set)
    m_launchParams.path_guide_data = nullptr;
    m_launchParams.path_guide_num_levels = 0;
    m_launchParams.path_guide_entry_stride = 0;
    m_launchParams.path_guide_base_resolution = 0;
    m_launchParams.path_guide_per_level_scale = 1.0f;
    m_launchParams.path_guide_bounds_min[0] = m_launchParams.path_guide_bounds_min[1] = m_launchParams.path_guide_bounds_min[2] = 0.0f;
    m_launchParams.path_guide_bounds_max[0] = m_launchParams.path_guide_bounds_max[1] = m_launchParams.path_guide_bounds_max[2] = 0.0f;
    m_launchParams.path_guide_hash_keys = nullptr;
    m_launchParams.path_guide_hash_values = nullptr;
    m_launchParams.path_guide_hash_table_size = 0;
    m_launchParams.path_guide_hash_shift = 64;
    m_launchParams.path_guide_cell_keys = nullptr;
    m_launchParams.path_guide_cell_counter = nullptr;
    m_launchParams.path_guide_cell_capacity = 0;
    m_launchParams.path_guide_enabled = 0;     // Disabled by default
    m_launchParams.path_guide_mis_weight = 0.5f;  // Conservative: guide needs time to converge
    // Adaptive level parameters (defaults, updated when grid is set)
    m_launchParams.path_guide_start_level = 2;
    m_launchParams.path_guide_min_level = 1;
    m_launchParams.path_guide_max_level = 6;

    // Allocate pick result buffer (PickResultBuffer: instanceId + hitX/Y/Z)
    err = cudaMalloc(reinterpret_cast<void**>(&m_pickBuffer), sizeof(PickResultBuffer));
    if (err != cudaSuccess) {
        std::cerr << "[OptiX] Failed to allocate pick buffer: " << cudaGetErrorString(err) << "\n";
        return false;
    }
    m_launchParams.pick_result = reinterpret_cast<PickResultBuffer*>(m_pickBuffer);

    // Allocate path guide debug stats buffer (6 counters)
    err = cudaMalloc(reinterpret_cast<void**>(&m_pathGuideDebugStats), 6 * sizeof(uint32_t));
    if (err != cudaSuccess) {
        std::cerr << "[OptiX] Failed to allocate path guide debug stats: " << cudaGetErrorString(err) << "\n";
        return false;
    }
    cudaMemset(m_pathGuideDebugStats, 0, 6 * sizeof(uint32_t));
    m_launchParams.path_guide_debug_stats = m_pathGuideDebugStats;
    m_launchParams.path_guide_debug_enabled = 0;  // Disabled by default

    return true;
}

bool OptixEngine::createPipeline(const std::filesystem::path& ptxDir) {
    m_pipelineCompileOptions = {};
    m_pipelineCompileOptions.usesMotionBlur = false;
    m_pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
    m_pipelineCompileOptions.numPayloadValues = 5;      // hit info: t, instance, prim, baryU, baryV
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

    // Link pipeline. The path tracer is iterative (loop in raygen), so the
    // only nesting is raygen -> radiance/shadow: trace depth 2. This keeps the
    // continuation stack tiny compared to the previous recursive design
    // (depth 11 with full shading state live across each trace).
    OptixPipelineLinkOptions linkOptions = {};
    linkOptions.maxTraceDepth = 2;

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

    // Compute proper stack sizes from actual program requirements
    {
        OptixStackSizes stackSizes = {};
        for (auto pg : programGroups) {
            OPTIX_CHECK(optixUtilAccumulateStackSizes(pg, &stackSizes, m_pipeline));
        }

        uint32_t directCallableStackSizeFromTraversal = 0;
        uint32_t directCallableStackSizeFromState = 0;
        uint32_t continuationStackSize = 0;
        OPTIX_CHECK(optixUtilComputeStackSizes(
            &stackSizes,
            linkOptions.maxTraceDepth,        // maxTraceDepth
            0,                                // maxCCDepth (no continuation callables)
            0,                                // maxDCDepth (no direct callables)
            &directCallableStackSizeFromTraversal,
            &directCallableStackSizeFromState,
            &continuationStackSize
        ));

        std::cout << "[OptiX] Computed stack sizes: DC_traversal=" << directCallableStackSizeFromTraversal
                  << " DC_state=" << directCallableStackSizeFromState
                  << " continuation=" << continuationStackSize << "\n";

        OPTIX_CHECK(optixPipelineSetStackSize(
            m_pipeline,
            directCallableStackSizeFromTraversal,
            directCallableStackSizeFromState,
            continuationStackSize,
            2  // Max traversable depth (single-level instancing)
        ));
    }

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

    OptixProgramGroupOptions pgOptions = {};

    // Raygen program group
    OptixProgramGroupDesc raygenDesc = {};
    raygenDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    raygenDesc.raygen.module = m_raygenModule;
    raygenDesc.raygen.entryFunctionName = "__raygen__simple";

    logSize = sizeof(log);
    OPTIX_CHECK_LOG(optixProgramGroupCreate(
        m_context, &raygenDesc, 1, &pgOptions, log, &logSize, &m_raygenPG));

    // Miss program group (radiance — flags "no hit")
    OptixProgramGroupDesc missDesc = {};
    missDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    missDesc.miss.module = m_missModule;
    missDesc.miss.entryFunctionName = "__miss__radiance";

    logSize = sizeof(log);
    OPTIX_CHECK_LOG(optixProgramGroupCreate(
        m_context, &missDesc, 1, &pgOptions, log, &logSize, &m_missPG));

    // Miss program group (shadow - visibility)
    OptixProgramGroupDesc missShadowDesc = {};
    missShadowDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    missShadowDesc.miss.module = m_missModule;
    missShadowDesc.miss.entryFunctionName = "__miss__shadow";

    logSize = sizeof(log);
    OPTIX_CHECK_LOG(optixProgramGroupCreate(
        m_context, &missShadowDesc, 1, &pgOptions, log, &logSize, &m_missShadowPG));

    // Hit group (radiance, opaque): hit-info CH, no anyhit
    OptixProgramGroupDesc hitgroupDesc = {};
    hitgroupDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hitgroupDesc.hitgroup.moduleCH = m_closesthitModule;
    hitgroupDesc.hitgroup.entryFunctionNameCH = "__closesthit__hitinfo";
    hitgroupDesc.hitgroup.moduleAH = nullptr;
    hitgroupDesc.hitgroup.entryFunctionNameAH = nullptr;
    hitgroupDesc.hitgroup.moduleIS = nullptr;
    hitgroupDesc.hitgroup.entryFunctionNameIS = nullptr;

    logSize = sizeof(log);
    OPTIX_CHECK_LOG(optixProgramGroupCreate(
        m_context, &hitgroupDesc, 1, &pgOptions, log, &logSize, &m_hitgroupPG));

    // Hit group (shadow, opaque)
    OptixProgramGroupDesc hitgroupShadowDesc = {};
    hitgroupShadowDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hitgroupShadowDesc.hitgroup.moduleCH = m_closesthitModule;
    hitgroupShadowDesc.hitgroup.entryFunctionNameCH = "__closesthit__shadow";
    hitgroupShadowDesc.hitgroup.moduleAH = nullptr;
    hitgroupShadowDesc.hitgroup.entryFunctionNameAH = nullptr;
    hitgroupShadowDesc.hitgroup.moduleIS = nullptr;
    hitgroupShadowDesc.hitgroup.entryFunctionNameIS = nullptr;

    logSize = sizeof(log);
    OPTIX_CHECK_LOG(optixProgramGroupCreate(
        m_context, &hitgroupShadowDesc, 1, &pgOptions, log, &logSize, &m_hitgroupShadowPG));

    // Hit group (radiance, alpha-masked): hit-info CH + alpha anyhit
    OptixProgramGroupDesc hitgroupAlphaDesc = {};
    hitgroupAlphaDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hitgroupAlphaDesc.hitgroup.moduleCH = m_closesthitModule;
    hitgroupAlphaDesc.hitgroup.entryFunctionNameCH = "__closesthit__hitinfo";
    hitgroupAlphaDesc.hitgroup.moduleAH = m_anyhitModule;
    hitgroupAlphaDesc.hitgroup.entryFunctionNameAH = "__anyhit__alpha";
    hitgroupAlphaDesc.hitgroup.moduleIS = nullptr;
    hitgroupAlphaDesc.hitgroup.entryFunctionNameIS = nullptr;

    logSize = sizeof(log);
    OPTIX_CHECK_LOG(optixProgramGroupCreate(
        m_context, &hitgroupAlphaDesc, 1, &pgOptions, log, &logSize, &m_hitgroupAlphaPG));

    // Hit group (shadow, alpha-masked): shadow CH + alpha anyhit (cut-out shadows)
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
        m_context, &hitgroupShadowAlphaDesc, 1, &pgOptions, log, &logSize, &m_hitgroupShadowAlphaPG));

    std::cout << "[OptiX] Program groups created (radiance + shadow, opaque + alpha)\n";

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

    // Miss records: [0] = radiance, [1] = shadow
    MissRecord missRecords[RAY_TYPE_COUNT];
    OPTIX_CHECK(optixSbtRecordPackHeader(m_missPG, &missRecords[RAY_TYPE_RADIANCE]));
    missRecords[RAY_TYPE_RADIANCE].backgroundColor = make_float3(0.0f, 0.0f, 0.0f);
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

    // Default hitgroup records (one material, all ray types)
    GpuMaterial defaultMat = {};
    defaultMat.baseColor = make_float4(0.8f, 0.8f, 0.8f, 1.0f);
    defaultMat.metallic = 0.0f;
    defaultMat.roughness = 0.5f;
    defaultMat.ior = 1.5f;
    defaultMat.attenuationColor = make_float3(1.0f, 1.0f, 1.0f);
    defaultMat.sheenColor = make_float3(0.0f, 0.0f, 0.0f);
    defaultMat.specularFactor = 1.0f;
    defaultMat.specularColorFactor = make_float3(1.0f, 1.0f, 1.0f);
    defaultMat.occlusionStrength = 1.0f;
    defaultMat.alphaMode = ALPHA_MODE_OPAQUE;
    defaultMat.alphaCutoff = 0.5f;

    std::vector<GpuMaterial> defaults = { defaultMat };
    std::vector<uint32_t> geomIndices = { 0 };

    // Set up SBT skeleton; buildSBT fills the hitgroup records
    m_sbt = {};
    m_sbt.raygenRecord = m_raygenRecord;
    m_sbt.missRecordBase = m_missRecord;
    m_sbt.missRecordStrideInBytes = sizeof(MissRecord);
    m_sbt.missRecordCount = RAY_TYPE_COUNT;

    if (!buildSBT(defaults, geomIndices)) {
        return false;
    }

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

    // Create hitgroup records for each material AND ray type.
    // Layout: [mat0_radiance, mat0_shadow, mat1_radiance, mat1_shadow, ...]
    // Alpha-masked materials get the anyhit-enabled hit groups so MASK
    // cutouts work for both visibility and shadows; opaque materials use the
    // anyhit-free groups (and their GASes set GEOMETRY_FLAG_DISABLE_ANYHIT).
    // Transmissive (glass) materials additionally take the SHADOW anyhit
    // group so shadow rays pass through them (transparent shadows) — shadow
    // rays trace with ENFORCE_ANYHIT, which overrides the per-GAS disable,
    // so this works even when a material turns transmissive at runtime.
    size_t numRecords = materials.size() * RAY_TYPE_COUNT;
    std::vector<HitGroupRecord> records(numRecords);
    m_materialAlphaModes.resize(materials.size());
    m_materialShadowAnyhit.resize(materials.size());

    for (size_t i = 0; i < materials.size(); ++i) {
        bool masked = (materials[i].alphaMode == ALPHA_MODE_MASK);
        bool shadowAnyhit = masked || (materials[i].transmission > 0.0f);
        m_materialAlphaModes[i] = materials[i].alphaMode;
        m_materialShadowAnyhit[i] = shadowAnyhit ? 1u : 0u;

        size_t radianceIdx = i * RAY_TYPE_COUNT + RAY_TYPE_RADIANCE;
        OPTIX_CHECK(optixSbtRecordPackHeader(
            masked ? m_hitgroupAlphaPG : m_hitgroupPG, &records[radianceIdx]));
        records[radianceIdx].material = materials[i];
        records[radianceIdx].geometryIndex = geometryIndices[i];

        size_t shadowIdx = i * RAY_TYPE_COUNT + RAY_TYPE_SHADOW;
        OPTIX_CHECK(optixSbtRecordPackHeader(
            shadowAnyhit ? m_hitgroupShadowAlphaPG : m_hitgroupShadowPG, &records[shadowIdx]));
        records[shadowIdx].material = materials[i];
        records[shadowIdx].geometryIndex = geometryIndices[i];
    }

    // Allocate and copy hitgroup records
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

    // Upload the material array for raygen-side shading (grow-only buffer)
    if (materials.size() > m_materialsBufferCapacity) {
        if (m_materialsBuffer) {
            cudaFree(reinterpret_cast<void*>(m_materialsBuffer));
            m_materialsBuffer = 0;
        }
        err = cudaMalloc(reinterpret_cast<void**>(&m_materialsBuffer),
                         sizeof(GpuMaterial) * materials.size());
        if (err != cudaSuccess) {
            std::cerr << "[OptiX] Failed to allocate materials buffer: " << cudaGetErrorString(err) << "\n";
            return false;
        }
        m_materialsBufferCapacity = materials.size();
    }
    err = cudaMemcpy(reinterpret_cast<void*>(m_materialsBuffer), materials.data(),
                     sizeof(GpuMaterial) * materials.size(), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "[OptiX] Failed to copy materials buffer: " << cudaGetErrorString(err) << "\n";
        return false;
    }
    m_launchParams.materials = reinterpret_cast<const GpuMaterial*>(m_materialsBuffer);

    // Update SBT
    m_sbt.hitgroupRecordBase = m_hitgroupRecords;
    m_sbt.hitgroupRecordStrideInBytes = sizeof(HitGroupRecord);
    m_sbt.hitgroupRecordCount = static_cast<unsigned int>(m_hitgroupRecordCount);

    std::cout << "[OptiX] SBT updated with " << materials.size() << " materials ("
              << numRecords << " records)\n";

    return true;
}

bool OptixEngine::updateMaterialRecord(uint32_t materialSlot, const GpuMaterial& material,
                                       cudaStream_t stream) {
    if (!m_hitgroupRecords || !m_materialsBuffer) return false;
    if ((size_t)materialSlot >= m_materialAlphaModes.size()) return false;
    if ((size_t)(materialSlot + 1) * RAY_TYPE_COUNT > m_hitgroupRecordCount) return false;

    // A change of alphaMode — or of the transmissive class (glass toggled
    // on/off changes the shadow hit-group) — switches hit-group program
    // headers; needs a full SBT rebuild (caller falls back).
    if (material.alphaMode != m_materialAlphaModes[materialSlot]) return false;
    bool shadowAnyhit = (material.alphaMode == ALPHA_MODE_MASK) || (material.transmission > 0.0f);
    if ((shadowAnyhit ? 1u : 0u) != m_materialShadowAnyhit[materialSlot]) return false;

    // Patch the material payload of both ray-type records in place. The SBT
    // header (program selection) is unchanged, so this is safe to do
    // stream-ordered without stalling the pipeline — unlike the previous full
    // rebuild which required a device sync on every slider drag.
    for (uint32_t rt = 0; rt < RAY_TYPE_COUNT; ++rt) {
        size_t recordIdx = (size_t)materialSlot * RAY_TYPE_COUNT + rt;
        CUdeviceptr dst = m_hitgroupRecords
            + recordIdx * sizeof(HitGroupRecord)
            + offsetof(HitGroupRecord, material);
        cudaMemcpyAsync(reinterpret_cast<void*>(dst), &material, sizeof(GpuMaterial),
                        cudaMemcpyHostToDevice, stream);
    }

    // Patch the raygen-side material array entry
    CUdeviceptr matDst = m_materialsBuffer + (size_t)materialSlot * sizeof(GpuMaterial);
    cudaMemcpyAsync(reinterpret_cast<void*>(matDst), &material, sizeof(GpuMaterial),
                    cudaMemcpyHostToDevice, stream);

    return true;
}

void OptixEngine::shutdown() {
    if (m_pickBuffer) {
        cudaFree(reinterpret_cast<void*>(m_pickBuffer));
        m_pickBuffer = 0;
    }
    if (m_pathGuideDebugStats) {
        cudaFree(m_pathGuideDebugStats);
        m_pathGuideDebugStats = nullptr;
    }
    if (m_launchParamsBuffer) {
        cudaFree(reinterpret_cast<void*>(m_launchParamsBuffer));
        m_launchParamsBuffer = 0;
    }
    for (int i = 0; i < 2; i++) {
        if (m_pinnedLaunchParams[i]) {
            cudaFreeHost(m_pinnedLaunchParams[i]);
            m_pinnedLaunchParams[i] = nullptr;
        }
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
    if (m_materialsBuffer) {
        cudaFree(reinterpret_cast<void*>(m_materialsBuffer));
        m_materialsBuffer = 0;
        m_materialsBufferCapacity = 0;
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

void OptixEngine::setInstanceData(const float* transforms,
                                  const float* normalTransforms,
                                  const uint32_t* materialIndices) {
    m_launchParams.instance_transforms = transforms;
    m_launchParams.instance_normal_transforms = normalTransforms;
    m_launchParams.instance_material_indices = materialIndices;
}

void OptixEngine::setInstanceLightIndices(const uint32_t* lightIndices) {
    m_launchParams.instance_light_indices = lightIndices;
}

void OptixEngine::setAreaLightTriangles(const float4* tris) {
    m_launchParams.area_light_tris = tris;
}

void OptixEngine::setEnvSelectionWeight(float weight) {
    m_launchParams.env_selection_weight = weight;
}

void OptixEngine::setMaxBounceDepth(uint32_t depth) {
    m_launchParams.max_bounce_depth = depth;
}

void OptixEngine::render(float4* outputBuffer, cudaStream_t stream) {
    // Update launch params
    m_launchParams.output_buffer = outputBuffer;
    m_launchParams.width = m_width;
    m_launchParams.height = m_height;
    m_launchParams.frame_index = m_frameIndex;

    // Precompute per-frame constants to avoid transcendentals on GPU
    m_launchParams.tan_half_fov_y = std::tan(m_launchParams.camera.fovY * 0.5f);
    m_launchParams.tan_half_fov_x = m_launchParams.tan_half_fov_y * m_launchParams.camera.aspectRatio;
    m_launchParams.pixel_world_size = (2.0f * m_launchParams.tan_half_fov_y) / static_cast<float>(m_height);
    // Note: accumulated_frames is managed by caller via resetAccumulation()

    // Copy to pinned staging buffer, then async DMA to device.
    // cudaMemcpyAsync from pageable memory implicitly synchronizes the stream,
    // blocking the CPU for the entire previous frame's GPU time. Pinned memory
    // eliminates this stall, letting the CPU stay ahead of the GPU.
    int idx = m_pinnedLaunchIdx;
    m_pinnedLaunchIdx = 1 - m_pinnedLaunchIdx;
    std::memcpy(m_pinnedLaunchParams[idx], &m_launchParams, sizeof(LaunchParams));
    cudaMemcpyAsync(
        reinterpret_cast<void*>(m_launchParamsBuffer),
        m_pinnedLaunchParams[idx],
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

void OptixEngine::setTotalLightLuminance(float luminance) {
    m_launchParams.total_light_luminance = luminance;
}

void OptixEngine::setEnvironmentMap(cudaTextureObject_t envMap, float intensity) {
    m_launchParams.environment_map = envMap;
    m_launchParams.environment_intensity = intensity;
}

void OptixEngine::setEnvironmentImportance(const float* aliasProb,
                                           const unsigned int* aliasIdx,
                                           const float* pmf,
                                           uint32_t width, uint32_t height,
                                           float totalLuminance) {
    m_launchParams.env_alias_prob = aliasProb;
    m_launchParams.env_alias_idx = aliasIdx;
    m_launchParams.env_pmf = pmf;
    m_launchParams.env_width = width;
    m_launchParams.env_height = height;
    m_launchParams.env_total_luminance = totalLuminance;
}

void OptixEngine::setQualityMode(QualityMode mode) {
    m_launchParams.quality_mode = mode;
    // Firefly clamp AND depth cap scale with the quality goal. The depth cap is
    // a BIASED path truncation (raygen drops the continuation past it with no RR
    // compensation), so ACCURATE — which advertises unbiasedness — must run a
    // cap high enough that Russian roulette (from depth 3) terminates paths
    // first in practice. Real-time modes keep a tight cap as a deliberate
    // perf/bias tradeoff. (Previously the cap stayed at its init value of 8 in
    // every mode because setMaxBounceDepth was never called — ACCURATE silently
    // truncated multiply-scattered specular/glass GI.)
    switch (mode) {
        case QUALITY_FAST:
        case QUALITY_BALANCED:
            m_launchParams.firefly_clamp = 100.0f;
            m_launchParams.max_bounce_depth = 8;
            break;
        case QUALITY_HIGH:
            m_launchParams.firefly_clamp = 1000.0f;
            m_launchParams.max_bounce_depth = 16;
            break;
        case QUALITY_ACCURATE:
            m_launchParams.firefly_clamp = FLT_MAX;
            m_launchParams.max_bounce_depth = 32;  // RR bounds cost; cap only backstops pathological mirror chains
            break;
        default:
            m_launchParams.firefly_clamp = 100.0f;
            m_launchParams.max_bounce_depth = 8;
            break;
    }
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

void OptixEngine::setAOVBuffers(float4* albedo, float4* normal) {
    m_launchParams.aov_albedo_buffer = albedo;
    m_launchParams.aov_normal_buffer = normal;
}

void OptixEngine::resetAccumulation() {
    m_launchParams.accumulated_frames = 0;
}

uint32_t OptixEngine::getAccumulatedFrames() const {
    return m_launchParams.accumulated_frames;
}

void OptixEngine::setPathGuideGridDescriptor(const SparsePathGuideDescriptor* sparse) {
    auto zeroBounds = [this]() {
        m_launchParams.path_guide_bounds_min[0] = m_launchParams.path_guide_bounds_min[1] = m_launchParams.path_guide_bounds_min[2] = 0.0f;
        m_launchParams.path_guide_bounds_max[0] = m_launchParams.path_guide_bounds_max[1] = m_launchParams.path_guide_bounds_max[2] = 0.0f;
    };
    if (!sparse) {
        m_launchParams.path_guide_data = nullptr;
        m_launchParams.path_guide_num_levels = 0;
        m_launchParams.path_guide_entry_stride = 0;
        m_launchParams.path_guide_base_resolution = 0;
        m_launchParams.path_guide_per_level_scale = 1.0f;
        zeroBounds();
        std::memset(m_launchParams.path_guide_level_resolutions, 0, sizeof(m_launchParams.path_guide_level_resolutions));
        m_launchParams.path_guide_hash_keys = nullptr;
        m_launchParams.path_guide_hash_values = nullptr;
        m_launchParams.path_guide_hash_table_size = 0;
        m_launchParams.path_guide_hash_shift = 64;
        m_launchParams.path_guide_cell_keys = nullptr;
        m_launchParams.path_guide_cell_counter = nullptr;
        m_launchParams.path_guide_cell_capacity = 0;
    } else {
        m_launchParams.path_guide_data = sparse->data;
        m_launchParams.path_guide_num_levels = sparse->num_levels;
        m_launchParams.path_guide_entry_stride = sparse->entry_stride;
        m_launchParams.path_guide_base_resolution = sparse->base_resolution;
        m_launchParams.path_guide_per_level_scale = sparse->per_level_scale;
        m_launchParams.path_guide_bounds_min[0] = sparse->bounds_min[0];
        m_launchParams.path_guide_bounds_min[1] = sparse->bounds_min[1];
        m_launchParams.path_guide_bounds_min[2] = sparse->bounds_min[2];
        m_launchParams.path_guide_bounds_max[0] = sparse->bounds_max[0];
        m_launchParams.path_guide_bounds_max[1] = sparse->bounds_max[1];
        m_launchParams.path_guide_bounds_max[2] = sparse->bounds_max[2];
        // Precompute level resolutions to avoid powf() on GPU
        for (uint32_t l = 0; l < 16; l++) {
            float res = std::floor(static_cast<float>(sparse->base_resolution) *
                std::pow(sparse->per_level_scale, static_cast<float>(l)));
            m_launchParams.path_guide_level_resolutions[l] = (res < 1.0f) ? 1u : static_cast<uint32_t>(res);
        }
        m_launchParams.path_guide_hash_keys = sparse->hash_keys;
        m_launchParams.path_guide_hash_values = sparse->hash_values;
        m_launchParams.path_guide_hash_table_size = sparse->hash_table_size;
        m_launchParams.path_guide_hash_shift = sparse->hash_shift;
        m_launchParams.path_guide_cell_keys = sparse->cell_keys;
        m_launchParams.path_guide_cell_counter = sparse->cell_counter;
        m_launchParams.path_guide_cell_capacity = sparse->cell_capacity;
    }
}

void OptixEngine::setPathGuideEnabled(bool enabled) {
    m_launchParams.path_guide_enabled = enabled ? 1u : 0u;
}

void OptixEngine::setPathGuideLevelConfig(uint32_t startLevel, uint32_t minLevel, uint32_t maxLevel) {
    m_launchParams.path_guide_start_level = startLevel;
    m_launchParams.path_guide_min_level = minLevel;
    m_launchParams.path_guide_max_level = maxLevel;
}

void OptixEngine::setPathGuideMISWeight(float weight) {
    m_launchParams.path_guide_mis_weight = weight;
}

void OptixEngine::setPathGuideDebugEnabled(bool enabled) {
    m_launchParams.path_guide_debug_enabled = enabled ? 1u : 0u;
}

void OptixEngine::resetPathGuideStats(cudaStream_t stream) {
    if (m_pathGuideDebugStats) {
        if (stream) {
            cudaMemsetAsync(m_pathGuideDebugStats, 0, 6 * sizeof(uint32_t), stream);
        } else {
            cudaMemset(m_pathGuideDebugStats, 0, 6 * sizeof(uint32_t));
        }
    }
}

OptixEngine::PathGuideStats OptixEngine::readPathGuideStats() {
    PathGuideStats stats = {};
    if (m_pathGuideDebugStats) {
        uint32_t data[6] = {0};
        cudaMemcpy(data, m_pathGuideDebugStats, 6 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        stats.attempts = data[0];
        stats.cellFound = data[1];
        stats.validLobe = data[2];
        stats.belowHorizon = data[3];
        stats.contributed = data[4];
        stats.bsdfSampled = data[5];
    }
    return stats;
}

uint32_t OptixEngine::pickInstance(uint32_t screenX, uint32_t screenY, cudaStream_t stream) {
    PickResultBuffer result = pickInstanceAndPosition(screenX, screenY, stream);
    return result.instanceId;
}

PickResultBuffer OptixEngine::pickInstanceAndPosition(uint32_t screenX, uint32_t screenY, cudaStream_t stream) {
    PickResultBuffer noHitResult = {};
    noHitResult.instanceId = UINT32_MAX;
    noHitResult.hitX = 0.0f;
    noHitResult.hitY = 0.0f;
    noHitResult.hitZ = 0.0f;

    if (!m_pipeline || !m_pickBuffer || screenX >= m_width || screenY >= m_height) {
        return noHitResult;
    }

    // Initialize pick result to "no hit"
    cudaMemcpyAsync(reinterpret_cast<void*>(m_pickBuffer), &noHitResult, sizeof(PickResultBuffer),
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
    OptixResult launchResult = optixLaunch(
        m_pipeline,
        stream,
        m_launchParamsBuffer,
        sizeof(LaunchParams),
        &m_sbt,
        1, 1, 1  // Single pixel launch
    );

    // Restore normal mode
    m_launchParams.pick_mode = 0;

    if (launchResult != OPTIX_SUCCESS) {
        std::cerr << "[OptiX] Pick launch failed: " << optixGetErrorName(launchResult) << "\n";
        return noHitResult;
    }

    // Read back full result
    PickResultBuffer result;
    cudaMemcpyAsync(&result, reinterpret_cast<void*>(m_pickBuffer), sizeof(PickResultBuffer),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    return result;
}

} // namespace spectra
