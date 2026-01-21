#include "optix_engine.h"
#include <optix_stubs.h>
#include <optix_function_table_definition.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cstring>

namespace spectra {

// SBT record structures with proper alignment
struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) RaygenRecord {
    char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    // No additional data for Phase 1
};

struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) MissRecord {
    char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    // No additional data for Phase 1
};

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

    return true;
}

bool OptixEngine::createPipeline(const std::filesystem::path& ptxDir) {
    // Set pipeline compile options
    m_pipelineCompileOptions = {};
    m_pipelineCompileOptions.usesMotionBlur = false;
    m_pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
    m_pipelineCompileOptions.numPayloadValues = 0;  // No payload for Phase 1
    m_pipelineCompileOptions.numAttributeValues = 0;  // No attributes for Phase 1
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

    // Create program groups
    if (!createProgramGroups()) {
        return false;
    }

    // Link pipeline
    OptixPipelineLinkOptions linkOptions = {};
    linkOptions.maxTraceDepth = 1;  // Phase 1: no actual tracing

    OptixProgramGroup programGroups[] = { m_raygenPG, m_missPG };

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

    // Set stack sizes (conservative for Phase 1)
    OPTIX_CHECK(optixPipelineSetStackSize(
        m_pipeline,
        2 * 1024,   // Direct callable stack size
        2 * 1024,   // Continuation stack size
        2 * 1024,   // Continuation stack size from traversal
        2           // Max traversable depth
    ));

    // Create Shader Binding Table
    if (!createSBT()) {
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

    // Miss program group
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

    std::cout << "[OptiX] Program groups created\n";

    return true;
}

bool OptixEngine::createSBT() {
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

    // Miss record
    MissRecord missRecord;
    OPTIX_CHECK(optixSbtRecordPackHeader(m_missPG, &missRecord));

    err = cudaMalloc(reinterpret_cast<void**>(&m_missRecord), sizeof(MissRecord));
    if (err != cudaSuccess) {
        std::cerr << "[OptiX] Failed to allocate miss record: " << cudaGetErrorString(err) << "\n";
        return false;
    }
    err = cudaMemcpy(reinterpret_cast<void*>(m_missRecord), &missRecord,
                     sizeof(MissRecord), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "[OptiX] Failed to copy miss record: " << cudaGetErrorString(err) << "\n";
        return false;
    }

    // Set up SBT
    m_sbt = {};
    m_sbt.raygenRecord = m_raygenRecord;
    m_sbt.missRecordBase = m_missRecord;
    m_sbt.missRecordStrideInBytes = sizeof(MissRecord);
    m_sbt.missRecordCount = 1;

    // No hit groups for Phase 1
    m_sbt.hitgroupRecordBase = 0;
    m_sbt.hitgroupRecordStrideInBytes = 0;
    m_sbt.hitgroupRecordCount = 0;

    std::cout << "[OptiX] Shader Binding Table created\n";

    return true;
}

void OptixEngine::shutdown() {
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
    if (m_raygenModule) {
        optixModuleDestroy(m_raygenModule);
        m_raygenModule = nullptr;
    }
    if (m_missModule) {
        optixModuleDestroy(m_missModule);
        m_missModule = nullptr;
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

void OptixEngine::render(float4* outputBuffer, cudaStream_t stream) {
    // Update launch params
    m_launchParams.output_buffer = outputBuffer;
    m_launchParams.width = m_width;
    m_launchParams.height = m_height;
    m_launchParams.frame_index = m_frameIndex;

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
}

} // namespace spectra
