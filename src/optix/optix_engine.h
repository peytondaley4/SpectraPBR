#pragma once

#include "shared_types.h"
#include <optix.h>
#include <cuda_runtime.h>
#include <filesystem>
#include <cstdint>
#include <vector>

namespace spectra {

struct SparsePathGuideDescriptor;

// OptiX error checking macro
#define OPTIX_CHECK(call)                                                        \
    do {                                                                         \
        OptixResult res = call;                                                  \
        if (res != OPTIX_SUCCESS) {                                              \
            std::cerr << "[OptiX] Error: " << optixGetErrorName(res)             \
                      << " (" << optixGetErrorString(res) << ")"                 \
                      << " at " << __FILE__ << ":" << __LINE__ << "\n";          \
            return false;                                                        \
        }                                                                        \
    } while (0)

// Variant for log output
#define OPTIX_CHECK_LOG(call)                                                    \
    do {                                                                         \
        OptixResult res = call;                                                  \
        if (res != OPTIX_SUCCESS) {                                              \
            std::cerr << "[OptiX] Error: " << optixGetErrorName(res)             \
                      << " (" << optixGetErrorString(res) << ")"                 \
                      << " at " << __FILE__ << ":" << __LINE__ << "\n";          \
            if (logSize > 1) {                                                   \
                std::cerr << "[OptiX] Log: " << log << "\n";                     \
            }                                                                    \
            return false;                                                        \
        } else if (logSize > 1) {                                                \
            std::cout << "[OptiX] Log: " << log << "\n";                         \
        }                                                                        \
    } while (0)

class OptixEngine {
public:
    OptixEngine() = default;
    ~OptixEngine();

    // Non-copyable
    OptixEngine(const OptixEngine&) = delete;
    OptixEngine& operator=(const OptixEngine&) = delete;

    // Initialize OptiX context on existing CUDA context
    bool init(CUcontext cudaContext);

    // Load PTX modules and create pipeline
    // ptxDir: directory containing raygen.ptx, miss.ptx, closesthit.ptx
    bool createPipeline(const std::filesystem::path& ptxDir);

    // Shutdown and cleanup
    void shutdown();

    // Set render dimensions (call before render if changed)
    void setDimensions(uint32_t width, uint32_t height);

    // Set camera parameters
    void setCamera(const CameraParams& camera);

    // Set scene traversable handle
    void setSceneHandle(OptixTraversableHandle handle);

    // Set geometry buffers for shader access
    void setGeometryBuffers(CUdeviceptr* vertexBuffers, CUdeviceptr* indexBuffers);

    // Per-instance data for raygen-side shading (device pointers, indexed by
    // instance ID): transforms (12 floats), inverse-transpose normal
    // transforms (12 floats), and instance -> material slot indices.
    void setInstanceData(const float* transforms,
                         const float* normalTransforms,
                         const uint32_t* materialIndices);

    // Instance -> area light index map (device pointer, UINT32_MAX = no light)
    void setInstanceLightIndices(const uint32_t* lightIndices);

    // Mesh-light triangle buffer (device pointer, 3 float4 per triangle)
    void setAreaLightTriangles(const float4* tris);

    // Environment selection weight for the NEE light pick
    void setEnvSelectionWeight(float weight);

    // Maximum path length (vertices after the camera)
    void setMaxBounceDepth(uint32_t depth);
    uint32_t getMaxBounceDepth() const { return m_launchParams.max_bounce_depth; }

    // Build/update SBT with materials. Also uploads the material array for
    // raygen-side shading and selects alpha-test hit groups per material.
    // materials: array of GpuMaterial for each hit group
    // geometryIndices: geometry index per material for buffer lookup
    bool buildSBT(const std::vector<GpuMaterial>& materials,
                  const std::vector<uint32_t>& geometryIndices);

    // Fast-path update of a single material slot (SBT records + material
    // array) without rebuilding the whole SBT. Returns false when the change
    // affects hit-group selection (alphaMode) — caller must do a full rebuild.
    bool updateMaterialRecord(uint32_t materialSlot, const GpuMaterial& material,
                              cudaStream_t stream);

    // Render to output buffer
    // outputBuffer: CUDA device pointer to float4 array
    // stream: CUDA stream for async execution
    void render(float4* outputBuffer, cudaStream_t stream);

    // Get current frame index
    uint32_t getFrameIndex() const { return m_frameIndex; }

    // Get OptiX context (for BVH building)
    OptixDeviceContext getContext() const { return m_context; }

    // Get pipeline compile options (for BVH building)
    const OptixPipelineCompileOptions& getPipelineCompileOptions() const {
        return m_pipelineCompileOptions;
    }

    // Set lighting data
    void setPointLights(GpuPointLight* lights, uint32_t count);
    void setDirectionalLights(GpuDirectionalLight* lights, uint32_t count);
    void setAreaLights(GpuAreaLight* lights, uint32_t count);
    void setTotalLightLuminance(float luminance);

    // Set environment map
    void setEnvironmentMap(cudaTextureObject_t envMap, float intensity);

    // Set environment map importance sampling CDFs
    void setEnvironmentCDF(cudaTextureObject_t conditionalCDF, 
                          cudaTextureObject_t marginalCDF,
                          uint32_t width, uint32_t height,
                          float totalLuminance);

    // Set quality mode
    void setQualityMode(QualityMode mode);

    // Set samples per pixel (higher = less noise, slower render)
    void setSamplesPerPixel(uint32_t spp);
    uint32_t getSamplesPerPixel() const;

    // Set selected instance for UI highlighting
    void setSelectedInstanceId(uint32_t instanceId);

    // Pick instance at screen coordinates (returns UINT32_MAX if no hit)
    uint32_t pickInstance(uint32_t screenX, uint32_t screenY, cudaStream_t stream = 0);

    // Pick instance and world-space hit position at screen coordinates
    PickResultBuffer pickInstanceAndPosition(uint32_t screenX, uint32_t screenY, cudaStream_t stream = 0);

    // Accumulation buffer for progressive AA
    void setAccumulationBuffer(float4* buffer);
    void resetAccumulation();
    uint32_t getAccumulatedFrames() const;

    // Path guide grid (device-resident cell table)
    void setPathGuideGridDescriptor(const SparsePathGuideDescriptor* sparse);
    void setPathGuideEnabled(bool enabled);
    void setPathGuideMISWeight(float weight);
    void setPathGuideLevelConfig(uint32_t startLevel, uint32_t minLevel, uint32_t maxLevel);

    // Path guide debugging
    struct PathGuideStats {
        uint32_t attempts = 0;      // Total indirect lighting attempts
        uint32_t cellFound = 0;     // Times a cell was found for position
        uint32_t validLobe = 0;     // Times cell had valid vMF lobes
        uint32_t belowHorizon = 0;  // Times sample was below horizon
        uint32_t contributed = 0;   // Times contribution was added
        uint32_t bsdfSampled = 0;   // Times BSDF sampling was used (vs guide)
    };
    void setPathGuideDebugEnabled(bool enabled);
    void resetPathGuideStats(cudaStream_t stream = nullptr);
    PathGuideStats readPathGuideStats();

private:
    bool createModule(const std::filesystem::path& ptxPath, OptixModule* module);
    bool createProgramGroups();
    bool createDefaultSBT();

    // OptiX context
    OptixDeviceContext m_context = nullptr;

    // Pipeline
    OptixModule m_raygenModule = nullptr;
    OptixModule m_missModule = nullptr;
    OptixModule m_closesthitModule = nullptr;
    OptixModule m_anyhitModule = nullptr;           // For alpha testing
    OptixPipeline m_pipeline = nullptr;
    OptixPipelineCompileOptions m_pipelineCompileOptions = {};

    // Program groups
    OptixProgramGroup m_raygenPG = nullptr;
    OptixProgramGroup m_missPG = nullptr;           // Radiance miss (hit-info "no hit")
    OptixProgramGroup m_missShadowPG = nullptr;     // Shadow miss (visibility)
    OptixProgramGroup m_hitgroupPG = nullptr;       // Radiance hit (opaque)
    OptixProgramGroup m_hitgroupShadowPG = nullptr; // Shadow hit (opaque)
    OptixProgramGroup m_hitgroupAlphaPG = nullptr;  // Radiance hit (alpha tested)
    OptixProgramGroup m_hitgroupShadowAlphaPG = nullptr; // Shadow hit (alpha tested)

    // Shader Binding Table
    OptixShaderBindingTable m_sbt = {};
    CUdeviceptr m_raygenRecord = 0;
    CUdeviceptr m_missRecord = 0;
    CUdeviceptr m_hitgroupRecords = 0;
    size_t m_hitgroupRecordCount = 0;

    // Device material array for raygen-side shading (mirrors SBT materials)
    CUdeviceptr m_materialsBuffer = 0;
    size_t m_materialsBufferCapacity = 0;   // in materials
    std::vector<uint32_t> m_materialAlphaModes;  // per slot, for fast-path updates

    // Launch parameters
    LaunchParams m_launchParams = {};
    CUdeviceptr m_launchParamsBuffer = 0;

    // Double-buffered pinned launch params for truly async H2D upload.
    // cudaMemcpyAsync from pageable memory implicitly synchronizes the stream,
    // blocking the CPU for the entire previous frame's GPU time (~50ms at 20 FPS).
    // Pinned memory eliminates this per-frame CPU stall.
    LaunchParams* m_pinnedLaunchParams[2] = { nullptr, nullptr };
    int m_pinnedLaunchIdx = 0;

    // State
    uint32_t m_width = 0;
    uint32_t m_height = 0;
    uint32_t m_frameIndex = 0;

    // Pick buffer (single uint32_t on device)
    CUdeviceptr m_pickBuffer = 0;

    // Path guide debug stats buffer (6 uint32_t counters on device)
    uint32_t* m_pathGuideDebugStats = nullptr;
};

} // namespace spectra
