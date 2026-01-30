#pragma once

#include "shared_types.h"
#include "geometry_manager.h"
#include "material_manager.h"
#include "optix_engine.h"
#include <optix.h>
#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <cstdint>

namespace spectra {

// GAS (Geometry Acceleration Structure) info
struct GasInfo {
    OptixTraversableHandle handle;
    CUdeviceptr buffer;
    size_t bufferSize;
    GeometryHandle geometryHandle;
    MaterialHandle materialHandle;
};

// Scene instance info
struct SceneInstance {
    uint32_t gasIndex;          // Index into m_gasList
    float transform[12];        // 3x4 row-major transform
    uint32_t sbtOffset;         // SBT hit group offset
    uint32_t instanceId;        // Custom instance ID for shader lookup
};

class SceneManager {
public:
    SceneManager() = default;
    ~SceneManager();

    // Non-copyable
    SceneManager(const SceneManager&) = delete;
    SceneManager& operator=(const SceneManager&) = delete;

    // Set managers (must be called before use)
    void setOptixEngine(OptixEngine* engine) { m_optixEngine = engine; }
    void setGeometryManager(GeometryManager* geomMgr) { m_geometryManager = geomMgr; }
    void setMaterialManager(MaterialManager* matMgr) { m_materialManager = matMgr; }

    // Build GAS for a mesh and add to scene
    // Returns GAS index on success, or UINT32_MAX on failure
    uint32_t addMesh(const MeshData& mesh);

    // Add instance of a GAS
    uint32_t addInstance(uint32_t gasIndex, const float* transform);

    // Build IAS from all instances
    bool buildIAS();

    // Get scene traversable handle
    OptixTraversableHandle getSceneHandle() const { return m_iasHandle; }

    // Get vertex/index buffer arrays for shader access
    CUdeviceptr* getVertexBuffers() { return m_d_vertexBuffers; }
    CUdeviceptr* getIndexBuffers() { return m_d_indexBuffers; }

    // Clear all scene data
    void clear();

    // Get statistics
    size_t getGasCount() const { return m_gasList.size(); }
    size_t getInstanceCount() const { return m_instances.size(); }
    size_t getGpuMemoryUsage() const;

    // Get instances (for UI scene tree)
    const std::vector<SceneInstance>& getInstances() const { return m_instances; }

    // Get material handle for an instance
    MaterialHandle getMaterialHandle(uint32_t instanceId) const {
        if (instanceId >= m_instances.size()) return INVALID_MATERIAL_HANDLE;
        uint32_t gasIndex = m_instances[instanceId].gasIndex;
        if (gasIndex >= m_gasList.size()) return INVALID_MATERIAL_HANDLE;
        return m_gasList[gasIndex].materialHandle;
    }

    // Track loaded model paths for serialization
    void addLoadedModelPath(const std::string& path) { m_loadedModelPaths.push_back(path); }
    const std::vector<std::string>& getLoadedModelPaths() const { return m_loadedModelPaths; }

    // Build/update SBT with scene materials
    bool updateSBT();

private:
    bool buildGAS(const GpuGeometry* geom, GasInfo& gas);
    bool compactGAS(GasInfo& gas);
    void freeGAS(GasInfo& gas);

    OptixEngine* m_optixEngine = nullptr;
    GeometryManager* m_geometryManager = nullptr;
    MaterialManager* m_materialManager = nullptr;

    // GAS list (one per unique mesh)
    std::vector<GasInfo> m_gasList;

    // Instance list
    std::vector<SceneInstance> m_instances;

    // Loaded model paths for serialization
    std::vector<std::string> m_loadedModelPaths;

    // IAS (top-level acceleration structure)
    OptixTraversableHandle m_iasHandle = 0;
    CUdeviceptr m_iasBuffer = 0;
    size_t m_iasBufferSize = 0;
    CUdeviceptr m_instanceBuffer = 0;

    // Buffer pointer arrays for shader access
    CUdeviceptr* m_d_vertexBuffers = nullptr;
    CUdeviceptr* m_d_indexBuffers = nullptr;
    size_t m_bufferArraySize = 0;

    // Temporary buffers for BVH building
    CUdeviceptr m_tempBuffer = 0;
    size_t m_tempBufferSize = 0;
};

} // namespace spectra
