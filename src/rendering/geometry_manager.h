#pragma once

#include "shared_types.h"
#include <cuda_runtime.h>
#include <unordered_map>
#include <vector>
#include <cstdint>

namespace spectra {

// Handle to a GPU geometry buffer
using GeometryHandle = uint32_t;
constexpr GeometryHandle INVALID_GEOMETRY_HANDLE = UINT32_MAX;

// GPU buffer info
struct GpuGeometry {
    CUdeviceptr vertexBuffer;       // GpuVertex array
    CUdeviceptr indexBuffer;        // uint32_t indices
    uint32_t vertexCount;
    uint32_t indexCount;
    uint32_t refCount;              // Reference counting
};

class GeometryManager {
public:
    GeometryManager() = default;
    ~GeometryManager();

    // Non-copyable
    GeometryManager(const GeometryManager&) = delete;
    GeometryManager& operator=(const GeometryManager&) = delete;

    // Upload mesh to GPU
    // Returns handle on success, INVALID_GEOMETRY_HANDLE on failure
    GeometryHandle upload(const MeshData& mesh);

    // Increment reference count for a geometry
    void addRef(GeometryHandle handle);

    // Decrement reference count, free if zero
    void release(GeometryHandle handle);

    // Get geometry info
    const GpuGeometry* get(GeometryHandle handle) const;

    // Free all GPU resources
    void clear();

    // Get device pointer to vertex buffer
    CUdeviceptr getVertexBuffer(GeometryHandle handle) const;

    // Get device pointer to index buffer
    CUdeviceptr getIndexBuffer(GeometryHandle handle) const;

    // Get total GPU memory used
    size_t getGpuMemoryUsage() const;

    // Get number of active geometries
    size_t getGeometryCount() const { return m_geometries.size(); }

private:
    std::unordered_map<GeometryHandle, GpuGeometry> m_geometries;
    GeometryHandle m_nextHandle = 0;
    size_t m_totalMemory = 0;
};

} // namespace spectra
