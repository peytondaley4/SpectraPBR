#include "geometry_manager.h"
#include <iostream>

namespace spectra {

GeometryManager::~GeometryManager() {
    clear();
}

GeometryHandle GeometryManager::upload(const MeshData& mesh) {
    if (mesh.vertices.empty() || mesh.indices.empty()) {
        std::cerr << "[GeometryManager] Cannot upload empty mesh\n";
        return INVALID_GEOMETRY_HANDLE;
    }

    GpuGeometry geom = {};
    geom.vertexCount = static_cast<uint32_t>(mesh.vertices.size());
    geom.indexCount = static_cast<uint32_t>(mesh.indices.size());
    geom.refCount = 1;

    size_t vertexSize = mesh.vertices.size() * sizeof(GpuVertex);
    size_t indexSize = mesh.indices.size() * sizeof(uint32_t);

    // Allocate vertex buffer
    cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&geom.vertexBuffer), vertexSize);
    if (err != cudaSuccess) {
        std::cerr << "[GeometryManager] Failed to allocate vertex buffer: "
                  << cudaGetErrorString(err) << "\n";
        return INVALID_GEOMETRY_HANDLE;
    }

    // Copy vertex data
    err = cudaMemcpy(reinterpret_cast<void*>(geom.vertexBuffer),
                     mesh.vertices.data(), vertexSize, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "[GeometryManager] Failed to copy vertex data: "
                  << cudaGetErrorString(err) << "\n";
        cudaFree(reinterpret_cast<void*>(geom.vertexBuffer));
        return INVALID_GEOMETRY_HANDLE;
    }

    // Allocate index buffer
    err = cudaMalloc(reinterpret_cast<void**>(&geom.indexBuffer), indexSize);
    if (err != cudaSuccess) {
        std::cerr << "[GeometryManager] Failed to allocate index buffer: "
                  << cudaGetErrorString(err) << "\n";
        cudaFree(reinterpret_cast<void*>(geom.vertexBuffer));
        return INVALID_GEOMETRY_HANDLE;
    }

    // Copy index data
    err = cudaMemcpy(reinterpret_cast<void*>(geom.indexBuffer),
                     mesh.indices.data(), indexSize, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "[GeometryManager] Failed to copy index data: "
                  << cudaGetErrorString(err) << "\n";
        cudaFree(reinterpret_cast<void*>(geom.vertexBuffer));
        cudaFree(reinterpret_cast<void*>(geom.indexBuffer));
        return INVALID_GEOMETRY_HANDLE;
    }

    GeometryHandle handle = m_nextHandle++;
    m_geometries[handle] = geom;
    m_totalMemory += vertexSize + indexSize;

    std::cout << "[GeometryManager] Uploaded geometry " << handle
              << " (" << geom.vertexCount << " vertices, "
              << geom.indexCount << " indices, "
              << (vertexSize + indexSize) / 1024 << " KB)\n";

    return handle;
}

void GeometryManager::addRef(GeometryHandle handle) {
    auto it = m_geometries.find(handle);
    if (it != m_geometries.end()) {
        it->second.refCount++;
    }
}

void GeometryManager::release(GeometryHandle handle) {
    auto it = m_geometries.find(handle);
    if (it == m_geometries.end()) {
        return;
    }

    it->second.refCount--;
    if (it->second.refCount == 0) {
        GpuGeometry& geom = it->second;

        size_t freedMemory = geom.vertexCount * sizeof(GpuVertex) +
                            geom.indexCount * sizeof(uint32_t);

        if (geom.vertexBuffer) {
            cudaFree(reinterpret_cast<void*>(geom.vertexBuffer));
        }
        if (geom.indexBuffer) {
            cudaFree(reinterpret_cast<void*>(geom.indexBuffer));
        }

        m_totalMemory -= freedMemory;
        m_geometries.erase(it);

        std::cout << "[GeometryManager] Released geometry " << handle
                  << " (freed " << freedMemory / 1024 << " KB)\n";
    }
}

const GpuGeometry* GeometryManager::get(GeometryHandle handle) const {
    auto it = m_geometries.find(handle);
    if (it == m_geometries.end()) {
        return nullptr;
    }
    return &it->second;
}

void GeometryManager::clear() {
    for (auto& [handle, geom] : m_geometries) {
        if (geom.vertexBuffer) {
            cudaFree(reinterpret_cast<void*>(geom.vertexBuffer));
        }
        if (geom.indexBuffer) {
            cudaFree(reinterpret_cast<void*>(geom.indexBuffer));
        }
    }
    m_geometries.clear();
    m_totalMemory = 0;
    std::cout << "[GeometryManager] Cleared all geometries\n";
}

CUdeviceptr GeometryManager::getVertexBuffer(GeometryHandle handle) const {
    const GpuGeometry* geom = get(handle);
    return geom ? geom->vertexBuffer : 0;
}

CUdeviceptr GeometryManager::getIndexBuffer(GeometryHandle handle) const {
    const GpuGeometry* geom = get(handle);
    return geom ? geom->indexBuffer : 0;
}

size_t GeometryManager::getGpuMemoryUsage() const {
    return m_totalMemory;
}

} // namespace spectra
