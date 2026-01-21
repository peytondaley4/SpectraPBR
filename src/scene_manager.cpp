#include "scene_manager.h"
#include <iostream>
#include <cstring>

namespace spectra {

SceneManager::~SceneManager() {
    clear();
}

uint32_t SceneManager::addMesh(const MeshData& mesh) {
    if (!m_optixEngine || !m_geometryManager || !m_materialManager) {
        std::cerr << "[SceneManager] Managers not set\n";
        return UINT32_MAX;
    }

    // Upload geometry
    GeometryHandle geomHandle = m_geometryManager->upload(mesh);
    if (geomHandle == INVALID_GEOMETRY_HANDLE) {
        return UINT32_MAX;
    }

    const GpuGeometry* geom = m_geometryManager->get(geomHandle);
    if (!geom) {
        return UINT32_MAX;
    }

    // Build GAS
    GasInfo gas = {};
    gas.geometryHandle = geomHandle;
    gas.materialHandle = mesh.materialIndex;

    if (!buildGAS(geom, gas)) {
        m_geometryManager->release(geomHandle);
        return UINT32_MAX;
    }

    // Compact GAS
    if (!compactGAS(gas)) {
        freeGAS(gas);
        m_geometryManager->release(geomHandle);
        return UINT32_MAX;
    }

    uint32_t gasIndex = static_cast<uint32_t>(m_gasList.size());
    m_gasList.push_back(gas);

    std::cout << "[SceneManager] Added mesh " << gasIndex
              << " (GAS: " << gas.bufferSize / 1024 << " KB)\n";

    return gasIndex;
}

uint32_t SceneManager::addInstance(uint32_t gasIndex, const float* transform) {
    if (gasIndex >= m_gasList.size()) {
        std::cerr << "[SceneManager] Invalid GAS index\n";
        return UINT32_MAX;
    }

    SceneInstance inst = {};
    inst.gasIndex = gasIndex;
    inst.sbtOffset = 0;  // Will be set when building IAS
    inst.instanceId = static_cast<uint32_t>(m_instances.size());

    if (transform) {
        memcpy(inst.transform, transform, sizeof(inst.transform));
    } else {
        // Identity transform
        inst.transform[0] = 1.0f; inst.transform[1] = 0.0f; inst.transform[2] = 0.0f; inst.transform[3] = 0.0f;
        inst.transform[4] = 0.0f; inst.transform[5] = 1.0f; inst.transform[6] = 0.0f; inst.transform[7] = 0.0f;
        inst.transform[8] = 0.0f; inst.transform[9] = 0.0f; inst.transform[10] = 1.0f; inst.transform[11] = 0.0f;
    }

    uint32_t instanceIndex = static_cast<uint32_t>(m_instances.size());
    m_instances.push_back(inst);

    return instanceIndex;
}

bool SceneManager::buildGAS(const GpuGeometry* geom, GasInfo& gas) {
    OptixDeviceContext context = m_optixEngine->getContext();

    // Set up build input
    OptixBuildInput buildInput = {};
    buildInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

    // Triangle vertex data
    CUdeviceptr d_vertices = geom->vertexBuffer;
    buildInput.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    buildInput.triangleArray.vertexStrideInBytes = sizeof(GpuVertex);
    buildInput.triangleArray.numVertices = geom->vertexCount;
    buildInput.triangleArray.vertexBuffers = &d_vertices;

    // Triangle index data
    CUdeviceptr d_indices = geom->indexBuffer;
    buildInput.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    buildInput.triangleArray.indexStrideInBytes = 3 * sizeof(uint32_t);
    buildInput.triangleArray.numIndexTriplets = geom->indexCount / 3;
    buildInput.triangleArray.indexBuffer = d_indices;

    // Flags (one per SBT record)
    uint32_t triangleFlags = OPTIX_GEOMETRY_FLAG_NONE;
    buildInput.triangleArray.flags = &triangleFlags;
    buildInput.triangleArray.numSbtRecords = 1;

    // Get build options
    OptixAccelBuildOptions buildOptions = {};
    buildOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION |
                              OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    buildOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

    // Get memory requirements
    OptixAccelBufferSizes bufferSizes = {};
    OptixResult res = optixAccelComputeMemoryUsage(
        context,
        &buildOptions,
        &buildInput,
        1,
        &bufferSizes
    );
    if (res != OPTIX_SUCCESS) {
        std::cerr << "[SceneManager] Failed to compute GAS memory: " << optixGetErrorName(res) << "\n";
        return false;
    }

    // Allocate temp buffer if needed
    if (bufferSizes.tempSizeInBytes > m_tempBufferSize) {
        if (m_tempBuffer) {
            cudaFree(reinterpret_cast<void*>(m_tempBuffer));
        }
        cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&m_tempBuffer), bufferSizes.tempSizeInBytes);
        if (err != cudaSuccess) {
            std::cerr << "[SceneManager] Failed to allocate temp buffer: " << cudaGetErrorString(err) << "\n";
            return false;
        }
        m_tempBufferSize = bufferSizes.tempSizeInBytes;
    }

    // Allocate output buffer (initial, will be compacted)
    cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&gas.buffer), bufferSizes.outputSizeInBytes);
    if (err != cudaSuccess) {
        std::cerr << "[SceneManager] Failed to allocate GAS buffer: " << cudaGetErrorString(err) << "\n";
        return false;
    }
    gas.bufferSize = bufferSizes.outputSizeInBytes;

    // Allocate compaction size output
    CUdeviceptr d_compactedSize;
    err = cudaMalloc(reinterpret_cast<void**>(&d_compactedSize), sizeof(size_t));
    if (err != cudaSuccess) {
        cudaFree(reinterpret_cast<void*>(gas.buffer));
        gas.buffer = 0;
        std::cerr << "[SceneManager] Failed to allocate compaction size: " << cudaGetErrorString(err) << "\n";
        return false;
    }

    // Set up compaction size emit
    OptixAccelEmitDesc emitDesc = {};
    emitDesc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emitDesc.result = d_compactedSize;

    // Build GAS
    res = optixAccelBuild(
        context,
        0,  // stream
        &buildOptions,
        &buildInput,
        1,
        m_tempBuffer,
        bufferSizes.tempSizeInBytes,
        gas.buffer,
        bufferSizes.outputSizeInBytes,
        &gas.handle,
        &emitDesc,
        1
    );

    if (res != OPTIX_SUCCESS) {
        cudaFree(reinterpret_cast<void*>(gas.buffer));
        cudaFree(reinterpret_cast<void*>(d_compactedSize));
        gas.buffer = 0;
        std::cerr << "[SceneManager] Failed to build GAS: " << optixGetErrorName(res) << "\n";
        return false;
    }

    // Get compacted size
    size_t compactedSize;
    cudaMemcpy(&compactedSize, reinterpret_cast<void*>(d_compactedSize), sizeof(size_t), cudaMemcpyDeviceToHost);
    cudaFree(reinterpret_cast<void*>(d_compactedSize));

    // Store for later compaction
    gas.bufferSize = compactedSize;

    return true;
}

bool SceneManager::compactGAS(GasInfo& gas) {
    OptixDeviceContext context = m_optixEngine->getContext();

    // Allocate compacted buffer
    CUdeviceptr compactedBuffer;
    cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&compactedBuffer), gas.bufferSize);
    if (err != cudaSuccess) {
        std::cerr << "[SceneManager] Failed to allocate compacted GAS: " << cudaGetErrorString(err) << "\n";
        return false;
    }

    // Compact
    OptixResult res = optixAccelCompact(
        context,
        0,  // stream
        gas.handle,
        compactedBuffer,
        gas.bufferSize,
        &gas.handle
    );

    if (res != OPTIX_SUCCESS) {
        cudaFree(reinterpret_cast<void*>(compactedBuffer));
        std::cerr << "[SceneManager] Failed to compact GAS: " << optixGetErrorName(res) << "\n";
        return false;
    }

    // Free old buffer and use compacted
    cudaFree(reinterpret_cast<void*>(gas.buffer));
    gas.buffer = compactedBuffer;

    return true;
}

void SceneManager::freeGAS(GasInfo& gas) {
    if (gas.buffer) {
        cudaFree(reinterpret_cast<void*>(gas.buffer));
        gas.buffer = 0;
    }
    gas.handle = 0;
}

bool SceneManager::buildIAS() {
    if (m_instances.empty()) {
        std::cout << "[SceneManager] No instances to build IAS\n";
        return true;
    }

    if (!m_optixEngine) {
        std::cerr << "[SceneManager] OptixEngine not set\n";
        return false;
    }

    OptixDeviceContext context = m_optixEngine->getContext();

    // Free old IAS
    if (m_iasBuffer) {
        cudaFree(reinterpret_cast<void*>(m_iasBuffer));
        m_iasBuffer = 0;
    }
    if (m_instanceBuffer) {
        cudaFree(reinterpret_cast<void*>(m_instanceBuffer));
        m_instanceBuffer = 0;
    }

    // Build OptixInstance array
    std::vector<OptixInstance> optixInstances(m_instances.size());

    for (size_t i = 0; i < m_instances.size(); ++i) {
        const SceneInstance& inst = m_instances[i];
        const GasInfo& gas = m_gasList[inst.gasIndex];

        OptixInstance& oi = optixInstances[i];
        memset(&oi, 0, sizeof(OptixInstance));

        // Copy transform (row-major)
        memcpy(oi.transform, inst.transform, sizeof(oi.transform));

        oi.instanceId = inst.instanceId;
        oi.sbtOffset = inst.gasIndex;  // Use GAS index as SBT offset (one hit group per GAS)
        oi.visibilityMask = 0xFF;
        oi.flags = OPTIX_INSTANCE_FLAG_NONE;
        oi.traversableHandle = gas.handle;
    }

    // Upload instances to GPU
    size_t instancesSize = sizeof(OptixInstance) * optixInstances.size();
    cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&m_instanceBuffer), instancesSize);
    if (err != cudaSuccess) {
        std::cerr << "[SceneManager] Failed to allocate instance buffer: " << cudaGetErrorString(err) << "\n";
        return false;
    }
    err = cudaMemcpy(reinterpret_cast<void*>(m_instanceBuffer), optixInstances.data(),
                     instancesSize, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "[SceneManager] Failed to copy instances: " << cudaGetErrorString(err) << "\n";
        return false;
    }

    // Set up build input
    OptixBuildInput buildInput = {};
    buildInput.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    buildInput.instanceArray.instances = m_instanceBuffer;
    buildInput.instanceArray.numInstances = static_cast<unsigned int>(optixInstances.size());

    // Build options
    OptixAccelBuildOptions buildOptions = {};
    buildOptions.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    buildOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

    // Get memory requirements
    OptixAccelBufferSizes bufferSizes = {};
    OptixResult res = optixAccelComputeMemoryUsage(
        context,
        &buildOptions,
        &buildInput,
        1,
        &bufferSizes
    );
    if (res != OPTIX_SUCCESS) {
        std::cerr << "[SceneManager] Failed to compute IAS memory: " << optixGetErrorName(res) << "\n";
        return false;
    }

    // Ensure temp buffer is large enough
    if (bufferSizes.tempSizeInBytes > m_tempBufferSize) {
        if (m_tempBuffer) {
            cudaFree(reinterpret_cast<void*>(m_tempBuffer));
        }
        err = cudaMalloc(reinterpret_cast<void**>(&m_tempBuffer), bufferSizes.tempSizeInBytes);
        if (err != cudaSuccess) {
            std::cerr << "[SceneManager] Failed to allocate temp buffer: " << cudaGetErrorString(err) << "\n";
            return false;
        }
        m_tempBufferSize = bufferSizes.tempSizeInBytes;
    }

    // Allocate IAS buffer
    err = cudaMalloc(reinterpret_cast<void**>(&m_iasBuffer), bufferSizes.outputSizeInBytes);
    if (err != cudaSuccess) {
        std::cerr << "[SceneManager] Failed to allocate IAS buffer: " << cudaGetErrorString(err) << "\n";
        return false;
    }
    m_iasBufferSize = bufferSizes.outputSizeInBytes;

    // Build IAS
    res = optixAccelBuild(
        context,
        0,  // stream
        &buildOptions,
        &buildInput,
        1,
        m_tempBuffer,
        bufferSizes.tempSizeInBytes,
        m_iasBuffer,
        bufferSizes.outputSizeInBytes,
        &m_iasHandle,
        nullptr,
        0
    );

    if (res != OPTIX_SUCCESS) {
        std::cerr << "[SceneManager] Failed to build IAS: " << optixGetErrorName(res) << "\n";
        return false;
    }

    // Build buffer pointer arrays for shader access
    if (m_d_vertexBuffers) {
        cudaFree(m_d_vertexBuffers);
    }
    if (m_d_indexBuffers) {
        cudaFree(m_d_indexBuffers);
    }

    std::vector<CUdeviceptr> vertexPtrs(m_instances.size());
    std::vector<CUdeviceptr> indexPtrs(m_instances.size());

    for (size_t i = 0; i < m_instances.size(); ++i) {
        const GasInfo& gas = m_gasList[m_instances[i].gasIndex];
        vertexPtrs[i] = m_geometryManager->getVertexBuffer(gas.geometryHandle);
        indexPtrs[i] = m_geometryManager->getIndexBuffer(gas.geometryHandle);
    }

    m_bufferArraySize = m_instances.size();

    err = cudaMalloc(reinterpret_cast<void**>(&m_d_vertexBuffers), sizeof(CUdeviceptr) * m_bufferArraySize);
    if (err != cudaSuccess) {
        std::cerr << "[SceneManager] Failed to allocate vertex buffer array: " << cudaGetErrorString(err) << "\n";
        return false;
    }
    cudaMemcpy(m_d_vertexBuffers, vertexPtrs.data(), sizeof(CUdeviceptr) * m_bufferArraySize, cudaMemcpyHostToDevice);

    err = cudaMalloc(reinterpret_cast<void**>(&m_d_indexBuffers), sizeof(CUdeviceptr) * m_bufferArraySize);
    if (err != cudaSuccess) {
        std::cerr << "[SceneManager] Failed to allocate index buffer array: " << cudaGetErrorString(err) << "\n";
        return false;
    }
    cudaMemcpy(m_d_indexBuffers, indexPtrs.data(), sizeof(CUdeviceptr) * m_bufferArraySize, cudaMemcpyHostToDevice);

    std::cout << "[SceneManager] Built IAS with " << m_instances.size()
              << " instances (" << m_iasBufferSize / 1024 << " KB)\n";

    return true;
}

bool SceneManager::updateSBT() {
    if (!m_optixEngine || !m_materialManager) {
        return false;
    }

    if (m_gasList.empty()) {
        return true;  // Nothing to update
    }

    // Build materials and geometry indices for SBT
    std::vector<GpuMaterial> materials;
    std::vector<uint32_t> geometryIndices;

    for (size_t i = 0; i < m_gasList.size(); ++i) {
        const GasInfo& gas = m_gasList[i];
        const GpuMaterial* mat = m_materialManager->get(gas.materialHandle);

        if (mat) {
            materials.push_back(*mat);
        } else {
            // Default material
            GpuMaterial defaultMat = {};
            defaultMat.baseColor = make_float4(0.8f, 0.8f, 0.8f, 1.0f);
            defaultMat.metallic = 0.0f;
            defaultMat.roughness = 0.5f;
            materials.push_back(defaultMat);
        }

        geometryIndices.push_back(static_cast<uint32_t>(i));
    }

    return m_optixEngine->buildSBT(materials, geometryIndices);
}

void SceneManager::clear() {
    // Free IAS
    if (m_iasBuffer) {
        cudaFree(reinterpret_cast<void*>(m_iasBuffer));
        m_iasBuffer = 0;
    }
    if (m_instanceBuffer) {
        cudaFree(reinterpret_cast<void*>(m_instanceBuffer));
        m_instanceBuffer = 0;
    }
    m_iasHandle = 0;

    // Free buffer arrays
    if (m_d_vertexBuffers) {
        cudaFree(m_d_vertexBuffers);
        m_d_vertexBuffers = nullptr;
    }
    if (m_d_indexBuffers) {
        cudaFree(m_d_indexBuffers);
        m_d_indexBuffers = nullptr;
    }

    // Free GAS and release geometry
    for (auto& gas : m_gasList) {
        freeGAS(gas);
        if (m_geometryManager) {
            m_geometryManager->release(gas.geometryHandle);
        }
    }
    m_gasList.clear();

    // Clear instances
    m_instances.clear();

    // Free temp buffer
    if (m_tempBuffer) {
        cudaFree(reinterpret_cast<void*>(m_tempBuffer));
        m_tempBuffer = 0;
        m_tempBufferSize = 0;
    }

    std::cout << "[SceneManager] Cleared scene\n";
}

size_t SceneManager::getGpuMemoryUsage() const {
    size_t total = 0;

    // GAS buffers
    for (const auto& gas : m_gasList) {
        total += gas.bufferSize;
    }

    // IAS buffer
    total += m_iasBufferSize;

    // Instance buffer
    total += m_instances.size() * sizeof(OptixInstance);

    // Buffer arrays
    total += m_bufferArraySize * sizeof(CUdeviceptr) * 2;

    // Temp buffer
    total += m_tempBufferSize;

    return total;
}

} // namespace spectra
