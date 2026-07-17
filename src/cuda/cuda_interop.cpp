#include "cuda_interop.h"
#include <iostream>
#include <cuda.h>

namespace spectra {

CudaInterop::~CudaInterop() {
    shutdown();
}

bool CudaInterop::init() {
    // Query CUDA devices that are compatible with the current OpenGL context
    unsigned int deviceCount = 0;
    int devices[8];

    cudaError_t err = cudaGLGetDevices(&deviceCount, devices, 8, cudaGLDeviceListAll);
    if (err != cudaSuccess || deviceCount == 0) {
        std::cerr << "[CUDA] No CUDA devices compatible with OpenGL found\n";
        std::cerr << "[CUDA] Error: " << cudaGetErrorString(err) << "\n";
        return false;
    }

    std::cout << "[CUDA] Found " << deviceCount << " OpenGL-compatible device(s)\n";

    // Select the first compatible device (should be same as OpenGL)
    m_deviceId = devices[0];
    CUDA_CHECK(cudaSetDevice(m_deviceId));

    // Verify device properties
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, m_deviceId));

    std::cout << "[CUDA] Selected device: " << prop.name << "\n";
    std::cout << "[CUDA] Compute capability: " << prop.major << "." << prop.minor << "\n";

    // Warn if compute capability is too low for RT cores
    if (prop.major < 7 || (prop.major == 7 && prop.minor < 5)) {
        std::cerr << "[CUDA] Warning: Compute capability " << prop.major << "." << prop.minor
                  << " may not have RT cores. RTX (7.5+) recommended.\n";
    }

    // Create CUDA context (use primary context associated with device)
    CUDA_CHECK(cudaFree(0));  // Force context creation

    CUresult cuErr = cuCtxGetCurrent(&m_cudaContext);
    if (cuErr != CUDA_SUCCESS) {
        std::cerr << "[CUDA] Failed to get current context\n";
        return false;
    }

    // Create stream for async scene rendering operations
    CUDA_CHECK(cudaStreamCreate(&m_stream));

    // Create separate stream for UI rendering (avoids blocking scene pipeline)
    CUDA_CHECK(cudaStreamCreate(&m_uiStream));

    printDeviceInfo();
    printMemoryUsage();

    return true;
}

void CudaInterop::shutdown() {
    // Unmap and unregister all triple-buffered PBOs
    unregisterPBOs();
    if (m_uiPboMapped) {
        unmapUIPBO();
    }
    if (m_uiPboResource) {
        unregisterUIPBO();
    }
    // Destroy events
    for (int i = 0; i < NUM_SCENE_BUFFERS; ++i) {
        if (m_renderComplete[i]) {
            cudaEventDestroy(m_renderComplete[i]);
            m_renderComplete[i] = nullptr;
        }
    }
    if (m_stream) {
        cudaStreamDestroy(m_stream);
        m_stream = nullptr;
    }
    if (m_uiStream) {
        cudaStreamDestroy(m_uiStream);
        m_uiStream = nullptr;
    }
    // Don't destroy the CUDA context - it's managed by the runtime
    m_cudaContext = nullptr;
    m_deviceId = -1;
}

bool CudaInterop::registerPBOs(uint32_t pbo0, uint32_t pbo1, uint32_t pbo2, size_t size) {
    // Unregister any existing PBOs
    unregisterPBOs();

    uint32_t pbos[NUM_SCENE_BUFFERS] = {pbo0, pbo1, pbo2};

    for (int i = 0; i < NUM_SCENE_BUFFERS; ++i) {
        CUDA_CHECK(cudaGraphicsGLRegisterBuffer(
            &m_pboResources[i],
            pbos[i],
            cudaGraphicsMapFlagsWriteDiscard
        ));

        // Use cudaEventDisableTiming for lower overhead synchronization
        // We don't need timing on these events, just completion notification
        CUDA_CHECK(cudaEventCreateWithFlags(&m_renderComplete[i], cudaEventDisableTiming));
        m_pboMapped[i] = false;
    }

    m_pboSize = size;

    std::cout << "[CUDA] Registered triple-buffered PBOs " << pbo0 << "/" << pbo1 << "/" << pbo2
              << " (" << size / (1024 * 1024) << " MB each)\n";

    return true;
}

void CudaInterop::unregisterPBOs() {
    for (int i = 0; i < NUM_SCENE_BUFFERS; ++i) {
        if (m_pboMapped[i]) {
            unmapBuffer(i);
        }
        if (m_pboResources[i]) {
            CUDA_CHECK_NORETURN(cudaGraphicsUnregisterResource(m_pboResources[i]));
            m_pboResources[i] = nullptr;
        }
        if (m_renderComplete[i]) {
            cudaEventDestroy(m_renderComplete[i]);
            m_renderComplete[i] = nullptr;
        }
        m_pboMapped[i] = false;
    }
    m_pboSize = 0;

    std::cout << "[CUDA] Unregistered all PBOs\n";
}

bool CudaInterop::registerUIPBO(uint32_t pbo, size_t size) {
    if (m_uiPboResource) {
        unregisterUIPBO();
    }

    CUDA_CHECK(cudaGraphicsGLRegisterBuffer(
        &m_uiPboResource,
        pbo,
        cudaGraphicsMapFlagsWriteDiscard
    ));

    m_uiPboSize = size;
    std::cout << "[CUDA] Registered UI PBO " << pbo << " (" << size / (1024 * 1024) << " MB)\n";

    return true;
}

void CudaInterop::unregisterUIPBO() {
    if (m_uiPboMapped) {
        unmapUIPBO();
    }
    if (m_uiPboResource) {
        CUDA_CHECK_NORETURN(cudaGraphicsUnregisterResource(m_uiPboResource));
        m_uiPboResource = nullptr;
        m_uiPboSize = 0;
        std::cout << "[CUDA] Unregistered UI PBO\n";
    }
}

float* CudaInterop::mapBuffer(int index) {
    if (index < 0 || index >= NUM_SCENE_BUFFERS) {
        std::cerr << "[CUDA] Invalid buffer index: " << index << "\n";
        return nullptr;
    }

    if (!m_pboResources[index]) {
        std::cerr << "[CUDA] Cannot map: PBO " << index << " not registered\n";
        return nullptr;
    }

    if (m_pboMapped[index]) {
        std::cerr << "[CUDA] Warning: PBO " << index << " already mapped\n";
        return nullptr;
    }

    cudaError_t err = cudaGraphicsMapResources(1, &m_pboResources[index], m_stream);
    if (err != cudaSuccess) {
        std::cerr << "[CUDA] Failed to map PBO " << index << ": " << cudaGetErrorString(err) << "\n";
        return nullptr;
    }

    void* devPtr = nullptr;
    size_t mappedSize = 0;
    err = cudaGraphicsResourceGetMappedPointer(&devPtr, &mappedSize, m_pboResources[index]);
    if (err != cudaSuccess) {
        std::cerr << "[CUDA] Failed to get mapped pointer for PBO " << index << ": " << cudaGetErrorString(err) << "\n";
        cudaGraphicsUnmapResources(1, &m_pboResources[index], m_stream);
        return nullptr;
    }

    m_pboMapped[index] = true;
    return static_cast<float*>(devPtr);
}

void CudaInterop::unmapBuffer(int index) {
    if (index < 0 || index >= NUM_SCENE_BUFFERS) return;

    if (!m_pboMapped[index]) {
        return;
    }

    CUDA_CHECK_NORETURN(cudaGraphicsUnmapResources(1, &m_pboResources[index], m_stream));
    m_pboMapped[index] = false;
}

void CudaInterop::recordRenderComplete(int index) {
    if (index < 0 || index >= NUM_SCENE_BUFFERS) return;
    if (m_renderComplete[index]) {
        cudaEventRecord(m_renderComplete[index], m_stream);
    }
}

bool CudaInterop::isRenderComplete(int index) {
    if (index < 0 || index >= NUM_SCENE_BUFFERS) return true;
    if (!m_renderComplete[index]) return true;
    return cudaEventQuery(m_renderComplete[index]) == cudaSuccess;
}

void CudaInterop::waitForRender(int index) {
    if (index < 0 || index >= NUM_SCENE_BUFFERS) return;
    if (m_renderComplete[index]) {
        cudaEventSynchronize(m_renderComplete[index]);
    }
}

float* CudaInterop::mapUIPBO() {
    if (!m_uiPboResource) {
        std::cerr << "[CUDA] Cannot map: UI PBO not registered\n";
        return nullptr;
    }

    if (m_uiPboMapped) {
        std::cerr << "[CUDA] Warning: UI PBO already mapped\n";
        return nullptr;
    }

    // Use UI stream — avoids blocking the render stream with GL-CUDA sync
    cudaError_t err = cudaGraphicsMapResources(1, &m_uiPboResource, m_uiStream);
    if (err != cudaSuccess) {
        std::cerr << "[CUDA] Failed to map UI PBO: " << cudaGetErrorString(err) << "\n";
        return nullptr;
    }

    void* devPtr = nullptr;
    size_t mappedSize = 0;
    err = cudaGraphicsResourceGetMappedPointer(&devPtr, &mappedSize, m_uiPboResource);
    if (err != cudaSuccess) {
        std::cerr << "[CUDA] Failed to get UI mapped pointer: " << cudaGetErrorString(err) << "\n";
        cudaGraphicsUnmapResources(1, &m_uiPboResource, m_uiStream);
        return nullptr;
    }

    m_uiPboMapped = true;
    return static_cast<float*>(devPtr);
}

void CudaInterop::unmapUIPBO() {
    if (!m_uiPboMapped) {
        return;
    }

    // Use UI stream to match mapUIPBO
    CUDA_CHECK_NORETURN(cudaGraphicsUnmapResources(1, &m_uiPboResource, m_uiStream));
    m_uiPboMapped = false;
}

void CudaInterop::synchronize() {
    if (m_stream) {
        CUDA_CHECK_NORETURN(cudaStreamSynchronize(m_stream));
    }
}

void CudaInterop::synchronizeUI() {
    if (m_uiStream) {
        CUDA_CHECK_NORETURN(cudaStreamSynchronize(m_uiStream));
    }
}

void CudaInterop::printDeviceInfo() const {
    if (m_deviceId < 0) {
        return;
    }

    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, m_deviceId) != cudaSuccess) {
        return;
    }

    std::cout << "[CUDA] Device Info:\n";
    std::cout << "  Name: " << prop.name << "\n";
    std::cout << "  Compute: " << prop.major << "." << prop.minor << "\n";
    std::cout << "  SM Count: " << prop.multiProcessorCount << "\n";
    // clockRate and memoryClockRate removed in CUDA 12+
    std::cout << "  Memory Bus: " << prop.memoryBusWidth << " bit\n";
    std::cout << "  Total Memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB\n";
}

void CudaInterop::printMemoryUsage() const {
    size_t free, total;
    if (cudaMemGetInfo(&free, &total) == cudaSuccess) {
        std::cout << "[CUDA] Memory: " << (total - free) / (1024 * 1024) << " / "
                  << total / (1024 * 1024) << " MB used\n";
    }
}

} // namespace spectra
