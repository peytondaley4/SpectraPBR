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

    // Create stream for async operations
    CUDA_CHECK(cudaStreamCreate(&m_stream));

    printDeviceInfo();
    printMemoryUsage();

    return true;
}

void CudaInterop::shutdown() {
    if (m_pboMapped) {
        unmapPBO();
    }
    if (m_pboResource) {
        unregisterPBO();
    }
    if (m_uiPboMapped) {
        unmapUIPBO();
    }
    if (m_uiPboResource) {
        unregisterUIPBO();
    }
    if (m_stream) {
        cudaStreamDestroy(m_stream);
        m_stream = nullptr;
    }
    // Don't destroy the CUDA context - it's managed by the runtime
    m_cudaContext = nullptr;
    m_deviceId = -1;
}

bool CudaInterop::registerPBO(uint32_t pbo, size_t size) {
    if (m_pboResource) {
        unregisterPBO();
    }

    CUDA_CHECK(cudaGraphicsGLRegisterBuffer(
        &m_pboResource,
        pbo,
        cudaGraphicsMapFlagsWriteDiscard  // We only write, never read
    ));

    m_pboSize = size;
    std::cout << "[CUDA] Registered PBO " << pbo << " (" << size / (1024 * 1024) << " MB)\n";

    return true;
}

void CudaInterop::unregisterPBO() {
    if (m_pboMapped) {
        unmapPBO();
    }
    if (m_pboResource) {
        CUDA_CHECK_NORETURN(cudaGraphicsUnregisterResource(m_pboResource));
        m_pboResource = nullptr;
        m_pboSize = 0;
        std::cout << "[CUDA] Unregistered PBO\n";
    }
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

float* CudaInterop::mapPBO() {
    if (!m_pboResource) {
        std::cerr << "[CUDA] Cannot map: PBO not registered\n";
        return nullptr;
    }

    if (m_pboMapped) {
        std::cerr << "[CUDA] Warning: PBO already mapped\n";
        return nullptr;
    }

    cudaError_t err = cudaGraphicsMapResources(1, &m_pboResource, m_stream);
    if (err != cudaSuccess) {
        std::cerr << "[CUDA] Failed to map PBO: " << cudaGetErrorString(err) << "\n";
        return nullptr;
    }

    void* devPtr = nullptr;
    size_t mappedSize = 0;
    err = cudaGraphicsResourceGetMappedPointer(&devPtr, &mappedSize, m_pboResource);
    if (err != cudaSuccess) {
        std::cerr << "[CUDA] Failed to get mapped pointer: " << cudaGetErrorString(err) << "\n";
        cudaGraphicsUnmapResources(1, &m_pboResource, m_stream);
        return nullptr;
    }

    m_pboMapped = true;
    return static_cast<float*>(devPtr);
}

void CudaInterop::unmapPBO() {
    if (!m_pboMapped) {
        return;
    }

    CUDA_CHECK_NORETURN(cudaGraphicsUnmapResources(1, &m_pboResource, m_stream));
    m_pboMapped = false;
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

    cudaError_t err = cudaGraphicsMapResources(1, &m_uiPboResource, m_stream);
    if (err != cudaSuccess) {
        std::cerr << "[CUDA] Failed to map UI PBO: " << cudaGetErrorString(err) << "\n";
        return nullptr;
    }

    void* devPtr = nullptr;
    size_t mappedSize = 0;
    err = cudaGraphicsResourceGetMappedPointer(&devPtr, &mappedSize, m_uiPboResource);
    if (err != cudaSuccess) {
        std::cerr << "[CUDA] Failed to get UI mapped pointer: " << cudaGetErrorString(err) << "\n";
        cudaGraphicsUnmapResources(1, &m_uiPboResource, m_stream);
        return nullptr;
    }

    m_uiPboMapped = true;
    return static_cast<float*>(devPtr);
}

void CudaInterop::unmapUIPBO() {
    if (!m_uiPboMapped) {
        return;
    }

    CUDA_CHECK_NORETURN(cudaGraphicsUnmapResources(1, &m_uiPboResource, m_stream));
    m_uiPboMapped = false;
}

void CudaInterop::synchronize() {
    if (m_stream) {
        CUDA_CHECK_NORETURN(cudaStreamSynchronize(m_stream));
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
