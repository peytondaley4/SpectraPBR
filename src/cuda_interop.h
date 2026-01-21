#pragma once

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <cstdint>

namespace spectra {

// CUDA error checking macro
#define CUDA_CHECK(call)                                                         \
    do {                                                                         \
        cudaError_t error = call;                                                \
        if (error != cudaSuccess) {                                              \
            std::cerr << "[CUDA] Error: " << cudaGetErrorString(error)           \
                      << " at " << __FILE__ << ":" << __LINE__ << "\n";          \
            return false;                                                        \
        }                                                                        \
    } while (0)

// Variant that doesn't return (for use in destructors)
#define CUDA_CHECK_NORETURN(call)                                                \
    do {                                                                         \
        cudaError_t error = call;                                                \
        if (error != cudaSuccess) {                                              \
            std::cerr << "[CUDA] Error: " << cudaGetErrorString(error)           \
                      << " at " << __FILE__ << ":" << __LINE__ << "\n";          \
        }                                                                        \
    } while (0)

class CudaInterop {
public:
    CudaInterop() = default;
    ~CudaInterop();

    // Non-copyable
    CudaInterop(const CudaInterop&) = delete;
    CudaInterop& operator=(const CudaInterop&) = delete;

    // Initialize CUDA and select device compatible with OpenGL
    // Must be called after OpenGL context is created
    bool init();

    // Shutdown CUDA
    void shutdown();

    // Register OpenGL PBO with CUDA
    // Returns false on failure
    bool registerPBO(uint32_t pbo, size_t size);

    // Unregister PBO (call before resizing)
    void unregisterPBO();

    // Map PBO for CUDA access
    // Returns device pointer, or nullptr on failure
    float* mapPBO();

    // Unmap PBO (must call after rendering, before OpenGL uses it)
    void unmapPBO();

    // Get CUDA stream for async operations
    cudaStream_t getStream() const { return m_stream; }

    // Get CUDA context (CUcontext)
    CUcontext getCudaContext() const { return m_cudaContext; }

    // Get selected device ID
    int getDeviceId() const { return m_deviceId; }

    // Synchronize stream
    void synchronize();

    // Print device info
    void printDeviceInfo() const;

    // Print memory usage
    void printMemoryUsage() const;

private:
    int m_deviceId = -1;
    CUcontext m_cudaContext = nullptr;
    cudaStream_t m_stream = nullptr;

    // PBO interop
    cudaGraphicsResource_t m_pboResource = nullptr;
    size_t m_pboSize = 0;
    bool m_pboMapped = false;
};

} // namespace spectra
