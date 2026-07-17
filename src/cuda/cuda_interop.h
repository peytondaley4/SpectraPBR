#pragma once

#include <glad/glad.h>  // Must be included before cuda_gl_interop.h
#include <cuda.h>       // For CUcontext
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
    static constexpr int NUM_SCENE_BUFFERS = 3;  // Triple buffering for scene

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

    // Register triple-buffered PBOs with CUDA
    // Returns false on failure
    bool registerPBOs(uint32_t pbo0, uint32_t pbo1, uint32_t pbo2, size_t size);

    // Register UI PBO with CUDA
    bool registerUIPBO(uint32_t pbo, size_t size);

    // Unregister all triple-buffered PBOs (call before resizing)
    void unregisterPBOs();

    // Unregister UI PBO
    void unregisterUIPBO();

    // Map specific buffer for CUDA access (triple buffering)
    // Returns device pointer, or nullptr on failure
    float* mapBuffer(int index);

    // Map UI PBO for CUDA access (uses UI stream, not render stream)
    float* mapUIPBO();

    // Unmap specific buffer (must call after rendering, before OpenGL uses it)
    void unmapBuffer(int index);

    // Unmap UI PBO (uses UI stream, not render stream)
    void unmapUIPBO();

    // Record an event when render completes (triple buffering)
    void recordRenderComplete(int index);

    // Check if render is complete (non-blocking)
    bool isRenderComplete(int index);

    // Wait for render to complete (blocking)
    void waitForRender(int index);

    // Get CUDA stream for async operations (scene rendering)
    cudaStream_t getStream() const { return m_stream; }

    // Get separate CUDA stream for UI rendering (avoids blocking scene pipeline)
    cudaStream_t getUIStream() const { return m_uiStream; }

    // Synchronize UI stream only (doesn't block scene rendering)
    void synchronizeUI();

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
    cudaStream_t m_stream = nullptr;      // Main stream for scene rendering
    cudaStream_t m_uiStream = nullptr;    // Separate stream for UI (avoids blocking scene)

    // Triple-buffered PBO interop
    cudaGraphicsResource_t m_pboResources[NUM_SCENE_BUFFERS] = {};
    cudaEvent_t m_renderComplete[NUM_SCENE_BUFFERS] = {};
    bool m_pboMapped[NUM_SCENE_BUFFERS] = {};
    size_t m_pboSize = 0;

    // UI PBO interop (single-buffered)
    cudaGraphicsResource_t m_uiPboResource = nullptr;
    size_t m_uiPboSize = 0;
    bool m_uiPboMapped = false;
};

} // namespace spectra
