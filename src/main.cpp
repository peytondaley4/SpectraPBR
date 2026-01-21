#include "gl_context.h"
#include "cuda_interop.h"
#include "optix_engine.h"
#include <iostream>
#include <chrono>
#include <filesystem>

using namespace spectra;

// Frame timing
struct FrameTimer {
    using Clock = std::chrono::high_resolution_clock;
    using TimePoint = Clock::time_point;

    TimePoint frameStart;
    double frameTimeMs = 0.0;
    double fps = 0.0;
    uint64_t frameCount = 0;

    // Rolling average
    static constexpr int SAMPLE_COUNT = 60;
    double samples[SAMPLE_COUNT] = {};
    int sampleIndex = 0;

    void beginFrame() {
        frameStart = Clock::now();
    }

    void endFrame() {
        auto now = Clock::now();
        frameTimeMs = std::chrono::duration<double, std::milli>(now - frameStart).count();

        samples[sampleIndex] = frameTimeMs;
        sampleIndex = (sampleIndex + 1) % SAMPLE_COUNT;

        double sum = 0.0;
        for (int i = 0; i < SAMPLE_COUNT; i++) {
            sum += samples[i];
        }
        double avgMs = sum / SAMPLE_COUNT;
        fps = 1000.0 / avgMs;

        frameCount++;
    }

    void print() const {
        std::cout << "[Timing] Frame: " << frameTimeMs << " ms, FPS: " << fps
                  << " (avg over " << SAMPLE_COUNT << " frames)\n";
    }
};

// Global state for keyboard callbacks
static FrameTimer g_timer;
static CudaInterop* g_cudaInterop = nullptr;
static GLFWkeyfun g_previousKeyCallback = nullptr;

void printTimingCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    (void)window; (void)scancode; (void)mods;

    if (action != GLFW_PRESS) return;

    switch (key) {
        case GLFW_KEY_T:
            g_timer.print();
            break;
        case GLFW_KEY_G:
            if (g_cudaInterop) {
                g_cudaInterop->printDeviceInfo();
                g_cudaInterop->printMemoryUsage();
            }
            break;
        default:
            break;
    }
}

int main(int argc, char* argv[]) {
    (void)argc; (void)argv;

    std::cout << "=== SpectraPBR - Phase 1 ===\n\n";

    // Get executable directory for finding shaders and PTX files
    std::filesystem::path exePath = std::filesystem::absolute(argv[0]).parent_path();
    std::filesystem::path shaderDir = exePath / "shaders";
    std::filesystem::path ptxDir = exePath / "optix_programs";

    // Fall back to source directories if running from build directory
    if (!std::filesystem::exists(shaderDir)) {
        shaderDir = std::filesystem::current_path() / "shaders";
    }
    if (!std::filesystem::exists(ptxDir)) {
        ptxDir = std::filesystem::current_path() / "optix_programs";
    }

    std::cout << "[Main] Shader directory: " << shaderDir << "\n";
    std::cout << "[Main] PTX directory: " << ptxDir << "\n\n";

    // Initialize OpenGL context
    GLContext glContext;
    if (!glContext.init(RESOLUTION_1080P.width, RESOLUTION_1080P.height, "SpectraPBR")) {
        std::cerr << "[Main] Failed to initialize OpenGL context\n";
        return 1;
    }

    // Initialize CUDA (must be after OpenGL)
    CudaInterop cudaInterop;
    g_cudaInterop = &cudaInterop;

    if (!cudaInterop.init()) {
        std::cerr << "[Main] Failed to initialize CUDA\n";
        return 1;
    }

    // Initialize OptiX
    OptixEngine optixEngine;
    if (!optixEngine.init(cudaInterop.getCudaContext())) {
        std::cerr << "[Main] Failed to initialize OptiX\n";
        return 1;
    }

    // Create display resources (texture, PBO, shaders)
    if (!glContext.createDisplayResources(shaderDir)) {
        std::cerr << "[Main] Failed to create display resources\n";
        return 1;
    }

    // Register PBO with CUDA
    if (!cudaInterop.registerPBO(glContext.getPBO(), glContext.getBufferSize())) {
        std::cerr << "[Main] Failed to register PBO with CUDA\n";
        return 1;
    }

    // Load OptiX pipeline
    if (!optixEngine.createPipeline(ptxDir)) {
        std::cerr << "[Main] Failed to create OptiX pipeline\n";
        return 1;
    }

    // Set initial dimensions
    optixEngine.setDimensions(glContext.getWidth(), glContext.getHeight());

    // Set up resize callback
    glContext.setResizeCallback([&](uint32_t width, uint32_t height) {
        std::cout << "[Main] Resize: " << width << "x" << height << "\n";

        // Unregister old PBO
        cudaInterop.unregisterPBO();

        // Re-register new PBO
        if (!cudaInterop.registerPBO(glContext.getPBO(), glContext.getBufferSize())) {
            std::cerr << "[Main] Failed to re-register PBO after resize\n";
        }

        // Update OptiX dimensions
        optixEngine.setDimensions(width, height);
    });

    // Add debug key handler (T for timing, G for GPU info)
    // Chain with existing GLContext key callback
    g_previousKeyCallback = glfwSetKeyCallback(glContext.getWindow(), [](GLFWwindow* window, int key, int scancode, int action, int mods) {
        // Call the original GLContext key callback first (handles ESC, V, F, 1-4)
        if (g_previousKeyCallback) {
            g_previousKeyCallback(window, key, scancode, action, mods);
        }

        // Handle our additional debug keys
        printTimingCallback(window, key, scancode, action, mods);
    });

    std::cout << "\n[Main] Initialization complete!\n";
    std::cout << "[Main] Controls:\n";
    std::cout << "  ESC - Quit\n";
    std::cout << "  V   - Toggle VSync\n";
    std::cout << "  F   - Toggle Fullscreen\n";
    std::cout << "  1-4 - Resolution (720p/1080p/1440p/4K)\n";
    std::cout << "  T   - Print timing info\n";
    std::cout << "  G   - Print GPU info\n";
    std::cout << "\n";

    // Main render loop
    while (!glContext.shouldClose()) {
        g_timer.beginFrame();

        // Poll events
        glContext.pollEvents();

        // Map PBO for CUDA access
        float4* devicePtr = reinterpret_cast<float4*>(cudaInterop.mapPBO());
        if (!devicePtr) {
            std::cerr << "[Main] Failed to map PBO\n";
            break;
        }

        // Render with OptiX
        optixEngine.render(devicePtr, cudaInterop.getStream());

        // Synchronize before unmapping
        cudaInterop.synchronize();

        // Unmap PBO
        cudaInterop.unmapPBO();

        // Update OpenGL texture from PBO
        glContext.updateTextureFromPBO();

        // Render fullscreen quad
        glContext.renderFullscreenQuad();

        // Swap buffers
        glContext.swapBuffers();

        g_timer.endFrame();
    }

    std::cout << "\n[Main] Shutting down...\n";
    std::cout << "[Main] Total frames rendered: " << g_timer.frameCount << "\n";

    // Cleanup (RAII handles most of it, but explicit order matters)
    g_cudaInterop = nullptr;

    std::cout << "[Main] Goodbye!\n";

    return 0;
}
