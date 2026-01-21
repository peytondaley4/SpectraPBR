# Real-Time OptiX Ray Tracing Engine - Phase 1 Specification

## Project Overview

A high-performance real-time ray tracing engine using NVIDIA OptiX 7+ with OpenGL/CUDA interoperability. This document covers **Phase 1: Core Infrastructure & Interop Setup** - establishing the foundational rendering pipeline.

**Target Hardware:** NVIDIA RTX GPU (Turing, Ampere, Ada Lovelace or newer) with RT cores  
**Target Performance:** 60+ FPS at 1080p for Phase 1 baseline  
**Platform:** Windows 10/11 primary, Linux secondary

---

## Phase 1: Core Infrastructure & Interop Setup

### Objective

Establish the foundational OpenGL-CUDA-OptiX pipeline and verify all components communicate correctly by displaying a solid color or simple gradient from OptiX to screen.

**Success Criteria:**
- Application window opens without errors
- OptiX initializes successfully on RTX GPU
- Display shows output from OptiX raygen program (solid color or gradient)
- Runs at 60+ FPS with VSync enabled
- Clean shutdown with no memory leaks
- Total frame overhead < 1ms (excluding VSync wait)

### System Architecture

**Complete Rendering Pipeline Flow:**

```
1. OpenGL Context Creation (GLFW/SDL2)
   ↓
2. CUDA Context Creation (same GPU as OpenGL)
   ↓
3. OptiX Context Initialization (on CUDA context)
   ↓
4. Create Display Buffer Chain:
   - OpenGL Texture (RGBA8 or RGBA32F)
   - OpenGL PBO (Pixel Buffer Object)
   - CUDA Device Buffer
   - Register PBO with CUDA graphics resource
   ↓
5. Per-Frame Render Loop:
   - Map PBO to CUDA
   - OptiX renders to CUDA buffer
   - Copy result to PBO
   - Unmap PBO
   - Update OpenGL texture from PBO
   - Render fullscreen quad with texture
   - Swap buffers
```

### Project Structure

```
project/
├── CMakeLists.txt
├── claude.md (this file)
├── src/
│   ├── main.cpp                # Application entry & main loop
│   ├── gl_context.h/cpp        # OpenGL initialization
│   ├── cuda_interop.h/cpp      # CUDA-GL buffer interop
│   ├── optix_engine.h/cpp      # OptiX setup & rendering
│   └── shader_utils.h/cpp      # Shader compilation
├── shaders/
│   ├── display.vert            # Fullscreen quad vertex shader
│   └── display.frag            # Texture display fragment shader
└── optix_programs/
    ├── raygen.cu               # Ray generation (outputs gradient)
    └── miss.cu                 # Miss program (background)
```

### Dependencies & Build Requirements

#### Required Software
- **CUDA Toolkit:** 11.0+ (12.x recommended)
- **OptiX SDK:** 7.7+ (download from NVIDIA Developer)
- **CMake:** 3.18+
- **C++ Compiler:** C++17 support (MSVC 2019+, GCC 9+, Clang 10+)

#### Required Libraries
- **OpenGL:** 4.5+
- **GLFW:** 3.3+ (or SDL2)
- **GLAD/GLEW:** OpenGL function loader
- **GLM:** 0.9.9+ (mathematics)

#### CMake Configuration Notes
- Enable `CMAKE_CUDA_SEPARABLE_COMPILATION ON`
- Set `CMAKE_CUDA_ARCHITECTURES` to target GPUs (75=Turing, 86=Ampere, 89=Ada)
- Find OptiX (may need custom FindOptiX.cmake module)
- Compile OptiX programs to PTX at build time
- Link: OpenGL, GLFW, CUDA runtime, OptiX

---

## Implementation Checklist

### 1. OpenGL Context Setup
**Reference:** https://www.glfw.org/documentation.html

- [ ] Initialize GLFW
- [ ] Request OpenGL 4.5+ core profile
- [ ] Create window (1920x1080 default)
- [ ] Load OpenGL functions with GLAD/GLEW
- [ ] Set up viewport and clear color
- [ ] Enable VSync (`glfwSwapInterval(1)`)
- [ ] Create fullscreen quad shader programs
  - Vertex shader: Use fullscreen triangle trick (no VBO needed)
  - Fragment shader: Sample from texture and output
- [ ] Compile and link display shaders

**Key Implementation Notes:**
- Use fullscreen triangle technique (3 vertices, no vertex buffer)
- Display shader samples from OptiX output texture
- Print OpenGL version on initialization

### 2. CUDA Initialization
**Reference:** https://docs.nvidia.com/cuda/cuda-runtime-api/

- [ ] Query available CUDA devices
- [ ] Select device with highest compute capability (7.5+ for RTX)
- [ ] Verify RT core support (compute capability check)
- [ ] Create CUDA context compatible with OpenGL
- [ ] Verify CUDA-GL interop support (`cudaGLGetDevices`)
- [ ] Create CUDA stream for async operations
- [ ] Implement error checking macro (`CUDA_CHECK`)

**Key Implementation Notes:**
- Print all available CUDA devices with compute capability
- Select best device automatically (highest compute)
- Warn if compute capability < 7.5

### 3. CUDA-OpenGL Interop
**Reference:** https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__OPENGL.html

- [ ] Create OpenGL texture (2D, RGBA32F for HDR path)
- [ ] Create OpenGL PBO for efficient transfers
  - Size: `width * height * 4 * sizeof(float)`
  - Usage: `GL_DYNAMIC_DRAW`
- [ ] Register PBO with CUDA: `cudaGraphicsGLRegisterBuffer`
  - Flags: `cudaGraphicsMapFlagsWriteDiscard`
- [ ] Allocate CUDA device buffer (same size as PBO)
- [ ] Implement map/unmap functions:
  - `cudaGraphicsMapResources` before OptiX render
  - `cudaGraphicsResourceGetMappedPointer` to get device pointer
  - `cudaGraphicsUnmapResources` after writing
- [ ] Update OpenGL texture from PBO using `glTexSubImage2D`

**Key Implementation Notes:**
- PBO is faster than direct texture writes
- Map before render, unmap after, then update texture
- Use write-discard flag for best performance

### 4. OptiX Context Initialization
**Reference:** https://raytracing-docs.nvidia.com/optix7/guide/index.html

- [ ] Initialize OptiX API: `optixInit()`
- [ ] Create OptiX device context on existing CUDA context
- [ ] Set up logging callback for OptiX messages
  - Log level 4 for Phase 1 (all messages)
  - Print to stderr with format: `[level][tag]: message`
- [ ] Configure device context options (validation mode)
- [ ] Set conservative stack sizes initially:
  - Direct callable: 2KB
  - Continuation: 2KB
  - From traversal: 2KB
  - Max depth: 2
- [ ] Implement error checking macro (`OPTIX_CHECK`)

**Key Implementation Notes:**
- Use CUcontext = 0 to use current CUDA context
- Enable validation in debug builds, disable in release
- Stack sizes can be tuned later based on actual usage

### 5. OptiX Module & Pipeline
**Reference:** https://raytracing-docs.nvidia.com/optix7/guide/index.html#program_pipeline_creation

- [ ] Load PTX from file (raygen.ptx, miss.ptx)
- [ ] Set module compile options:
  - Optimization: `OPTIX_COMPILE_OPTIMIZATION_DEFAULT`
  - Debug: `OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL` (release)
- [ ] Set pipeline compile options:
  - No motion blur
  - Allow single-level instancing
  - No payload/attributes (Phase 1)
  - Launch params variable name: "params"
- [ ] Create OptiX module from PTX: `optixModuleCreateFromPTX`
- [ ] Create program groups:
  - Raygen: `__raygen__simple`
  - Miss: `__miss__background`
- [ ] Link pipeline: `optixPipelineCreate`
- [ ] Compute and set stack sizes: `optixUtilComputeStackSizes`

**Key Implementation Notes:**
- Check log output after module creation
- Program group kinds: `OPTIX_PROGRAM_GROUP_KIND_RAYGEN`, `OPTIX_PROGRAM_GROUP_KIND_MISS`
- Pipeline link options: max trace depth = 1 for Phase 1

### 6. Shader Binding Table (SBT)
**Reference:** https://raytracing-docs.nvidia.com/optix7/guide/index.html#shader_binding_table

- [ ] Define SBT record structures:
  ```cpp
  struct RaygenRecord {
      __align__(OPTIX_SBT_RECORD_ALIGNMENT) 
      char header[OPTIX_SBT_RECORD_HEADER_SIZE];
      // No additional data for Phase 1
  };
  
  struct MissRecord {
      __align__(OPTIX_SBT_RECORD_ALIGNMENT)
      char header[OPTIX_SBT_RECORD_HEADER_SIZE];
      // No additional data for Phase 1
  };
  ```
- [ ] Pack program group headers: `optixSbtRecordPackHeader`
- [ ] Allocate device memory for records
- [ ] Copy records to device
- [ ] Set up `OptixShaderBindingTable` structure:
  - `raygenRecord`: Pointer to raygen record
  - `missRecordBase`: Pointer to miss records
  - `missRecordStrideInBytes`: sizeof(MissRecord)
  - `missRecordCount`: 1

**Key Implementation Notes:**
- Alignment is critical (use OPTIX_SBT_RECORD_ALIGNMENT)
- All SBT records must be on device memory
- Miss/hitgroup records can be arrays (just 1 each for Phase 1)

### 7. Launch Parameters
**Reference:** https://raytracing-docs.nvidia.com/optix7/guide/index.html#launch_parameters

- [ ] Define launch parameters structure:
  ```cpp
  struct LaunchParams {
      float4* output_buffer;
      unsigned int width;
      unsigned int height;
  };
  ```
- [ ] Allocate device memory for launch params
- [ ] Update launch params each frame (or when changed):
  - Set output_buffer pointer
  - Set width/height
- [ ] Copy to device before `optixLaunch`

**Key Implementation Notes:**
- Launch params passed to all OptiX programs
- Accessed via `optixGetLaunchIndex()` in programs
- Can be expanded in later phases

### 8. OptiX Programs (CUDA)

**Raygen Program (raygen.cu):**
- [ ] Include `<optix.h>`
- [ ] Declare launch params as `extern "C" __constant__` or access via device pointer
- [ ] Implement `__raygen__simple` function:
  - Get launch index: `optixGetLaunchIndex()`
  - Get launch dimensions: `optixGetLaunchDimensions()`
  - Calculate normalized coordinates (0-1)
  - Generate simple gradient: `color = (x/width, y/height, 0.5, 1.0)`
  - Write to output buffer: `output[y * width + x] = color`

**Miss Program (miss.cu):**
- [ ] Include `<optix.h>`
- [ ] Implement `__miss__background` function:
  - For Phase 1, this won't be called (no rays traced)
  - Can leave empty or set background color

**Key Implementation Notes:**
- Programs compiled to PTX at build time
- Entry point names must match those in program group creation
- No ray tracing yet - just write gradient to buffer

### 9. Main Render Loop

- [ ] Initialize all systems in order:
  1. OpenGL context
  2. CUDA interop
  3. OptiX engine
- [ ] Allocate CUDA output buffer matching display size
- [ ] Main loop structure:
  ```cpp
  while (!window_should_close) {
      // Handle input (ESC to quit)
      poll_events();
      
      // Map OpenGL PBO to CUDA
      cuda_interop.mapResources();
      float4* device_ptr = cuda_interop.getMappedPointer();
      
      // Render with OptiX
      optix_engine.render(device_ptr);
      
      // Unmap PBO for OpenGL
      cuda_interop.unmapResources();
      
      // Update texture from PBO
      glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
      glBindTexture(GL_TEXTURE_2D, texture);
      glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, 
                      GL_RGBA, GL_FLOAT, nullptr);
      
      // Render fullscreen quad
      glClear(GL_COLOR_BUFFER_BIT);
      glUseProgram(display_shader);
      glBindTexture(GL_TEXTURE_2D, texture);
      glDrawArrays(GL_TRIANGLES, 0, 3);
      
      // Swap buffers
      swap_buffers();
  }
  ```
- [ ] Synchronize CUDA stream after OptiX launch
- [ ] Clean up all resources on exit

**Key Implementation Notes:**
- Order matters: map → render → unmap → update texture
- PBO must be bound as PIXEL_UNPACK_BUFFER for texture update
- Use nullptr for texture data pointer (reads from bound PBO)

### 10. Performance Monitoring & Validation

- [ ] Implement frame time measurement (CPU-side):
  - Use `std::chrono::high_resolution_clock`
  - Measure entire frame time
- [ ] Implement CUDA event timing (GPU-side):
  - Create events: `cudaEventCreate`
  - Record before/after OptiX launch
  - Calculate elapsed time: `cudaEventElapsedTime`
- [ ] Add FPS counter (print to console)
- [ ] Print memory usage at startup
- [ ] Verify no CUDA/OptiX errors in console

**Performance Baselines (1080p):**
- Application overhead: < 0.1ms
- OptiX launch (gradient): < 0.05ms
- CUDA-GL transfer: < 0.5ms
- OpenGL display: < 0.1ms
- Total frame time: < 1ms (excluding VSync)
- Target FPS: 60+ (VSync limited)

---

## Common Issues & Solutions

### OptiX Initialization Fails
**Symptoms:** `optixInit()` or `optixDeviceContextCreate()` returns error  
**Solutions:**
- Verify GPU has RT cores (RTX series required)
- Update NVIDIA drivers to latest version
- Check OptiX SDK is properly installed
- Verify CUDA Toolkit version compatibility

### CUDA-GL Interop Fails
**Symptoms:** `cudaGraphicsGLRegisterBuffer()` fails  
**Solutions:**
- Ensure OpenGL context created before CUDA initialization
- Verify same physical GPU for both OpenGL and CUDA
- Check `cudaGLGetDevices()` returns valid device
- Try explicit device selection with `cudaSetDevice()`

### Black Screen or Corrupted Display
**Symptoms:** Window shows but no output or garbage pixels  
**Solutions:**
- Verify map/unmap sequence is correct
- Check buffer sizes match across all components
- Ensure OptiX writes to correct buffer pointer
- Verify texture format matches PBO data (RGBA32F)
- Add synchronization: `cudaStreamSynchronize()` after launch

### Poor Performance
**Symptoms:** Low FPS even with simple gradient  
**Solutions:**
- Disable debug/validation mode in release build
- Check VSync is not artificially limiting FPS
- Verify release build optimizations enabled
- Profile with CUDA events to find bottleneck
- Ensure no CPU-GPU sync points in loop

### Crash in optixLaunch
**Symptoms:** Segfault or CUDA error during launch  
**Solutions:**
- Verify stack sizes are set correctly
- Check SBT alignment and record sizes
- Validate launch params structure matches CUDA code
- Ensure pipeline is fully linked before launch
- Check PTX files loaded correctly

---

## Validation Tests

Run these tests to verify Phase 1 is complete:

1. **Resolution Test:** Launch at 720p, 1080p, 1440p, 4K
   - Verify no crashes or corruption at any resolution
   - Check frame times scale appropriately

2. **VSync Test:** Toggle VSync on/off
   - With VSync: Should cap at 60 FPS
   - Without VSync: Should exceed 500+ FPS (minimal work)

3. **Stability Test:** Run for 10+ minutes
   - Monitor for memory leaks (GPU memory usage should be stable)
   - Verify no performance degradation over time
   - Check for any CUDA/OptiX errors in console

4. **Gradient Test:** Verify visual output
   - Should see smooth color gradient (red left→right, green bottom→top)
   - No banding, artifacts, or black pixels
   - Gradient should update if window resized (if dynamic resize implemented)

5. **Shutdown Test:** Close application cleanly
   - No crash on exit
   - All resources freed (check with CUDA leak detector)
   - No error messages in console

---

## Reference Documentation

### Official Documentation

**NVIDIA OptiX:**
- Programming Guide: https://raytracing-docs.nvidia.com/optix7/guide/index.html
- API Reference: https://raytracing-docs.nvidia.com/optix7/api/index.html
- SDK Examples: https://github.com/NVIDIA/OptiX_Apps

**CUDA:**
- Toolkit Documentation: https://docs.nvidia.com/cuda/
- C Programming Guide: https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- Runtime API: https://docs.nvidia.com/cuda/cuda-runtime-api/
- OpenGL Interop: https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__OPENGL.html

**OpenGL:**
- 4.6 Specification: https://registry.khronos.org/OpenGL/specs/gl/glspec46.core.pdf
- Reference Pages: https://registry.khronos.org/OpenGL-Refpages/gl4/
- GLFW Documentation: https://www.glfw.org/documentation.html

### Additional Resources

**Learning Materials:**
- Ray Tracing in One Weekend: https://raytracing.github.io/
- Learn OpenGL: https://learnopengl.com/
- PBRT Book (free online): https://www.pbr-book.org/

**Example Code:**
- NVIDIA OptiX Samples: https://github.com/NVIDIA/OptiX_Apps
- CUDA Samples: https://github.com/NVIDIA/cuda-samples

---

## Next Steps After Phase 1

Once Phase 1 is validated and working correctly:

**Phase 2: Geometry & BVH** - Load 3D models, build acceleration structures, implement actual ray tracing with surface intersections

**Phase 3: ReSTIR Lighting** - Add realistic materials (GGX BRDF), implement ReSTIR GI and PT with runtime switching, integrate denoiser

**Phase 4: Spectral Rendering** - Hero wavelength sampling, dispersion effects

**Phase 5: Lens-Space ReSTIR** - Depth of field, realistic camera effects

**Phase 6-7: UI & Editing** - Custom UI system, interactive scene editing tools

---

## Development Notes

**Getting Started:**
1. Install CUDA Toolkit and OptiX SDK
2. Set up project structure as outlined above
3. Configure CMake to find all dependencies
4. Implement components in order (OpenGL → CUDA → OptiX)
5. Test each component before moving to next
6. Run validation tests when complete

**Debugging Tips:**
- Enable OptiX validation mode in debug builds
- Use CUDA error checking after every CUDA call
- Add OpenGL debug context for detailed error messages
- Print initialization success/failure for each subsystem
- Use `cudaDeviceSynchronize()` to catch async errors

**Performance Tips:**
- Profile early and often with CUDA events
- Minimize CPU-GPU synchronization points
- Use persistent buffers (no per-frame allocation)
- Disable validation in release builds
- Consider async compute for later phases

This completes Phase 1 specification. All subsequent phases will build on this foundation.