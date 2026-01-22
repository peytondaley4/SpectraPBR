# Real-Time OptiX Ray Tracing Engine - Phase 2 Specification

## Phase 2: Geometry Pipeline & BVH Construction

### Objective

Implement high-performance model loading, GPU memory management, and OptiX acceleration structure construction. Enable real-time ray tracing of complex models with multiple materials, textures, and instances.

**Success Criteria:**
- Load complex glTF 2.0 models with multiple meshes, materials, and textures
- Build optimized BVH structures (GAS and IAS) with compaction
- Render scenes with 100K-1M triangles at 60 FPS (1080p)
- Display correct materials with texture sampling and normal mapping
- Smooth camera navigation with FPS controls
- No crashes or memory leaks when loading/unloading models

---

## System Architecture Overview

### Complete Pipeline Flow

```
1. Model Loading (CPU)
   - Parse glTF file
   - Extract meshes, materials, textures
   - Process vertex data
   ↓
2. GPU Upload
   - Allocate and upload geometry buffers
   - Create CUDA texture objects
   - Upload material parameters
   ↓
3. BVH Construction
   - Build GAS per mesh (bottom-level)
   - Compact GAS (30-50% memory reduction)
   - Build IAS for scene (top-level)
   ↓
4. SBT Organization
   - One hit group per material
   - Store material params and textures
   ↓
5. Ray Tracing
   - Generate camera rays
   - Trace against IAS
   - Evaluate materials at hit points
```

---

## Implementation Checklist

### 1. Model Loading System

**Library:** tinygltf (header-only glTF 2.0 parser)

**What to Load:**
- [ ] Meshes (vertices: position, normal, tangent, UV; indices as triangles)
- [ ] Materials (PBR: base color, metallic, roughness, emissive, textures)
- [ ] Textures (decode images, extract sampler settings)
- [ ] Scene graph (nodes with transforms, hierarchy)

**Data Processing:**
- [ ] Generate tangents if missing (for normal mapping)
- [ ] Generate normals if missing (smooth normals)
- [ ] Handle missing materials (use defaults)
- [ ] Convert all data to GPU-friendly formats

**Data Structures Needed:**
- Vertex (position, normal, tangent, UV)
- Mesh (vertex array, index array, material ID, bounds)
- Material (PBR params, texture IDs)
- Texture (decoded pixels, width, height, sampler settings)
- Node (transform, mesh IDs, child IDs)
- Model (collection of all above)

---

### 2. GPU Memory Management

**Geometry Manager:**
- [ ] Track all GPU allocations
- [ ] Upload vertex buffers to CUDA device memory
- [ ] Upload index buffers (uint32 triangles)
- [ ] Implement reference counting for shared meshes
- [ ] Support instancing (same mesh, multiple transforms)
- [ ] Clean up when ref count reaches zero

**Memory Layout:**
- Each mesh: vertex buffer + index buffer on GPU
- Track metadata: counts, sizes, pointers, material ID
- Monitor total GPU memory usage

---

### 3. Texture Management

**CUDA Texture Objects:**
- [ ] Upload images to CUDA arrays
- [ ] Create texture objects with proper samplers
- [ ] Configure filtering (linear/nearest, mipmapping)
- [ ] Configure wrap modes (repeat/clamp/mirror from glTF)
- [ ] Track texture memory usage

**Texture Formats:**
- RGBA8 for color textures (base color, emissive)
- RG8 for normal maps (reconstruct Z to save memory)
- Single channel for metallic/roughness

**Optional (Recommended):**
- Generate mipmaps for better quality
- Use BC compression to reduce memory

---

### 4. Material Management

**Material Data:**
- [ ] Create GPU material structure
- [ ] Store PBR parameters (base color factor, metallic, roughness, emissive)
- [ ] Store texture object handles (0 if not present)
- [ ] Store alpha mode (opaque/mask/blend) and cutoff
- [ ] Upload array of materials to device memory

---

### 5. OptiX Acceleration Structures

#### GAS (Geometry Acceleration Structure) - Per Mesh

**Build Process:**
- [ ] Configure build input (vertex/index buffers, format, stride)
- [ ] Set build flags: `PREFER_FAST_TRACE` + `ALLOW_COMPACTION`
- [ ] Query memory requirements
- [ ] Allocate temporary and output buffers
- [ ] Build GAS with compaction property
- [ ] Compact to final size (30-50% reduction)
- [ ] Free temporary buffers
- [ ] Store compacted GAS handle and buffer

**Performance Notes:**
- Build time: ~50-100ms per million triangles (one-time)
- Only rebuild if geometry changes (never in Phase 2)
- Compaction is critical for memory efficiency

#### IAS (Instance Acceleration Structure) - Scene Level

**Build Process:**
- [ ] Create instance array (one entry per mesh instance)
- [ ] Set transform (3x4 matrix, row-major)
- [ ] Set GAS handle reference
- [ ] Set instance ID (for identification)
- [ ] Set SBT offset (material ID for hit group lookup)
- [ ] Upload instances to device
- [ ] Build IAS (no compaction needed, already small)

**Instance Management:**
- Multiple instances can reference same GAS (instancing)
- Different transforms per instance
- Rebuild IAS when instances change (fast, < 1ms)

---

### 6. Shader Binding Table Organization

**SBT Structure:**
- Raygen record (one)
- Miss record (one)
- Hit group records (one per material)

**Hit Group Record Contents:**
- Material parameters (base color, metallic, roughness, emissive)
- Texture object handles (base color, normal, metallic/roughness, emissive)
- Alpha mode and cutoff

**Material-to-SBT Mapping:**
- Instance sets SBT offset = material_id
- OptiX automatically selects correct hit group
- Multiple meshes can share same material → same SBT record

---

### 7. Camera System

**Camera Model:**
- [ ] Pinhole camera (position, look-at, up, FOV, aspect)
- [ ] Compute camera basis vectors (u, v, w)
- [ ] Update when camera moves

**FPS Controls:**
- [ ] WASD for movement (forward/back, strafe)
- [ ] QE for up/down
- [ ] Mouse look for rotation
- [ ] Shift for speed boost
- [ ] Smooth movement with delta time

**Integration:**
- [ ] Update camera each frame from input
- [ ] Upload camera parameters to launch params
- [ ] No BVH rebuild needed (camera is not geometry)

---

### 8. OptiX Programs

#### Raygen Program (Camera Rays)

**Functionality:**
- [ ] Calculate pixel position and NDC coordinates
- [ ] Generate ray from camera (origin + direction)
- [ ] Trace ray against IAS
- [ ] Receive color from hit/miss via payload
- [ ] Write result to output buffer

**Payload:**
- 3 float registers for RGB color

#### Closest Hit Program (Surface Shading)

**Functionality:**
- [ ] Get material data from SBT
- [ ] Get hit info (barycentrics, primitive index)
- [ ] Interpolate vertex attributes (UV, normal, tangent)
- [ ] Sample textures if present
- [ ] Apply normal mapping if normal texture exists
- [ ] For Phase 2: Output normal as color (debugging visualization)
- [ ] Pack color into payload

**Geometry Access:**
- Need vertex/index buffer pointers (in launch params or SBT)
- Use instance ID and primitive ID to index into buffers

**Normal Mapping:**
- Sample normal texture, remap [0,1] to [-1,1]
- Build TBN matrix (tangent, bitangent, normal)
- Transform normal from tangent space to world space

#### Miss Program (Background)

**Functionality:**
- [ ] Return background color (gradient or solid)
- [ ] Pack color into payload

---

### 9. Scene Management

**Scene Manager:**
- [ ] Track all loaded models
- [ ] Track all instances
- [ ] Coordinate geometry, texture, material managers
- [ ] Rebuild IAS when instances change

**Operations:**
- [ ] Add model (load from file, upload to GPU, build GAS)
- [ ] Add instance (create instance entry, rebuild IAS)
- [ ] Remove instance (decrease ref count, rebuild IAS, free if needed)
- [ ] Update instance transform (modify transform, rebuild IAS)

---

### 10. Main Loop Integration

**Startup:**
- [ ] Load test model(s)
- [ ] Initialize camera position/orientation
- [ ] Build all BVH structures

**Each Frame:**
- [ ] Process camera input
- [ ] Update camera parameters
- [ ] Rebuild IAS if needed (not needed in Phase 2 unless dynamic)
- [ ] Upload updated launch params
- [ ] Map PBO to CUDA
- [ ] Launch OptiX
- [ ] Unmap PBO
- [ ] Display to screen

**Additional:**
- [ ] Handle window resize (recreate buffers, update aspect ratio)
- [ ] Performance monitoring (FPS, frame time breakdown)
- [ ] Memory usage tracking

---

## Performance Targets

**BVH Construction (One-time at Load):**
- Small models (< 100K tris): < 50ms
- Medium models (100K-500K tris): 50-200ms
- Large models (500K-1M tris): 200-500ms
- Compaction: 30-50% memory reduction

**Runtime Performance (60 FPS @ 1080p):**
- Primary ray tracing: 1-5ms
- Simple shading (normals): < 1ms
- Total frame time: 5-10ms
- No stuttering during camera movement

**Memory Usage:**
- Vertices: ~48 bytes per vertex
- Indices: ~12 bytes per triangle
- BVH: ~1.5x geometry size (after compaction)
- Textures: Variable (RGBA8 = 4 bytes per pixel)
- Example: 1M triangle model + 4K textures ≈ 300-400 MB

**Scene Complexity:**
- 100K-500K triangles: 60 FPS easily
- 500K-1M triangles: 60 FPS achievable
- 1M-5M triangles: 30-60 FPS (depends on visibility)

---

## Common Issues & Solutions

**Model Loading Fails:**
- Verify glTF file is valid
- Check file paths and texture references
- Handle missing attributes gracefully

**Black Screen:**
- Check camera position (not inside geometry)
- Verify normals are valid (not NaN/zero)
- Ensure BVH build succeeded
- Test with simple solid color before textures

**Corrupted Geometry:**
- Verify index buffer format (uint32 triangles)
- Check vertex stride matches structure size
- Validate barycentrics interpolation
- Check transform matrices

**Texture Issues:**
- Verify texture handles are valid
- Check UV coordinates are reasonable
- Test sampler settings match expectations
- Start with solid colors, add textures incrementally

**Performance Issues:**
- Verify BVH compaction is working
- Check for unnecessary BVH rebuilds
- Consider texture mipmaps
- Profile OptiX launch vs display time

**Memory Leaks:**
- Track all CUDA allocations
- Implement proper cleanup in destructors
- Use reference counting for shared resources
- Test load/unload cycles

---

## Validation Tests

**Model Loading:**
- [ ] Simple cube (single mesh, single material)
- [ ] Complex model (DamagedHelmet - multiple materials, textures)
- [ ] Large scene (Sponza - many meshes)
- [ ] Model with missing data (should use defaults)

**Rendering:**
- [ ] Normal-mapped surfaces display correctly
- [ ] Texture sampling works (colors match)
- [ ] Multiple instances render correctly
- [ ] Camera navigation is smooth at 60 FPS

**Performance:**
- [ ] Measure BVH build time
- [ ] Verify 60 FPS with target triangle counts
- [ ] Check memory usage is reasonable
- [ ] Profile frame time breakdown

**Stability:**
- [ ] Load/unload models without leaks
- [ ] Window resize works correctly
- [ ] Extended runtime (10+ minutes) without issues
- [ ] Clean shutdown

---

## Reference Documentation

**Model Loading:**
- tinygltf: https://github.com/syoyo/tinygltf
- glTF 2.0 Spec: https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html
- Sample Models: https://github.com/KhronosGroup/glTF-Sample-Models

**OptiX:**
- Acceleration Structures Guide: https://raytracing-docs.nvidia.com/optix7/guide/index.html#acceleration_structures
- Performance Best Practices: https://raytracing-docs.nvidia.com/optix7/guide/index.html#performance

**CUDA:**
- Texture Memory: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#texture-memory
- Texture Objects: https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TEXTURE__OBJECT.html

---

## Next Steps After Phase 2

**Phase 3: ReSTIR Lighting**
- PBR material evaluation (GGX BRDF, Disney principled)
- Light sources (point, directional, area, environment)
- ReSTIR GI (1-2 bounce indirect) - 60 FPS target
- ReSTIR PT (full path tracing) - 30-60 FPS target
- Runtime mode switching
- OptiX denoiser integration

This completes Phase 2 specification.