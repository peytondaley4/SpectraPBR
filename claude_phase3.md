# Real-Time OptiX Ray Tracing Engine - Phase 3 Specification

## Phase 3: PBR Materials & Basic Lighting

### Objective

Implement comprehensive physically-based material system supporting all glTF 2.0 materials and common extensions. Add basic lighting with proper shadow rays and BRDF evaluation for photorealistic rendering.

**Success Criteria:**
- All glTF 2.0 base materials render correctly (metallic-roughness workflow)
- Support common extensions (transmission, IOR, clearcoat, sheen)
- Multiple light types working (point, directional, area lights)
- Shadow rays for proper visibility testing
- Materials look photorealistic under direct lighting
- Maintain 60 FPS at 1080p with moderate scene complexity
- Runtime BRDF quality modes (Fast/Balanced/Quality/Accurate)

---

## System Architecture

### Material & Lighting Pipeline

```
1. Material Evaluation
   - Load glTF material properties
   - Sample all textures (base color, normal, metallic/roughness, etc.)
   - Apply normal mapping
   - Determine BRDF type based on material properties
   ↓
2. Light Sampling
   - Select light source
   - Calculate light direction and intensity
   - Trace shadow ray for visibility
   ↓
3. BRDF Evaluation
   - Evaluate appropriate BRDF (GGX, Lambertian, Dielectric, etc.)
   - Apply Fresnel, masking-shadowing, normal distribution
   - Combine diffuse and specular components
   ↓
4. Final Shading
   - Multiply BRDF * light_intensity * visibility
   - Add emissive contribution
   - Output final color
```

---

## Implementation Checklist

### 1. Material System Foundation

**Reference:** https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#materials

#### glTF 2.0 Base Properties

**Core Metallic-Roughness Workflow:**
- [ ] Base color factor (RGBA) + texture
- [ ] Metallic factor (0-1) + texture (B channel)
- [ ] Roughness factor (0-1) + texture (G channel)
- [ ] Normal texture (tangent-space)
- [ ] Occlusion texture (R channel, ambient occlusion)
- [ ] Emissive factor (RGB) + texture
- [ ] Alpha mode: OPAQUE, MASK, BLEND
- [ ] Alpha cutoff value (for MASK mode)
- [ ] Double-sided flag

**GPU Material Structure:**
- [ ] Expand Phase 2 material struct
- [ ] Add all texture object handles
- [ ] Add all factors and flags
- [ ] Ensure 16-byte alignment for GPU

#### glTF Extension Support

**KHR_materials_transmission (Glass/Water):**
- [ ] Transmission factor (0-1)
- [ ] Transmission texture
- [ ] Enables refraction

**KHR_materials_ior (Index of Refraction):**
- [ ] IOR value (1.0-2.5, default 1.5)
- [ ] Affects Fresnel and refraction

**KHR_materials_volume (Absorption):**
- [ ] Thickness factor
- [ ] Attenuation distance and color
- [ ] Beer's law absorption

**KHR_materials_clearcoat (Car Paint):**
- [ ] Clearcoat factor (0-1)
- [ ] Clearcoat roughness + texture
- [ ] Clearcoat normal map
- [ ] Second specular layer

**KHR_materials_sheen (Cloth/Velvet):**
- [ ] Sheen color factor + texture
- [ ] Sheen roughness + texture
- [ ] Retroreflective lobe for fabrics

**KHR_materials_specular (Fine-tune):**
- [ ] Specular factor and color
- [ ] Additional control over reflectance

**Extension Loading:**
- [ ] Check material.extensions map in tinygltf
- [ ] Parse extension properties if present
- [ ] Set defaults if extension not present
- [ ] Store in GPU material structure

---

### 2. BRDF Implementation

**Reference:** 
- GGX: https://www.cs.cornell.edu/~srm/publications/EGSR07-btdf.html
- VNDF Sampling: http://jcgt.org/published/0007/04/01/
- Disney BRDF: https://blog.selfshadow.com/publications/s2012-shading-course/burley/s2012_pbs_disney_brdf_notes_v3.pdf

#### Lambertian Diffuse (Fast Mode)

**When to Use:**
- Fast preview mode
- Fully diffuse materials (roughness = 1, metallic = 0)

**Implementation:**
- [ ] BRDF evaluation: base_color / π
- [ ] Cosine-weighted hemisphere sampling
- [ ] PDF: cos(θ) / π

**Performance:** ~0.1ms overhead per frame

#### GGX Microfacet BRDF (Recommended Default)

**When to Use:**
- Default for all materials
- Handles full metallic-roughness range
- Best quality/performance balance

**Components:**

**Normal Distribution Function (D - GGX):**
- [ ] Input: half vector, roughness
- [ ] Formula: D_GGX(h, α) = α² / (π((n·h)²(α²-1)+1)²)
- [ ] Roughness remapping: α = roughness²

**Geometry Term (G - Smith Height-Correlated):**
- [ ] Input: view direction, light direction, roughness
- [ ] Smith masking-shadowing function
- [ ] Height-correlated for accuracy

**Fresnel Term (F - Schlick Approximation):**
- [ ] Input: view-half angle, F0
- [ ] F0 for dielectrics: 0.04
- [ ] F0 for metals: base_color
- [ ] Formula: F0 + (1-F0)(1-cos(θ))⁵

**Complete BRDF:**
- [ ] Specular: D * G * F / (4 * NoV * NoL)
- [ ] Diffuse: (1 - F) * (1 - metallic) * base_color / π
- [ ] Total: specular + diffuse

**Importance Sampling (VNDF):**
- [ ] Sample visible normal distribution
- [ ] Better variance than uniform hemisphere
- [ ] Returns sample direction and PDF
- [ ] Critical for low sample counts

**Performance:** ~0.5ms overhead per frame

#### Accurate Dielectric BSDF (Glass, Water)

**When to Use:**
- Materials with transmission > 0
- Needs exact Fresnel for total internal reflection
- Glass, water, clear plastics

**Fresnel Dielectric (Exact):**
- [ ] Full unpolarized Fresnel equations
- [ ] Handles entering vs exiting medium
- [ ] Detects total internal reflection
- [ ] Returns exact reflection coefficient

**Refraction:**
- [ ] Snell's law for direction calculation
- [ ] Handle TIR case (100% reflection)
- [ ] Support nested dielectrics (IOR stack)

**BSDF Sampling:**
- [ ] Choose reflection vs transmission based on Fresnel
- [ ] For rough dielectrics: sample GGX microfacet normal first
- [ ] Calculate direction from microfacet
- [ ] Return direction, PDF, and type (reflect/refract)

**Rough Transmission:**
- [ ] Use GGX distribution for microfacets
- [ ] Sample normal, then refract through it
- [ ] Frosted glass, rough water surface

**Performance:** ~2-5ms overhead (only when transmission active)

#### Conductor BRDF (Complex Metals)

**Reference:** http://jcgt.org/published/0003/04/03/

**When to Use:**
- High-quality metal rendering
- Colored reflections (gold, copper)
- Accurate mode only

**Complex IOR:**
- [ ] Store eta (n) and k for RGB channels
- [ ] Presets for common metals:
  - Gold: n=[0.47,0.37,1.44], k=[2.82,2.35,1.77]
  - Silver: n=[0.15,0.15,0.16], k=[3.64,3.48,2.88]
  - Copper: n=[0.27,0.68,1.29], k=[3.61,2.63,2.29]
  - Aluminum: n=[1.66,0.88,0.52], k=[9.22,6.27,4.86]

**Conductor Fresnel:**
- [ ] Full conductor equations with complex IOR
- [ ] Returns RGB reflectance
- [ ] More accurate than Schlick for metals

**Usage:**
- [ ] Detect metallic = 1.0 materials
- [ ] Use conductor Fresnel instead of Schlick
- [ ] Slightly more expensive but visually better

**Performance:** ~1ms overhead

---

### 3. Material Evaluation Pipeline

**In Closest Hit Program:**

**Step 1: Get Material Data**
- [ ] Retrieve material from SBT or launch params
- [ ] Get surface interaction (position, normal, tangent, UV)

**Step 2: Sample Textures**
- [ ] Base color: sample and multiply by factor
- [ ] Metallic/roughness: sample packed texture (G=rough, B=metal)
- [ ] Normal: sample, remap [0,1]→[-1,1], transform to world space
- [ ] Emissive: sample and multiply by factor
- [ ] Occlusion: sample (affects ambient only, skip for now)

**Step 3: Normal Mapping**
- [ ] Build TBN matrix (tangent, bitangent, normal)
- [ ] Transform sampled normal from tangent to world space
- [ ] Normalize result
- [ ] Use as shading normal

**Step 4: Alpha Testing (MASK mode)**
- [ ] If alpha < alpha_cutoff: discard hit (anyhit program)
- [ ] For BLEND mode: store alpha for later blending

**Step 5: Determine BRDF Type**
- [ ] Check transmission factor
- [ ] Check metallic factor
- [ ] Check extensions (clearcoat, sheen)
- [ ] Select appropriate BRDF evaluation function

---

### 4. Light Sources

**Reference:** https://www.pbr-book.org/ (Chapter 12)

#### Light Types

**Point Light:**
- [ ] Position (world space)
- [ ] Intensity (color * power in watts or lumens)
- [ ] Radius (for soft shadows, optional)
- [ ] Falloff: intensity / distance²

**Directional Light:**
- [ ] Direction (normalized, points toward source)
- [ ] Irradiance (color * intensity)
- [ ] No falloff (infinite distance)
- [ ] Optional: angular diameter for soft shadows

**Area Light (Emissive Mesh):**
- [ ] Triangle mesh with emissive material
- [ ] Emissive factor * base color
- [ ] Uniform triangle sampling
- [ ] Area and normal for intensity calculation

**Environment Map (HDR Skybox):**
- [ ] Equirectangular HDR texture
- [ ] Precomputed importance sampling map (CDF)
- [ ] Direction to UV conversion
- [ ] Intensity from texture lookup

#### Light Management

**Light Array:**
- [ ] Store all lights in GPU array
- [ ] Upload to device memory
- [ ] Pass pointer in launch params
- [ ] Include light count

**Light Selection:**
- [ ] For few lights (< 10): iterate all lights
- [ ] For many lights: sample based on distance/power
- [ ] Optional: build light BVH (future optimization)

---

### 5. Direct Lighting Implementation

**Basic Algorithm:**

```
For each pixel:
  1. Generate camera ray
  2. Find closest intersection
  3. Evaluate material at hit point
  4. For each light (or sample N lights):
     a. Get light sample (position/direction, intensity)
     b. Calculate light vector and distance
     c. Trace shadow ray
     d. If visible:
        - Evaluate BRDF(view, light, surface)
        - Accumulate: BRDF * light_intensity * cos(θ)
  5. Add emissive contribution
  6. Return total radiance
```

**Shadow Rays:**
- [ ] Origin: hit point + normal * epsilon (avoid self-intersection)
- [ ] Direction: toward light sample
- [ ] tmax: distance to light (for point/area), infinity (directional)
- [ ] Flags: TERMINATE_ON_FIRST_HIT (optimization)
- [ ] Return: boolean visibility

**Soft Shadows (Optional):**
- [ ] Sample multiple points on area lights
- [ ] Sample points within directional light disk
- [ ] Average results
- [ ] More samples = softer shadows, more expensive

**Multiple Importance Sampling (MIS):**
- [ ] For area lights: combine light sampling + BRDF sampling
- [ ] Balance heuristic: w = pdf_a / (pdf_a + pdf_b)
- [ ] Power heuristic: w = pdf_a² / (pdf_a² + pdf_b²)
- [ ] Reduces variance significantly
- [ ] Optional but recommended for quality

---

### 6. Performance Modes

**Fast Mode:**
- Lambertian diffuse only
- GGX specular with Schlick Fresnel
- No transmission/refraction
- Single shadow ray per light
- Target: 60+ FPS

**Balanced Mode (Default):**
- Full GGX BRDF
- Schlick Fresnel
- Basic transmission (alpha blend, no refraction)
- Single shadow ray per light
- Target: 60 FPS

**Quality Mode:**
- Full GGX with VNDF sampling
- Conductor Fresnel for metals
- Rough transmission with microfacets
- Multiple shadow samples for soft shadows
- Clearcoat and sheen support
- Target: 30-45 FPS

**Accurate Mode:**
- Exact dielectric Fresnel
- Full refraction with IOR
- Conductor with complex IOR
- All extensions supported
- High-quality soft shadows
- Target: 15-30 FPS

**Implementation:**
- [ ] Global quality setting in UI
- [ ] Per-material override (optional)
- [ ] Runtime switching without rebuild
- [ ] Conditional compilation or runtime branching

---

### 7. OptiX Program Updates

**Launch Params Additions:**
- [ ] Light array pointer and count
- [ ] Environment map texture (if present)
- [ ] Quality mode setting
- [ ] Random seed per pixel

**Raygen Program:**
- [ ] Generate camera ray (unchanged from Phase 2)
- [ ] Initialize random seed per pixel
- [ ] Trace into scene

**Closest Hit Program (Major Rewrite):**
- [ ] Get material from SBT
- [ ] Interpolate vertex attributes
- [ ] Sample all textures
- [ ] Apply normal mapping
- [ ] Determine BRDF type
- [ ] Loop over lights:
  - Sample light
  - Trace shadow ray
  - Evaluate BRDF if visible
  - Accumulate contribution
- [ ] Add emissive
- [ ] Return final color via payload

**Anyhit Program (New, for Alpha):**
- [ ] For MASK materials: test alpha < cutoff, optixIgnoreIntersection()
- [ ] For BLEND materials: modify payload with alpha
- [ ] For opaque: do nothing (no anyhit needed)

**Miss Program:**
- [ ] Sample environment map if present
- [ ] Return sky color or solid background
- [ ] Pack into payload

---

### 8. Testing & Validation

**Test Models:**
- [ ] DamagedHelmet.glb - Metal + paint surfaces
- [ ] FlightHelmet.glb - Complex multi-material
- [ ] WaterBottle.glb - Glass with transmission
- [ ] MetalRoughSpheres.glb - Material parameter sweep
- [ ] ClearCoatTest.glb - Car paint (if available)

**Visual Validation:**
- [ ] Metals appear reflective with colored tint
- [ ] Rough surfaces scatter light appropriately
- [ ] Glass refracts correctly (objects visible through)
- [ ] Normal maps add surface detail
- [ ] Shadows are sharp and correct
- [ ] Emissive materials glow

**Performance Validation:**
- [ ] Measure frame time per BRDF type
- [ ] Verify 60 FPS with balanced mode
- [ ] Check GPU memory usage
- [ ] Profile shadow ray cost

---

## Performance Targets

**Frame Budget (1080p, 60 FPS = 16.67ms):**
- BVH traversal: 1-2ms (from Phase 2)
- Material evaluation: 1-2ms
- BRDF evaluation: 0.5-2ms (depends on mode)
- Shadow rays: 2-5ms (depends on light count)
- Display overhead: 0.5ms
- **Total: 5-12ms → 60-120 FPS**

**Quality Mode Targets:**
- Fast: 80-120 FPS
- Balanced: 60-80 FPS
- Quality: 40-60 FPS
- Accurate: 30-45 FPS

**Memory Usage:**
- Material data: ~128 bytes per material
- Light data: ~64 bytes per light
- G-buffer (if needed): ~50 bytes per pixel @ 1080p = 100 MB
- Total additional: < 200 MB

---

## Common Issues & Solutions

**Materials Look Wrong:**
- Check texture sampling (correct UV, proper formats)
- Verify normal mapping TBN matrix construction
- Ensure metallic/roughness texture channels correct (G=rough, B=metal)
- Test with solid colors before textures

**Glass Not Transparent:**
- Verify transmission factor > 0
- Check IOR extension is parsed
- Ensure refraction ray is traced
- Test alpha blending before full refraction

**Performance Issues:**
- Too many lights → use light culling or importance sampling
- Complex BRDFs → verify quality mode is appropriate
- Shadow rays expensive → reduce light count or use simpler shadows
- Profile to find actual bottleneck

**Shadow Artifacts:**
- Self-intersection → increase epsilon offset
- Missing shadows → check shadow ray tmax
- Acne/artifacts → adjust bias based on surface slope

---

## Reference Documentation

**Materials:**
- glTF 2.0 Spec: https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#materials
- Extension Registry: https://github.com/KhronosGroup/glTF/tree/main/extensions/2.0/Khronos

**BRDF Theory:**
- GGX Paper: https://www.cs.cornell.edu/~srm/publications/EGSR07-btdf.html
- VNDF Sampling: http://jcgt.org/published/0007/04/01/
- Disney BRDF: https://blog.selfshadow.com/publications/s2012-shading-course/burley/s2012_pbs_disney_brdf_notes_v3.pdf
- Conductor Fresnel: http://jcgt.org/published/0003/04/03/

**Rendering:**
- PBRT Book: https://www.pbr-book.org/
- Real-Time Rendering: https://www.realtimerendering.com/

---

## Next Steps After Phase 3

Once basic lighting and materials are working well:

**Future Phase: Advanced Lighting**
- ReSTIR for efficient many-light sampling
- Global illumination (indirect bounces)
- Caustics and complex light transport
- OptiX denoiser integration
- Temporal accumulation

**Future Phase: Spectral Rendering**
- Hero wavelength sampling
- Dispersion effects
- Wavelength-dependent materials

**Future Phase: Advanced Camera**
- Depth of field (thin lens model)
- Lens-space ReSTIR
- Motion blur

This completes Phase 3 specification focused on materials and direct lighting.