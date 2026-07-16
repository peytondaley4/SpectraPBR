#pragma once

//------------------------------------------------------------------------------
// GPU Types Header for OptiX Programs
//
// This file contains all GPU-side structures shared between OptiX .cu files.
// It uses pure C-compatible syntax (no namespaces, no STL) for CUDA compatibility.
//
// These types must stay in sync with src/core/shared_types.h (CPU-side
// definitions). Field order, types, and padding must match exactly.
//------------------------------------------------------------------------------

#include <cuda_runtime.h>
#include <optix.h>

//------------------------------------------------------------------------------
// Constants
//------------------------------------------------------------------------------

// Ray types. The path tracer is iterative (loop in raygen), so there is no
// separate "indirect" ray type — every path segment is a RADIANCE ray whose
// closest-hit only reports hit info back to raygen.
constexpr unsigned int RAY_TYPE_RADIANCE = 0;
constexpr unsigned int RAY_TYPE_SHADOW   = 1;
constexpr unsigned int RAY_TYPE_COUNT    = 2;

// Alpha modes
constexpr unsigned int ALPHA_MODE_OPAQUE = 0;
constexpr unsigned int ALPHA_MODE_MASK   = 1;
constexpr unsigned int ALPHA_MODE_BLEND  = 2;

// NEE light-selection kinds — must match src/core/shared_types.h; packed
// into the high byte of light-alias-table entries.
constexpr unsigned int LIGHT_KIND_NONE  = 0;
constexpr unsigned int LIGHT_KIND_POINT = 1;
constexpr unsigned int LIGHT_KIND_DIR   = 2;
constexpr unsigned int LIGHT_KIND_AREA  = 3;
constexpr unsigned int LIGHT_KIND_ENV   = 4;

// Quality modes
constexpr unsigned int QUALITY_FAST     = 0;
constexpr unsigned int QUALITY_BALANCED = 1;
constexpr unsigned int QUALITY_HIGH     = 2;
constexpr unsigned int QUALITY_ACCURATE = 3;

//------------------------------------------------------------------------------
// GPU Vertex Structure (48 bytes, aligned)
// Layout optimized to avoid padding from float4 alignment requirements
//------------------------------------------------------------------------------
struct GpuVertex {
    float3 position;   // 12 bytes, offset 0  (required at offset 0 for OptiX)
    float u;           // 4 bytes,  offset 12 (UV.x in padding slot)
    float3 normal;     // 12 bytes, offset 16
    float v;           // 4 bytes,  offset 28 (UV.y in padding slot)
    float4 tangent;    // 16 bytes, offset 32 (w = bitangent sign: +1 or -1)
};

//------------------------------------------------------------------------------
// GPU Material Structure (for SBT hit group records and the global material
// array read by raygen)
// Supports glTF 2.0 metallic-roughness workflow and common extensions
//------------------------------------------------------------------------------
struct GpuMaterial {
    // Core PBR properties
    float4 baseColor;           // Base color factor (RGBA)
    float metallic;             // Metallic factor [0, 1]
    float roughness;            // Roughness factor [0, 1]
    float2 _pad0;               // Padding
    float3 emissive;            // Emissive color (RGB)
    float _pad1;                // Padding

    // Core texture objects (0 = no texture)
    cudaTextureObject_t baseColorTex;
    cudaTextureObject_t normalTex;
    cudaTextureObject_t metallicRoughnessTex;
    cudaTextureObject_t emissiveTex;

    // KHR_materials_transmission (Glass/Water)
    float transmission;         // Transmission factor [0, 1]
    float ior;                  // Index of refraction (default 1.5)
    float2 _pad2;               // Padding
    cudaTextureObject_t transmissionTex;

    // KHR_materials_volume (Absorption)
    float3 attenuationColor;    // Absorption color
    float attenuationDistance;  // Distance for Beer's law (0 = no absorption)
    float thickness;            // Material thickness
    float3 _pad3;               // Padding

    // KHR_materials_clearcoat (Car Paint)
    float clearcoat;            // Clearcoat intensity [0, 1]
    float clearcoatRoughness;   // Clearcoat roughness [0, 1]
    float2 _pad4;               // Padding
    cudaTextureObject_t clearcoatTex;
    cudaTextureObject_t clearcoatRoughnessTex;
    cudaTextureObject_t clearcoatNormalTex;

    // KHR_materials_sheen (Cloth/Velvet)
    float3 sheenColor;          // Sheen color factor
    float sheenRoughness;       // Sheen roughness [0, 1]
    cudaTextureObject_t sheenColorTex;
    cudaTextureObject_t sheenRoughnessTex;

    // KHR_materials_specular (Fine-tune reflectance)
    float specularFactor;       // Specular strength [0, 1]
    float3 _pad5;               // Padding
    float3 specularColorFactor; // Specular color tint
    float _pad6;                // Padding
    cudaTextureObject_t specularTex;
    cudaTextureObject_t specularColorTex;

    // Occlusion texture
    cudaTextureObject_t occlusionTex;
    float occlusionStrength;    // Occlusion strength [0, 1]

    // Alpha settings
    unsigned int alphaMode;     // 0 = OPAQUE, 1 = MASK, 2 = BLEND
    float alphaCutoff;          // Cutoff for MASK mode
    unsigned int doubleSided;   // Non-zero if double-sided
};

//------------------------------------------------------------------------------
// Light Structures (GPU-side)
//------------------------------------------------------------------------------
struct GpuPointLight {
    float3 position;        // World position
    float radius;           // Radius for soft shadows (0 = point source)
    float3 intensity;       // Color * power (lumens or watts)
    float _pad;
};

struct GpuDirectionalLight {
    float3 direction;       // Normalized direction (points toward source)
    float angularDiameter;  // Angular diameter for soft shadows (0 = hard)
    float3 irradiance;      // Color * intensity
    float _pad;
};

// Area light. Two kinds share this struct:
//  - Virtual rectangle (triCount == 0): UI-created light with no geometry in
//    the BVH. Sampled as a rectangle; BSDF rays can never hit it, so NEE takes
//    full weight (no MIS needed).
//  - Mesh light (triCount > 0): extracted from an emissive mesh that IS in the
//    BVH. Sampled from its actual triangles (area_light_tris); MIS-weighted
//    against BSDF/guide sampling because BSDF rays can also hit the geometry.
struct GpuAreaLight {
    float3 position;        // Center position (virtual rect) / centroid (mesh)
    float _pad0;
    float3 normal;          // Surface normal (virtual rect orientation)
    float _pad1;
    float3 tangent;         // Surface tangent (virtual rect orientation)
    float _pad2;
    float3 emission;        // Emitted radiance (color * intensity)
    float area;             // Total surface area for PDF calculation
    float2 size;            // Width and height (virtual rect only)
    unsigned int triOffset; // First triangle in area_light_tris (mesh light)
    unsigned int triCount;  // Triangle count (0 = virtual rectangle)
    unsigned int instanceId;// Owning instance (mesh light; 0xFFFFFFFF for virtual)
    unsigned int _pad3[3];
};

//------------------------------------------------------------------------------
// Camera Parameters
//------------------------------------------------------------------------------
struct GpuCameraParams {
    float3 position;            // Camera world position
    float _pad0;
    float3 forward;             // Camera forward direction (normalized)
    float _pad1;
    float3 right;               // Camera right direction (normalized)
    float _pad2;
    float3 up;                  // Camera up direction (normalized)
    float _pad3;

    float fovY;                 // Vertical field of view in radians
    float aspectRatio;          // Width / Height
    float nearPlane;            // Near clipping plane
    float farPlane;             // Far clipping plane
};

//------------------------------------------------------------------------------
// Pick Result Buffer (returned from pick mode: instance ID + world hit position)
//------------------------------------------------------------------------------
struct PickResultBuffer {
    unsigned int instanceId;
    float hitX;
    float hitY;
    float hitZ;
};

//------------------------------------------------------------------------------
// Launch Parameters (passed to all OptiX programs)
// MUST match src/core/shared_types.h::LaunchParams field-for-field.
//------------------------------------------------------------------------------
struct GpuLaunchParams {
    // Output buffer (final display)
    float4* output_buffer;
    // Accumulation buffer (for progressive AA)
    float4* accumulation_buffer;
    unsigned int width;
    unsigned int height;
    unsigned int frame_index;
    unsigned int accumulated_frames;    // Number of frames accumulated (reset on camera move)

    // Camera
    GpuCameraParams camera;

    // Scene traversable
    OptixTraversableHandle scene_handle;

    // Geometry buffer arrays (indexed by instance ID)
    CUdeviceptr* vertex_buffers;    // Array of pointers to GpuVertex arrays
    CUdeviceptr* index_buffers;     // Array of pointers to uint32_t index arrays

    // Per-instance data (indexed by instance ID). Raygen shades hits itself,
    // so it needs material + transform access outside the SBT.
    const GpuMaterial* materials;                   // One per GAS/material slot
    const unsigned int* instance_material_indices;  // instance -> material slot
    const float* instance_transforms;               // 12 floats per instance (3x4 row-major, object->world)
    const float* instance_normal_transforms;        // 12 floats per instance (inverse-transpose linear part)
    const unsigned int* instance_light_indices;     // instance -> area light index, 0xFFFFFFFF if none

    // Lighting
    GpuPointLight* point_lights;
    unsigned int point_light_count;
    unsigned int _pad_lights0;
    GpuDirectionalLight* directional_lights;
    unsigned int directional_light_count;
    unsigned int _pad_lights1;
    GpuAreaLight* area_lights;
    unsigned int area_light_count;
    // Lights-only selection total. Selection weights (must match
    // LightManager::syncToGpu exactly): point = lum(intensity),
    // directional = lum(irradiance) * 10, area = lum(emission) * area.
    float total_light_luminance;
    // Mesh-light triangles: 3 float4 per triangle.
    //   tri[0].xyz = v0, tri[0].w = cumulative area fraction within the light (CDF)
    //   tri[1].xyz = v1, tri[1].w = triangle area
    //   tri[2].xyz = v2, tri[2].w = unused
    const float4* area_light_tris;
    // Environment selection weight in the NEE light pick (host-computed,
    // ~total incident env flux). 0 when no env importance data.
    float env_selection_weight;
    // Per-bounce contribution clamp for firefly suppression. Biased — set to
    // FLT_MAX (host: QUALITY_ACCURATE) for unbiased accumulation.
    float firefly_clamp;

    // Environment map (equirectangular HDR)
    cudaTextureObject_t environment_map;  // 0 = none
    float environment_intensity;
    float _pad_env;

    // Environment map importance sampling — Walker/Vose alias table over the
    // W*H texels (O(1) sampling vs the old log2(W)+log2(H) dependent CDF
    // binary search). Distribution is identical (per-texel selection prob =
    // sin(theta)-weighted luminance, normalized; uniform sub-texel jitter), so
    // convergence is unchanged — this is purely a latency win. env_pmf gives
    // the per-texel probability for environmentPdf() (the MIS density), kept in
    // lockstep with the sampler so MIS stays exact.
    const float* env_alias_prob;        // [W*H] accept-probability of each bucket
    const unsigned int* env_alias_idx;  // [W*H] fallback texel of each bucket
    const float* env_pmf;               // [W*H] per-texel selection probability
    unsigned int env_width;
    unsigned int env_height;
    float env_total_luminance;              // (host-side selection weight; unused on device)
    float _pad_env_cdf;

    // Scene-light alias table (Walker/Vose) for O(1) NEE light selection —
    // replaces the per-vertex linear luminance-CDF walk over all lights.
    // Host-built by LightManager::syncToGpu from the SAME selection weights
    // that sum to total_light_luminance; each entry packs (kind << 24 | index)
    // with the LIGHT_KIND_* constants.
    const float* light_alias_prob;           // [light_alias_count] accept prob
    const unsigned int* light_alias_idx;     // [light_alias_count] fallback bucket
    const unsigned int* light_alias_entries; // [light_alias_count] packed kind/index
    unsigned int light_alias_count;

    // Quality and rendering settings
    unsigned int quality_mode;
    unsigned int samples_per_pixel;     // SPP per frame (higher = less noise, slower)
    unsigned int max_bounce_depth;      // Maximum path length (path vertices after the camera)
    unsigned int _pad_quality;

    // UI selection (UINT32_MAX = no selection)
    unsigned int selected_instance_id;
    unsigned int _pad_selection;

    // Picking mode
    PickResultBuffer* pick_result;  // Device buffer for pick result (instance ID + world hit position)
    unsigned int pick_x;            // Pick pixel X coordinate
    unsigned int pick_y;            // Pick pixel Y coordinate
    unsigned int pick_mode;         // 0 = normal render, 1 = pick mode (single ray)

    // Path guide grid (sparse, device-resident cell table)
    float* path_guide_data;
    unsigned int path_guide_num_levels;
    unsigned int path_guide_entry_stride;
    unsigned int path_guide_base_resolution;
    float path_guide_per_level_scale;
    float path_guide_bounds_min[3];
    float path_guide_bounds_max[3];
    unsigned int path_guide_enabled;    // 1 = use guide for sampling, 0 = BSDF only
    float path_guide_mis_weight;        // Probability of sampling the guide (one-sample MIS alpha)
    // Adaptive level parameters (from config)
    unsigned int path_guide_start_level; // Base level allocated on first touch
    unsigned int path_guide_min_level;   // Retired from the hot path (UI compat)
    unsigned int path_guide_max_level;   // Finest allowed level
    unsigned int path_guide_level_resolutions[16];  // Precomputed floor(base_res * scale^level)

    // Cell table: hash for O(1) lookup + insert-on-first-touch allocation
    unsigned long long* path_guide_hash_keys;          // (level<<48 | morton), empty = 0xFFFFFFFFFFFFFFFF (CAS target)
    unsigned int* path_guide_hash_values;              // cell index or sentinel (PENDING/FULL)
    unsigned int path_guide_hash_table_size;            // power of 2
    unsigned int path_guide_hash_shift;                 // 64 - log2(hash_table_size)
    unsigned long long* path_guide_cell_keys;           // packed key per allocated cell
    unsigned int* path_guide_cell_counter;              // bump allocator (1 element)
    unsigned int path_guide_cell_capacity;              // max live cells

    // Debug statistics (atomic counters, reset each frame)
    unsigned int* path_guide_debug_stats;  // [0]=attempts, [1]=cell_found, [2]=valid_lobe, [3]=below_horizon, [4]=contributed, [5]=bsdf_sampled
    unsigned int path_guide_debug_enabled; // 1 = collect debug stats

    // Precomputed per-frame constants (avoid transcendentals on GPU)
    float tan_half_fov_y;               // tanf(camera.fovY * 0.5f)
    float tan_half_fov_x;               // tan_half_fov_y * camera.aspectRatio
    float pixel_world_size;             // (2 * tan_half_fov_y) / height
    unsigned int path_guide_training;   // 1 = deposit training samples (host refits are running)

    // Denoiser AOV guide buffers (progressive average, same frame counter)
    float4* aov_albedo_buffer;          // First-hit baseColor
    float4* aov_normal_buffer;          // First-hit camera-space normal
};

//------------------------------------------------------------------------------
// SBT Record Data Structures
//------------------------------------------------------------------------------

// Hit group data follows the SBT record header.
// Only the anyhit alpha-test programs read this (raygen shades from the global
// material array); kept per-record so anyhit needs no extra indirection.
struct HitGroupData {
    GpuMaterial material;
    unsigned int geometryIndex;     // Index into vertex_buffers/index_buffers
};

// Miss data follows the SBT record header
struct MissData {
    float3 backgroundColor;
    float _pad;
};

//------------------------------------------------------------------------------
// Global Launch Parameters Declaration
// Each .cu file includes this header and gets access to params
//------------------------------------------------------------------------------
extern "C" {
    __constant__ GpuLaunchParams params;
}
