#pragma once

#include <cuda_runtime.h>
#include <optix.h>
#include <cstdint>
#include <vector>
#include <string>

namespace spectra {

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
static_assert(sizeof(GpuVertex) == 48, "GpuVertex must be 48 bytes");

//------------------------------------------------------------------------------
// GPU Material Structure (for SBT hit group records)
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
    float attenuationDistance;  // Distance for Beer's law
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
    uint32_t alphaMode;         // 0 = OPAQUE, 1 = MASK, 2 = BLEND
    float alphaCutoff;          // Cutoff for MASK mode
    uint32_t doubleSided;       // Non-zero if double-sided
};

// Alpha mode constants
constexpr uint32_t ALPHA_MODE_OPAQUE = 0;
constexpr uint32_t ALPHA_MODE_MASK   = 1;
constexpr uint32_t ALPHA_MODE_BLEND  = 2;

//------------------------------------------------------------------------------
// Quality Mode (runtime BRDF quality selection)
//------------------------------------------------------------------------------
enum QualityMode : uint32_t {
    QUALITY_FAST     = 0,   // Lambertian + basic GGX, Schlick Fresnel
    QUALITY_BALANCED = 1,   // Full GGX, Schlick Fresnel (default)
    QUALITY_HIGH     = 2,   // VNDF sampling, conductor Fresnel for metals
    QUALITY_ACCURATE = 3    // Exact dielectric Fresnel, full refraction
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

struct GpuAreaLight {
    float3 position;        // Center position
    float _pad0;
    float3 normal;          // Surface normal
    float _pad1;
    float3 tangent;         // Surface tangent (for orientation)
    float _pad2;
    float3 emission;        // Emitted radiance (color * intensity)
    float area;             // Surface area for PDF calculation
    float2 size;            // Width and height (for rectangular lights)
    float2 _pad3;
};

//------------------------------------------------------------------------------
// Camera Parameters
//------------------------------------------------------------------------------
struct CameraParams {
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
    uint32_t instanceId;
    float hitX;
    float hitY;
    float hitZ;
};

//------------------------------------------------------------------------------
// Launch Parameters (passed to all OptiX programs)
//------------------------------------------------------------------------------
struct LaunchParams {
    // Output buffer (final display)
    float4* output_buffer;
    // Accumulation buffer (for progressive AA)
    float4* accumulation_buffer;
    uint32_t width;
    uint32_t height;
    uint32_t frame_index;
    uint32_t accumulated_frames;    // Number of frames accumulated (reset on camera move)

    // Camera
    CameraParams camera;

    // Scene traversable
    OptixTraversableHandle scene_handle;

    // Geometry buffer arrays (indexed by instance ID)
    CUdeviceptr* vertex_buffers;    // Array of pointers to GpuVertex arrays
    CUdeviceptr* index_buffers;     // Array of pointers to uint32_t index arrays

    // Material indices per instance (maps instance ID -> material SBT index)
    uint32_t* instance_material_indices;

    // Lighting
    GpuPointLight* point_lights;
    uint32_t point_light_count;
    uint32_t _pad_lights0;
    GpuDirectionalLight* directional_lights;
    uint32_t directional_light_count;
    uint32_t _pad_lights1;
    GpuAreaLight* area_lights;
    uint32_t area_light_count;
    float total_light_luminance;  // Precomputed sum of all light luminances for importance sampling

    // Environment map (equirectangular HDR)
    cudaTextureObject_t environment_map;  // 0 = none
    float environment_intensity;
    float _pad_env;

    // Environment map importance sampling CDFs
    cudaTextureObject_t env_conditional_cdf;  // P(u|v) - CDF per row (2D: width x height)
    cudaTextureObject_t env_marginal_cdf;     // P(v) - CDF for rows (1D: height elements)
    uint32_t env_width;
    uint32_t env_height;
    float env_total_luminance;              // For PDF normalization
    float _pad_env_cdf;

    // Quality and rendering settings
    QualityMode quality_mode;
    uint32_t samples_per_pixel;     // SPP per frame (higher = less noise, slower)
    uint32_t random_seed;           // Per-frame random seed for sampling
    uint32_t _pad_quality;

    // UI selection (UINT32_MAX = no selection)
    uint32_t selected_instance_id;
    uint32_t _pad_selection;

    // Picking mode
    PickResultBuffer* pick_result;  // Device buffer for pick result (instance ID + world hit position)
    uint32_t pick_x;            // Pick pixel X coordinate
    uint32_t pick_y;            // Pick pixel Y coordinate
    uint32_t pick_mode;         // 0 = normal render, 1 = pick mode (single ray)

    // Path guide grid (sparse, collision-free)
    const uint64_t* path_guide_morton_codes;   // Sorted per level (null = no grid)
    float* path_guide_data;
    const uint32_t* path_guide_level_offsets;  // num_levels+1
    uint32_t path_guide_num_levels;
    uint32_t path_guide_entry_stride;
    uint32_t path_guide_base_resolution;
    float path_guide_per_level_scale;
    float path_guide_bounds_min[3];
    float path_guide_bounds_max[3];
    // Staging for occupancy collection (closest-hit appends)
    uint32_t* path_guide_staging_buffer;
    uint32_t* path_guide_staging_count;   // Atomic
    uint32_t path_guide_staging_capacity;
    uint32_t debug_grid_visualize;  // 0 = off, 1 = show grid in viewport
    uint32_t debug_grid_level;      // Level to visualize (0 .. num_levels-1)
    uint32_t path_guide_no_jitter;  // When set, use center-pixel rays for deterministic staging
    uint32_t path_guide_enabled;    // 1 = use guide for sampling, 0 = BSDF only
    float path_guide_mis_weight;    // Blend factor for MIS (0.5 = balanced)
    // Adaptive level parameters (from config)
    uint32_t path_guide_start_level; // Initial level for new regions
    uint32_t path_guide_min_level;   // Coarsest allowed level
    uint32_t path_guide_max_level;   // Finest allowed level
    uint32_t path_guide_level_resolutions[16];  // Precomputed floor(base_res * scale^level)

    // Hash table for O(1) cell lookup (replaces binary search)
    const uint64_t* path_guide_hash_keys;        // (level<<48 | morton), empty = 0xFFFFFFFFFFFFFFFF
    const uint32_t* path_guide_hash_values;      // flat cell index
    uint32_t path_guide_hash_table_size;          // power of 2
    uint32_t path_guide_hash_shift;               // 64 - log2(hash_table_size)

    // Debug statistics (atomic counters, reset each frame)
    uint32_t* path_guide_debug_stats;  // [0]=attempts, [1]=cell_found, [2]=valid_lobe, [3]=below_horizon, [4]=contributed
    uint32_t path_guide_debug_enabled; // 1 = collect debug stats

    uint32_t max_bounce_depth;          // Max recursive bounces for glass/transmission

    // Precomputed per-frame constants (avoid transcendentals on GPU)
    float tan_half_fov_y;               // tanf(camera.fovY * 0.5f)
    float tan_half_fov_x;               // tan_half_fov_y * camera.aspectRatio
    float pixel_world_size;             // (2 * tan_half_fov_y) / height
    uint32_t stratified_grid_dim;       // ceil(sqrt(samples_per_pixel))
};

//------------------------------------------------------------------------------
// SBT Record Structures
//------------------------------------------------------------------------------
struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) RaygenRecord {
    char header[OPTIX_SBT_RECORD_HEADER_SIZE];
};

struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) MissRecord {
    char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    float3 backgroundColor;
    float _pad;
};

struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) HitGroupRecord {
    char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    GpuMaterial material;
    uint32_t geometryIndex;     // Index into vertex_buffers/index_buffers
    uint32_t _pad[3];
};

//------------------------------------------------------------------------------
// Ray Types
//------------------------------------------------------------------------------
constexpr uint32_t RAY_TYPE_RADIANCE = 0;
constexpr uint32_t RAY_TYPE_SHADOW   = 1;
constexpr uint32_t RAY_TYPE_INDIRECT = 2;  // Lightweight secondary bounce (no path guiding)
constexpr uint32_t RAY_TYPE_COUNT    = 3;

//------------------------------------------------------------------------------
// Mesh Data (CPU-side representation for loading)
//------------------------------------------------------------------------------
struct MeshData {
    std::vector<GpuVertex> vertices;
    std::vector<uint32_t> indices;
    uint32_t materialIndex;
};

//------------------------------------------------------------------------------
// Material Data (CPU-side representation for loading)
//------------------------------------------------------------------------------
struct MaterialData {
    // Core PBR properties
    float4 baseColor = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
    float metallic = 0.0f;
    float roughness = 1.0f;
    float3 emissive = make_float3(0.0f, 0.0f, 0.0f);

    // Core texture paths
    std::string baseColorTexPath;
    std::string normalTexPath;
    bool normalTexIsBumpMap = false;  // true = grayscale height map (OBJ), needs gradient conversion
    float bumpStrength = 1.0f;       // Strength for bump-to-normal conversion
    std::string metallicRoughnessTexPath;
    std::string emissiveTexPath;

    // KHR_materials_transmission
    float transmission = 0.0f;
    float ior = 1.5f;
    std::string transmissionTexPath;

    // KHR_materials_volume
    float3 attenuationColor = make_float3(1.0f, 1.0f, 1.0f);
    float attenuationDistance = 0.0f;  // 0 = infinite (no absorption)
    float thickness = 0.0f;

    // KHR_materials_clearcoat
    float clearcoat = 0.0f;
    float clearcoatRoughness = 0.0f;
    std::string clearcoatTexPath;
    std::string clearcoatRoughnessTexPath;
    std::string clearcoatNormalTexPath;

    // KHR_materials_sheen
    float3 sheenColor = make_float3(0.0f, 0.0f, 0.0f);
    float sheenRoughness = 0.0f;
    std::string sheenColorTexPath;
    std::string sheenRoughnessTexPath;

    // KHR_materials_specular
    float specularFactor = 1.0f;
    float3 specularColorFactor = make_float3(1.0f, 1.0f, 1.0f);
    std::string specularTexPath;
    std::string specularColorTexPath;

    // Occlusion
    std::string occlusionTexPath;
    float occlusionStrength = 1.0f;

    // Alpha settings
    uint32_t alphaMode = ALPHA_MODE_OPAQUE;
    float alphaCutoff = 0.5f;
    bool doubleSided = false;
};

//------------------------------------------------------------------------------
// Model Instance (for scene graph)
//------------------------------------------------------------------------------
struct ModelInstance {
    uint32_t meshIndex;
    float transform[12];        // 3x4 row-major transform matrix
};

//------------------------------------------------------------------------------
// Loaded Model (CPU-side)
//------------------------------------------------------------------------------
struct LoadedModel {
    std::vector<MeshData> meshes;
    std::vector<MaterialData> materials;
    std::vector<ModelInstance> instances;
    std::string name;
};

} // namespace spectra
