#pragma once

//------------------------------------------------------------------------------
// GPU Types Header for OptiX Programs
//
// This file contains all GPU-side structures shared between OptiX .cu files.
// It uses pure C-compatible syntax (no namespaces, no STL) for CUDA compatibility.
//
// These types must stay in sync with src/shared_types.h (CPU-side definitions).
//------------------------------------------------------------------------------

#include <cuda_runtime.h>
#include <optix.h>

//------------------------------------------------------------------------------
// Constants
//------------------------------------------------------------------------------

// Ray types
constexpr unsigned int RAY_TYPE_RADIANCE = 0;
constexpr unsigned int RAY_TYPE_SHADOW   = 1;
constexpr unsigned int RAY_TYPE_COUNT    = 2;

// Alpha modes
constexpr unsigned int ALPHA_MODE_OPAQUE = 0;
constexpr unsigned int ALPHA_MODE_MASK   = 1;
constexpr unsigned int ALPHA_MODE_BLEND  = 2;

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
// Launch Parameters (passed to all OptiX programs)
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

    // Material indices per instance (maps instance ID -> material SBT index)
    unsigned int* instance_material_indices;

    // Lighting
    GpuPointLight* point_lights;
    unsigned int point_light_count;
    unsigned int _pad_lights0;
    GpuDirectionalLight* directional_lights;
    unsigned int directional_light_count;
    unsigned int _pad_lights1;
    GpuAreaLight* area_lights;
    unsigned int area_light_count;
    unsigned int _pad_lights2;

    // Environment map (equirectangular HDR)
    cudaTextureObject_t environment_map;  // 0 = none
    float environment_intensity;
    float _pad_env;

    // Quality and rendering settings
    unsigned int quality_mode;
    unsigned int random_seed;           // Per-frame random seed for sampling

    // UI selection (UINT32_MAX = no selection)
    unsigned int selected_instance_id;
    unsigned int _pad_selection;

    // Picking mode
    unsigned int* pick_result;      // Device buffer to store picked instance ID (1 element)
    unsigned int pick_x;            // Pick pixel X coordinate
    unsigned int pick_y;            // Pick pixel Y coordinate
    unsigned int pick_mode;         // 0 = normal render, 1 = pick mode (single ray)
};

//------------------------------------------------------------------------------
// SBT Record Data Structures
//------------------------------------------------------------------------------

// Hit group data follows the SBT record header
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
