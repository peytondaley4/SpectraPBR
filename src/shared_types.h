#pragma once

#include <cuda_runtime.h>
#include <optix.h>
#include <cstdint>
#include <vector>
#include <string>

namespace spectra {

//------------------------------------------------------------------------------
// GPU Vertex Structure (48 bytes, aligned)
//------------------------------------------------------------------------------
struct GpuVertex {
    float3 position;   // 12 bytes
    float3 normal;     // 12 bytes
    float4 tangent;    // 16 bytes (w = bitangent sign: +1 or -1)
    float2 uv;         // 8 bytes
};
static_assert(sizeof(GpuVertex) == 48, "GpuVertex must be 48 bytes");

//------------------------------------------------------------------------------
// GPU Material Structure (for SBT hit group records)
//------------------------------------------------------------------------------
struct GpuMaterial {
    float4 baseColor;           // Base color factor (RGBA)
    float metallic;             // Metallic factor [0, 1]
    float roughness;            // Roughness factor [0, 1]
    float2 _pad0;               // Padding
    float3 emissive;            // Emissive color (RGB)
    float _pad1;                // Padding

    // Texture objects (0 = no texture)
    cudaTextureObject_t baseColorTex;
    cudaTextureObject_t normalTex;
    cudaTextureObject_t metallicRoughnessTex;
    cudaTextureObject_t emissiveTex;

    // Alpha settings
    uint32_t alphaMode;         // 0 = OPAQUE, 1 = MASK, 2 = BLEND
    float alphaCutoff;          // Cutoff for MASK mode
    float2 _pad2;               // Padding
};

// Alpha mode constants
constexpr uint32_t ALPHA_MODE_OPAQUE = 0;
constexpr uint32_t ALPHA_MODE_MASK   = 1;
constexpr uint32_t ALPHA_MODE_BLEND  = 2;

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
// Launch Parameters (passed to all OptiX programs)
//------------------------------------------------------------------------------
struct LaunchParams {
    // Output buffer
    float4* output_buffer;
    uint32_t width;
    uint32_t height;
    uint32_t frame_index;

    // Camera
    CameraParams camera;

    // Scene traversable
    OptixTraversableHandle scene_handle;

    // Geometry buffer arrays (indexed by instance ID)
    CUdeviceptr* vertex_buffers;    // Array of pointers to GpuVertex arrays
    CUdeviceptr* index_buffers;     // Array of pointers to uint32_t index arrays

    // Material indices per instance (maps instance ID -> material SBT index)
    uint32_t* instance_material_indices;
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
constexpr uint32_t RAY_TYPE_COUNT    = 2;

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
    float4 baseColor = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
    float metallic = 0.0f;
    float roughness = 1.0f;
    float3 emissive = make_float3(0.0f, 0.0f, 0.0f);

    std::string baseColorTexPath;
    std::string normalTexPath;
    std::string metallicRoughnessTexPath;
    std::string emissiveTexPath;

    uint32_t alphaMode = ALPHA_MODE_OPAQUE;
    float alphaCutoff = 0.5f;
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
