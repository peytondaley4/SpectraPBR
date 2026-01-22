#include <optix.h>
#include "shared_device.h"

//------------------------------------------------------------------------------
// Anyhit Program for Alpha Mask Testing
//
// This program runs for every potential intersection and decides whether to
// accept or ignore it based on the alpha value. For MASK mode materials,
// we sample the base color texture and compare alpha against the cutoff.
//------------------------------------------------------------------------------

// Alpha mode constants
constexpr unsigned int ALPHA_MODE_OPAQUE = 0;
constexpr unsigned int ALPHA_MODE_MASK   = 1;
constexpr unsigned int ALPHA_MODE_BLEND  = 2;

// Vertex structure matching GpuVertex from shared_types.h
struct GpuVertex {
    float3 position;
    float u;
    float3 normal;
    float v;
    float4 tangent;
};

// Simplified material structure for anyhit (only need alpha testing fields)
// Must match the layout in GpuMaterial from shared_types.h
struct GpuMaterial {
    float4 baseColor;
    float metallic;
    float roughness;
    float2 _pad0;
    float3 emissive;
    float _pad1;

    // Core textures
    cudaTextureObject_t baseColorTex;
    cudaTextureObject_t normalTex;
    cudaTextureObject_t metallicRoughnessTex;
    cudaTextureObject_t emissiveTex;

    // KHR_materials_transmission
    float transmission;
    float ior;
    float2 _pad2;
    cudaTextureObject_t transmissionTex;

    // KHR_materials_volume
    float3 attenuationColor;
    float attenuationDistance;
    float thickness;
    float3 _pad3;

    // KHR_materials_clearcoat
    float clearcoat;
    float clearcoatRoughness;
    float2 _pad4;
    cudaTextureObject_t clearcoatTex;
    cudaTextureObject_t clearcoatRoughnessTex;
    cudaTextureObject_t clearcoatNormalTex;

    // KHR_materials_sheen
    float3 sheenColor;
    float sheenRoughness;
    cudaTextureObject_t sheenColorTex;
    cudaTextureObject_t sheenRoughnessTex;

    // KHR_materials_specular
    float specularFactor;
    float3 _pad5;
    float3 specularColorFactor;
    float _pad6;
    cudaTextureObject_t specularTex;
    cudaTextureObject_t specularColorTex;

    // Occlusion
    cudaTextureObject_t occlusionTex;
    float occlusionStrength;

    // Alpha settings
    unsigned int alphaMode;
    float alphaCutoff;
    unsigned int doubleSided;
};

// HitGroupRecord data
struct HitGroupData {
    GpuMaterial material;
    unsigned int geometryIndex;
};

// Launch parameters - must match LaunchParams structure
extern "C" {
__constant__ struct {
    float4* output_buffer;
    float4* accumulation_buffer;
    unsigned int width;
    unsigned int height;
    unsigned int frame_index;
    unsigned int accumulated_frames;

    struct {
        float3 position;
        float _pad0;
        float3 forward;
        float _pad1;
        float3 right;
        float _pad2;
        float3 up;
        float _pad3;
        float fovY;
        float aspectRatio;
        float nearPlane;
        float farPlane;
    } camera;

    OptixTraversableHandle scene_handle;

    CUdeviceptr* vertex_buffers;
    CUdeviceptr* index_buffers;

    unsigned int* instance_material_indices;

    // Lighting (not used in anyhit, but must match layout)
    void* point_lights;
    unsigned int point_light_count;
    unsigned int _pad_lights0;
    void* directional_lights;
    unsigned int directional_light_count;
    unsigned int _pad_lights1;
    void* area_lights;
    unsigned int area_light_count;
    unsigned int _pad_lights2;

    cudaTextureObject_t environment_map;
    float environment_intensity;
    float _pad_env;

    unsigned int quality_mode;
    unsigned int random_seed;
} params;
}

//------------------------------------------------------------------------------
// Anyhit for radiance rays - handles alpha masking
//------------------------------------------------------------------------------
extern "C" __global__ void __anyhit__alpha() {
    const HitGroupData* sbtData = reinterpret_cast<HitGroupData*>(optixGetSbtDataPointer());
    const GpuMaterial& material = sbtData->material;

    // Only process alpha mask materials
    if (material.alphaMode != ALPHA_MODE_MASK) {
        return; // Accept hit (opaque or blend)
    }

    // Get instance ID for buffer lookup
    const unsigned int instanceId = optixGetInstanceId();

    // Get primitive index and barycentrics
    const unsigned int primIdx = optixGetPrimitiveIndex();
    const float2 barycentrics = optixGetTriangleBarycentrics();
    const float u = barycentrics.x;
    const float v = barycentrics.y;

    // Get vertex/index buffers
    const GpuVertex* vertices = reinterpret_cast<const GpuVertex*>(params.vertex_buffers[instanceId]);
    const unsigned int* indices = reinterpret_cast<const unsigned int*>(params.index_buffers[instanceId]);

    // Get triangle vertices
    const unsigned int i0 = indices[primIdx * 3 + 0];
    const unsigned int i1 = indices[primIdx * 3 + 1];
    const unsigned int i2 = indices[primIdx * 3 + 2];

    const GpuVertex& v0 = vertices[i0];
    const GpuVertex& v1 = vertices[i1];
    const GpuVertex& v2 = vertices[i2];

    // Interpolate texture coordinates
    float2 uv0 = make_float2(v0.u, v0.v);
    float2 uv1 = make_float2(v1.u, v1.v);
    float2 uv2 = make_float2(v2.u, v2.v);
    float w = 1.0f - u - v;
    float2 texCoord = make_float2(
        w * uv0.x + u * uv1.x + v * uv2.x,
        w * uv0.y + u * uv1.y + v * uv2.y
    );

    // Get alpha value
    float alpha = material.baseColor.w;
    if (material.baseColorTex != 0) {
        float4 texColor = tex2D<float4>(material.baseColorTex, texCoord.x, texCoord.y);
        alpha = material.baseColor.w * texColor.w;
    }

    // Discard if below cutoff
    if (alpha < material.alphaCutoff) {
        optixIgnoreIntersection();
    }
}

//------------------------------------------------------------------------------
// Anyhit for shadow rays - also handles alpha masking
//------------------------------------------------------------------------------
extern "C" __global__ void __anyhit__shadow_alpha() {
    const HitGroupData* sbtData = reinterpret_cast<HitGroupData*>(optixGetSbtDataPointer());
    const GpuMaterial& material = sbtData->material;

    // Only process alpha mask materials
    if (material.alphaMode != ALPHA_MODE_MASK) {
        return; // Accept hit (opaque) - shadow is blocked
    }

    // Get instance ID for buffer lookup
    const unsigned int instanceId = optixGetInstanceId();

    // Get primitive index and barycentrics
    const unsigned int primIdx = optixGetPrimitiveIndex();
    const float2 barycentrics = optixGetTriangleBarycentrics();
    const float u = barycentrics.x;
    const float v = barycentrics.y;

    // Get vertex/index buffers
    const GpuVertex* vertices = reinterpret_cast<const GpuVertex*>(params.vertex_buffers[instanceId]);
    const unsigned int* indices = reinterpret_cast<const unsigned int*>(params.index_buffers[instanceId]);

    // Get triangle vertices
    const unsigned int i0 = indices[primIdx * 3 + 0];
    const unsigned int i1 = indices[primIdx * 3 + 1];
    const unsigned int i2 = indices[primIdx * 3 + 2];

    const GpuVertex& v0 = vertices[i0];
    const GpuVertex& v1 = vertices[i1];
    const GpuVertex& v2 = vertices[i2];

    // Interpolate texture coordinates
    float2 uv0 = make_float2(v0.u, v0.v);
    float2 uv1 = make_float2(v1.u, v1.v);
    float2 uv2 = make_float2(v2.u, v2.v);
    float w = 1.0f - u - v;
    float2 texCoord = make_float2(
        w * uv0.x + u * uv1.x + v * uv2.x,
        w * uv0.y + u * uv1.y + v * uv2.y
    );

    // Get alpha value
    float alpha = material.baseColor.w;
    if (material.baseColorTex != 0) {
        float4 texColor = tex2D<float4>(material.baseColorTex, texCoord.x, texCoord.y);
        alpha = material.baseColor.w * texColor.w;
    }

    // Discard if below cutoff (shadow ray passes through)
    if (alpha < material.alphaCutoff) {
        optixIgnoreIntersection();
    }
}
