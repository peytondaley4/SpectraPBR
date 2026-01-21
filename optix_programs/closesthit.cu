#include <optix.h>
#include "shared_device.h"

// Vertex structure matching GpuVertex from shared_types.h
struct GpuVertex {
    float3 position;
    float3 normal;
    float4 tangent;
    float2 uv;
};

// Material structure matching GpuMaterial from shared_types.h
struct GpuMaterial {
    float4 baseColor;
    float metallic;
    float roughness;
    float2 _pad0;
    float3 emissive;
    float _pad1;

    cudaTextureObject_t baseColorTex;
    cudaTextureObject_t normalTex;
    cudaTextureObject_t metallicRoughnessTex;
    cudaTextureObject_t emissiveTex;

    unsigned int alphaMode;
    float alphaCutoff;
    float2 _pad2;
};

// HitGroupRecord data following the header
struct HitGroupData {
    GpuMaterial material;
    unsigned int geometryIndex;
};

// Launch parameters - must match LaunchParams in shared_types.h
extern "C" {
__constant__ struct {
    float4* output_buffer;
    unsigned int width;
    unsigned int height;
    unsigned int frame_index;

    // Camera
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

    // Scene traversable
    OptixTraversableHandle scene_handle;

    // Geometry buffers
    CUdeviceptr* vertex_buffers;
    CUdeviceptr* index_buffers;

    unsigned int* instance_material_indices;
} params;
}

extern "C" __global__ void __closesthit__radiance() {
    // Get SBT data
    const HitGroupData* sbtData = reinterpret_cast<HitGroupData*>(optixGetSbtDataPointer());
    const GpuMaterial& material = sbtData->material;
    const unsigned int geomIdx = sbtData->geometryIndex;

    // Get instance ID for buffer lookup
    const unsigned int instanceId = optixGetInstanceId();

    // Get primitive index and barycentrics
    const unsigned int primIdx = optixGetPrimitiveIndex();
    const float2 barycentrics = optixGetTriangleBarycentrics();
    const float u = barycentrics.x;
    const float v = barycentrics.y;

    // Get vertex/index buffers for this geometry
    const GpuVertex* vertices = reinterpret_cast<const GpuVertex*>(params.vertex_buffers[instanceId]);
    const unsigned int* indices = reinterpret_cast<const unsigned int*>(params.index_buffers[instanceId]);

    // Get triangle vertices
    const unsigned int i0 = indices[primIdx * 3 + 0];
    const unsigned int i1 = indices[primIdx * 3 + 1];
    const unsigned int i2 = indices[primIdx * 3 + 2];

    const GpuVertex& v0 = vertices[i0];
    const GpuVertex& v1 = vertices[i1];
    const GpuVertex& v2 = vertices[i2];

    // Interpolate attributes
    float3 worldNormal = interpolate(v0.normal, v1.normal, v2.normal, u, v);
    worldNormal = normalize(worldNormal);

    float2 texCoord = interpolate(v0.uv, v1.uv, v2.uv, u, v);

    // Transform normal to world space if instance has transform
    // For now, assume normals are already in world space

    // Sample base color texture or use factor
    float4 baseColor = material.baseColor;
    if (material.baseColorTex != 0) {
        float4 texColor = tex2D<float4>(material.baseColorTex, texCoord.x, texCoord.y);
        baseColor = make_float4(
            baseColor.x * texColor.x,
            baseColor.y * texColor.y,
            baseColor.z * texColor.z,
            baseColor.w * texColor.w
        );
    }

    // Simple normal visualization for Phase 2
    // Convert normal from [-1,1] to [0,1] for visualization
    float3 color = make_float3(
        worldNormal.x * 0.5f + 0.5f,
        worldNormal.y * 0.5f + 0.5f,
        worldNormal.z * 0.5f + 0.5f
    );

    // Apply base color tint
    color = color * make_float3(baseColor.x, baseColor.y, baseColor.z);

    // Simple directional light for visibility
    float3 lightDir = normalize(make_float3(1.0f, 1.0f, 1.0f));
    float NdotL = fmaxf(0.0f, dot(worldNormal, lightDir));
    float ambient = 0.2f;
    float lighting = ambient + (1.0f - ambient) * NdotL;

    color = make_float3(baseColor.x, baseColor.y, baseColor.z) * lighting;

    // Set payload
    setPayloadColor(color);
    setPayloadHitDistance(optixGetRayTmax());
}

extern "C" __global__ void __closesthit__shadow() {
    // Shadow ray hit - set distance to indicate hit
    setPayloadHitDistance(optixGetRayTmax());
}
