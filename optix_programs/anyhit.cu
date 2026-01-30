#include <optix.h>
#include "gpu_types.h"
#include "shared_device.h"

//------------------------------------------------------------------------------
// Anyhit Program for Alpha Mask Testing
//
// This program runs for every potential intersection and decides whether to
// accept or ignore it based on the alpha value. For MASK mode materials,
// we sample the base color texture and compare alpha against the cutoff.
//------------------------------------------------------------------------------

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
