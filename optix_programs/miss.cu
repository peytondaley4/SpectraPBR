#include <optix.h>
#include "gpu_types.h"
#include "shared_device.h"
#include "brdf.h"

//------------------------------------------------------------------------------
// Phase 3: Miss Programs
//
// - Background miss: Sky gradient or environment map sampling
// - Shadow miss: Returns visibility (not occluded)
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// Background Miss (Radiance Rays)
//------------------------------------------------------------------------------

extern "C" __global__ void __miss__background() {
    // Get SBT data for background color override
    const MissData* sbtData = reinterpret_cast<MissData*>(optixGetSbtDataPointer());

    // Get ray direction for sky/environment sampling
    const float3 rayDir = optixGetWorldRayDirection();

    float3 color;

    // Check if we have an environment map
    if (params.environment_map != 0) {
        // Sample environment map using equirectangular mapping
        float2 uv = directionToEquirectangular(rayDir);
        float4 envSample = tex2D<float4>(params.environment_map, uv.x, uv.y);
        color = make_float3(envSample.x, envSample.y, envSample.z) * params.environment_intensity;
    }
    // Check for explicit background color in SBT
    else if (sbtData && (sbtData->backgroundColor.x > 0.0f ||
                         sbtData->backgroundColor.y > 0.0f ||
                         sbtData->backgroundColor.z > 0.0f)) {
        color = sbtData->backgroundColor;
    }
    else {
        // Default: Black background
        color = make_float3(0.05f, 0.05f, 0.05f);
    }

    // Set payload
    setPayloadColor(color);
    setPayloadHitDistance(-1.0f);  // Negative distance indicates miss
    setPayloadInstanceId(0xFFFFFFFFu);  // No instance hit
}

//------------------------------------------------------------------------------
// Shadow Miss (Shadow Rays)
// When shadow ray misses, the light is visible (not occluded)
//------------------------------------------------------------------------------

extern "C" __global__ void __miss__shadow() {
    // Shadow ray miss - light is visible (payload = 0 means not occluded)
    optixSetPayload_0(0);
}
