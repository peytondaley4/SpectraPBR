#include <optix.h>
#include "shared_device.h"

// MissRecord data following the header
struct MissData {
    float3 backgroundColor;
    float _pad;
};

extern "C" __global__ void __miss__background() {
    // Get SBT data for background color
    const MissData* sbtData = reinterpret_cast<MissData*>(optixGetSbtDataPointer());

    // Get ray direction for sky gradient
    const float3 rayDir = optixGetWorldRayDirection();

    // Simple sky gradient based on ray Y direction
    // Blend from horizon color to zenith color
    const float t = 0.5f * (rayDir.y + 1.0f);  // Map [-1,1] to [0,1]

    // Sky colors
    const float3 horizonColor = make_float3(0.7f, 0.8f, 0.9f);  // Light blue-gray
    const float3 zenithColor = make_float3(0.3f, 0.5f, 0.8f);   // Deeper blue

    // Interpolate between horizon and zenith
    float3 skyColor = make_float3(
        (1.0f - t) * horizonColor.x + t * zenithColor.x,
        (1.0f - t) * horizonColor.y + t * zenithColor.y,
        (1.0f - t) * horizonColor.z + t * zenithColor.z
    );

    // Optionally use background color from SBT if set
    if (sbtData && (sbtData->backgroundColor.x > 0.0f ||
                    sbtData->backgroundColor.y > 0.0f ||
                    sbtData->backgroundColor.z > 0.0f)) {
        skyColor = sbtData->backgroundColor;
    }

    // Set payload
    setPayloadColor(skyColor);
    setPayloadHitDistance(-1.0f);  // Negative distance indicates miss
}

extern "C" __global__ void __miss__shadow() {
    // Shadow ray miss - no occlusion
    setPayloadHitDistance(-1.0f);
}
