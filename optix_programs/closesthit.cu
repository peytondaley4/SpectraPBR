#include <optix.h>
#include "gpu_types.h"
#include "shared_device.h"

//------------------------------------------------------------------------------
// Closest-Hit Programs
//
// The path tracer is iterative: raygen drives the loop and does ALL shading.
// Closest-hit only reports where the ray hit (t, instance, primitive,
// barycentrics) through the payload. This keeps the OptiX trace depth at 2
// (radiance + shadow), minimizes the continuation stack, and leaves shading
// in one place instead of duplicated primary/bounce shaders.
//------------------------------------------------------------------------------

extern "C" __global__ void __closesthit__hitinfo() {
    const float2 barycentrics = optixGetTriangleBarycentrics();
    setPayloadHitInfo(
        optixGetRayTmax(),
        optixGetInstanceId(),
        optixGetPrimitiveIndex(),
        barycentrics.x,
        barycentrics.y);
}

// Shadow rays trace with OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT, so this never
// runs in practice — it exists so the shadow hit group has a valid CH that
// marks occlusion if the flag is ever dropped.
extern "C" __global__ void __closesthit__shadow() {
    optixSetPayload_0(1);
}
