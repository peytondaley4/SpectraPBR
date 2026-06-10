#include <optix.h>
#include "gpu_types.h"
#include "shared_device.h"

//------------------------------------------------------------------------------
// Miss Programs
//
// Radiance miss only flags "no hit" (t = -1). Environment radiance is
// evaluated in raygen, where the previous bounce's sampling PDF is available —
// that is what makes the env contribution MIS-weighted against environment
// NEE instead of double counted.
//------------------------------------------------------------------------------

extern "C" __global__ void __miss__radiance() {
    optixSetPayload_0(__float_as_uint(-1.0f));
    optixSetPayload_1(0xFFFFFFFFu);
}

// Shadow ray miss: light is visible (payload = 0 means not occluded)
extern "C" __global__ void __miss__shadow() {
    optixSetPayload_0(0);
}
