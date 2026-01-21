#include <optix.h>

// Miss program - called when a ray doesn't hit any geometry
// In Phase 1, this won't be called since we're not actually tracing rays
// It's included for pipeline completeness and will be used in Phase 2+

extern "C" __global__ void __miss__background() {
    // For Phase 1, this function does nothing
    // In Phase 2+, this would set the ray payload to a background color
    // Example for future use:
    // optixSetPayload_0(__float_as_uint(0.1f));  // R
    // optixSetPayload_1(__float_as_uint(0.1f));  // G
    // optixSetPayload_2(__float_as_uint(0.2f));  // B
}
