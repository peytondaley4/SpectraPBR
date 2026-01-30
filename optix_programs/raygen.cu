#include <optix.h>
#include "gpu_types.h"
#include "shared_device.h"

//------------------------------------------------------------------------------
// Phase 3: Ray Generation Program
//
// Generates primary camera rays and writes final color to output buffer.
// Updated to support 2 ray types (radiance + shadow) in SBT.
//------------------------------------------------------------------------------

extern "C" __global__ void __raygen__simple() {
    // Get the launch index (pixel coordinates)
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();

    // Pick mode: use pick coordinates instead of launch index
    unsigned int pixelX = params.pick_mode ? params.pick_x : idx.x;
    unsigned int pixelY = params.pick_mode ? params.pick_y : idx.y;
    const unsigned int linear_idx = pixelY * params.width + pixelX;

    // Generate per-pixel random seed (unique per pixel and per frame)
    unsigned int seed = (pixelX * 1973u + pixelY * 9277u + params.frame_index * 26699u) | 1u;

    // Generate sub-pixel jitter for anti-aliasing (random offset within pixel)
    // No jitter in pick mode for precise selection
    float jitterX = params.pick_mode ? 0.0f : (randomFloat(seed) - 0.5f);
    float jitterY = params.pick_mode ? 0.0f : (randomFloat(seed) - 0.5f);

    // Calculate normalized device coordinates with jitter
    const float u = (static_cast<float>(pixelX) + 0.5f + jitterX) / static_cast<float>(params.width);
    const float v = (static_cast<float>(pixelY) + 0.5f + jitterY) / static_cast<float>(params.height);

    // Convert to [-1, 1] range
    // Note: Screen Y is flipped (0 at top), so we negate to get Y-up
    const float ndcX = 2.0f * u - 1.0f;
    const float ndcY = 1.0f - 2.0f * v;  // Flip Y: screen top -> +Y (up)

    // Calculate ray direction using camera parameters
    const float tanHalfFovY = tanf(params.camera.fovY * 0.5f);
    const float tanHalfFovX = tanHalfFovY * params.camera.aspectRatio;

    // Ray direction in camera space, then transform to world space
    float3 rayDir = params.camera.forward
                  + params.camera.right * (ndcX * tanHalfFovX)
                  + params.camera.up * (ndcY * tanHalfFovY);
    rayDir = normalize(rayDir);

    // Initialize payload
    unsigned int p0, p1, p2, p3, p4;
    p0 = __float_as_uint(0.0f);  // color.x
    p1 = __float_as_uint(0.0f);  // color.y
    p2 = __float_as_uint(0.0f);  // color.z
    p3 = __float_as_uint(-1.0f); // hitDistance (-1 = miss)
    p4 = 0xFFFFFFFFu;            // instanceId (UINT32_MAX = no hit)

    // Trace ray if we have a valid scene
    if (params.scene_handle != 0) {
        optixTrace(
            params.scene_handle,
            params.camera.position,
            rayDir,
            params.camera.nearPlane,      // tmin
            params.camera.farPlane,       // tmax
            0.0f,                         // rayTime
            0xFF,                         // visibilityMask
            OPTIX_RAY_FLAG_NONE,
            RAY_TYPE_RADIANCE,            // SBT offset (ray type 0 = radiance)
            RAY_TYPE_COUNT,               // SBT stride (2 ray types: radiance + shadow)
            RAY_TYPE_RADIANCE,            // missSBTIndex
            p0, p1, p2, p3, p4            // payload (5 values)
        );
    } else {
        // No scene - render gradient for debugging
        p0 = __float_as_uint(u);
        p1 = __float_as_uint(v);
        p2 = __float_as_uint(0.2f);
    }

    // In pick mode, write instance ID to pick result buffer and return
    if (params.pick_mode && params.pick_result != nullptr) {
        *params.pick_result = p4;
        return;
    }

    // Retrieve color from payload
    float3 newColor = make_float3(
        __uint_as_float(p0),
        __uint_as_float(p1),
        __uint_as_float(p2)
    );

    // Progressive accumulation for anti-aliasing
    if (params.accumulated_frames > 0 && params.accumulation_buffer != nullptr) {
        // Blend with previous accumulated samples
        float4 accumulated = params.accumulation_buffer[linear_idx];
        float weight = 1.0f / (params.accumulated_frames + 1.0f);

        float3 blended = make_float3(
            accumulated.x + (newColor.x - accumulated.x) * weight,
            accumulated.y + (newColor.y - accumulated.y) * weight,
            accumulated.z + (newColor.z - accumulated.z) * weight
        );

        // Store accumulated result
        params.accumulation_buffer[linear_idx] = make_float4(blended.x, blended.y, blended.z, 1.0f);
        params.output_buffer[linear_idx] = make_float4(blended.x, blended.y, blended.z, 1.0f);
    } else {
        // First frame or no accumulation buffer - just store new color
        if (params.accumulation_buffer != nullptr) {
            params.accumulation_buffer[linear_idx] = make_float4(newColor.x, newColor.y, newColor.z, 1.0f);
        }
        params.output_buffer[linear_idx] = make_float4(newColor.x, newColor.y, newColor.z, 1.0f);
    }
}
