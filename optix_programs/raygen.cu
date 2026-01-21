#include <optix.h>
#include "shared_device.h"

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

    // Geometry buffers (unused in raygen but needed for struct layout)
    CUdeviceptr* vertex_buffers;
    CUdeviceptr* index_buffers;

    unsigned int* instance_material_indices;
} params;
}

extern "C" __global__ void __raygen__simple() {
    // Get the launch index (pixel coordinates)
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();

    // Calculate normalized device coordinates [-1, 1]
    // Add 0.5 for pixel center
    const float u = (static_cast<float>(idx.x) + 0.5f) / static_cast<float>(dim.x);
    const float v = (static_cast<float>(idx.y) + 0.5f) / static_cast<float>(dim.y);

    // Convert to [-1, 1] range
    const float ndcX = 2.0f * u - 1.0f;
    const float ndcY = 2.0f * v - 1.0f;  // Y up

    // Calculate ray direction using camera parameters
    const float tanHalfFovY = tanf(params.camera.fovY * 0.5f);
    const float tanHalfFovX = tanHalfFovY * params.camera.aspectRatio;

    // Ray direction in camera space, then transform to world space
    float3 rayDir = params.camera.forward
                  + params.camera.right * (ndcX * tanHalfFovX)
                  + params.camera.up * (ndcY * tanHalfFovY);
    rayDir = normalize(rayDir);

    // Initialize payload
    unsigned int p0, p1, p2, p3;
    p0 = __float_as_uint(0.0f);  // color.x
    p1 = __float_as_uint(0.0f);  // color.y
    p2 = __float_as_uint(0.0f);  // color.z
    p3 = __float_as_uint(-1.0f); // hitDistance (-1 = miss)

    // Trace ray if we have a valid scene
    if (params.scene_handle != 0) {
        optixTrace(
            params.scene_handle,
            params.camera.position,
            rayDir,
            params.camera.nearPlane,  // tmin
            params.camera.farPlane,   // tmax
            0.0f,                      // rayTime
            0xFF,                      // visibilityMask
            OPTIX_RAY_FLAG_NONE,
            0,                         // SBT offset (ray type 0 = radiance)
            1,                         // SBT stride (1 ray type)
            0,                         // missSBTIndex
            p0, p1, p2, p3             // payload
        );
    } else {
        // No scene - render gradient for debugging
        p0 = __float_as_uint(u);
        p1 = __float_as_uint(v);
        p2 = __float_as_uint(0.2f);
    }

    // Retrieve color from payload
    float3 color = make_float3(
        __uint_as_float(p0),
        __uint_as_float(p1),
        __uint_as_float(p2)
    );

    // Write to output buffer
    const unsigned int linear_idx = idx.y * params.width + idx.x;
    params.output_buffer[linear_idx] = make_float4(color.x, color.y, color.z, 1.0f);
}
