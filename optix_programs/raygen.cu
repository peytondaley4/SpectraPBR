#include <optix.h>
#include "shared_device.h"

//------------------------------------------------------------------------------
// Phase 3: Ray Generation Program
//
// Generates primary camera rays and writes final color to output buffer.
// Updated to support 2 ray types (radiance + shadow) in SBT.
//------------------------------------------------------------------------------

// Ray type constants
constexpr unsigned int RAY_TYPE_RADIANCE = 0;
constexpr unsigned int RAY_TYPE_COUNT    = 2;

// Light structures (for launch params layout)
struct GpuPointLight {
    float3 position;
    float radius;
    float3 intensity;
    float _pad;
};

struct GpuDirectionalLight {
    float3 direction;
    float angularDiameter;
    float3 irradiance;
    float _pad;
};

struct GpuAreaLight {
    float3 position;
    float _pad0;
    float3 normal;
    float _pad1;
    float3 tangent;
    float _pad2;
    float3 emission;
    float area;
    float2 size;
    float2 _pad3;
};

// Simple PCG hash for jitter
__forceinline__ __device__ unsigned int pcgHash(unsigned int input) {
    unsigned int state = input * 747796405u + 2891336453u;
    unsigned int word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

__forceinline__ __device__ float randomFloat(unsigned int& seed) {
    seed = pcgHash(seed);
    return (float)(seed & 0x00FFFFFFu) / (float)0x01000000u;
}

// Launch parameters - must match LaunchParams in shared_types.h
extern "C" {
__constant__ struct {
    float4* output_buffer;
    float4* accumulation_buffer;
    unsigned int width;
    unsigned int height;
    unsigned int frame_index;
    unsigned int accumulated_frames;

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

    // Lighting
    GpuPointLight* point_lights;
    unsigned int point_light_count;
    unsigned int _pad_lights0;
    GpuDirectionalLight* directional_lights;
    unsigned int directional_light_count;
    unsigned int _pad_lights1;
    GpuAreaLight* area_lights;
    unsigned int area_light_count;
    unsigned int _pad_lights2;

    cudaTextureObject_t environment_map;
    float environment_intensity;
    float _pad_env;

    unsigned int quality_mode;
    unsigned int random_seed;

    // UI selection (UINT32_MAX = no selection)
    unsigned int selected_instance_id;
    unsigned int _pad_selection;

    // Picking mode
    unsigned int* pick_result;
    unsigned int pick_x;
    unsigned int pick_y;
    unsigned int pick_mode;
} params;
}

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
