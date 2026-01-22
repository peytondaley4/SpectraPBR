#include <optix.h>
#include "shared_device.h"
#include "brdf.h"

//------------------------------------------------------------------------------
// Phase 3: Miss Programs
//
// - Background miss: Sky gradient or environment map sampling
// - Shadow miss: Returns visibility (not occluded)
//------------------------------------------------------------------------------

// Light structures (for launch params)
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

// MissRecord data following the header
struct MissData {
    float3 backgroundColor;
    float _pad;
};

// Launch parameters
extern "C" {
__constant__ struct {
    float4* output_buffer;
    float4* accumulation_buffer;
    unsigned int width;
    unsigned int height;
    unsigned int frame_index;
    unsigned int accumulated_frames;

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

    OptixTraversableHandle scene_handle;

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
        color = make_float3(0.0f, 0.0f, 0.0f);
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
