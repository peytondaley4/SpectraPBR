#include <optix.h>
#include "gpu_types.h"
#include "shared_device.h"

extern "C" __global__ void __raygen__simple() {
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();

    unsigned int pixelX = params.pick_mode ? params.pick_x : idx.x;
    unsigned int pixelY = params.pick_mode ? params.pick_y : idx.y;
    const unsigned int linear_idx = pixelY * params.width + pixelX;

    const float tanHalfFovY = tanf(params.camera.fovY * 0.5f);
    const float tanHalfFovX = tanHalfFovY * params.camera.aspectRatio;

    const unsigned int spp = max(1u, params.samples_per_pixel);

    float3 accumulatedColor = make_float3(0.0f, 0.0f, 0.0f);
    unsigned int lastInstanceId = 0xFFFFFFFFu;

    unsigned int gridDim = (unsigned int)ceilf(sqrtf((float)spp));

    for (unsigned int sampleIdx = 0; sampleIdx < spp; ++sampleIdx) {
        unsigned int seed = mixSeed(pixelX, pixelY, params.frame_index, sampleIdx);

        float jitterX, jitterY;
        if (params.pick_mode) {
            jitterX = 0.0f;
            jitterY = 0.0f;
        } else if (spp > 1) {
            unsigned int stratumX = sampleIdx % gridDim;
            unsigned int stratumY = sampleIdx / gridDim;
            float cellSize = 1.0f / (float)gridDim;
            jitterX = (stratumX + randomFloat(seed)) * cellSize - 0.5f;
            jitterY = (stratumY + randomFloat(seed)) * cellSize - 0.5f;
        } else {
            jitterX = randomFloat(seed) - 0.5f;
            jitterY = randomFloat(seed) - 0.5f;
        }

        const float u = (static_cast<float>(pixelX) + 0.5f + jitterX) / static_cast<float>(params.width);
        const float v = (static_cast<float>(pixelY) + 0.5f + jitterY) / static_cast<float>(params.height);

        const float ndcX = 2.0f * u - 1.0f;
        const float ndcY = 1.0f - 2.0f * v;

        float3 rayDir = params.camera.forward
                      + params.camera.right * (ndcX * tanHalfFovX)
                      + params.camera.up * (ndcY * tanHalfFovY);
        rayDir = normalize(rayDir);

        unsigned int p0, p1, p2, p3, p4;
        p0 = __float_as_uint(0.0f);
        p1 = __float_as_uint(0.0f);
        p2 = __float_as_uint(0.0f);
        p3 = __float_as_uint(-1.0f);
        p4 = 0xFFFFFFFFu;

        if (params.scene_handle != 0) {
            optixTrace(
                params.scene_handle,
                params.camera.position,
                rayDir,
                params.camera.nearPlane,
                params.camera.farPlane,
                0.0f,
                0xFF,
                OPTIX_RAY_FLAG_NONE,
                RAY_TYPE_RADIANCE,
                RAY_TYPE_COUNT,
                RAY_TYPE_RADIANCE,
                p0, p1, p2, p3, p4
            );
        } else {
            p0 = __float_as_uint(u);
            p1 = __float_as_uint(v);
            p2 = __float_as_uint(0.2f);
        }

        accumulatedColor.x += __uint_as_float(p0);
        accumulatedColor.y += __uint_as_float(p1);
        accumulatedColor.z += __uint_as_float(p2);
        lastInstanceId = p4;

        if (params.pick_mode) break;
    }

    if (params.pick_mode && params.pick_result != nullptr) {
        *params.pick_result = lastInstanceId;
        return;
    }

    float invSpp = 1.0f / static_cast<float>(spp);
    float3 newColor = make_float3(
        accumulatedColor.x * invSpp,
        accumulatedColor.y * invSpp,
        accumulatedColor.z * invSpp
    );

    if (params.accumulated_frames > 0 && params.accumulation_buffer != nullptr) {
        float4 accumulated = params.accumulation_buffer[linear_idx];
        float n = (float)(params.accumulated_frames + 1);

        float3 blended = make_float3(
            accumulated.x + (newColor.x - accumulated.x) / n,
            accumulated.y + (newColor.y - accumulated.y) / n,
            accumulated.z + (newColor.z - accumulated.z) / n
        );

        params.accumulation_buffer[linear_idx] = make_float4(blended.x, blended.y, blended.z, 1.0f);
        params.output_buffer[linear_idx] = make_float4(blended.x, blended.y, blended.z, 1.0f);
    } else {
        if (params.accumulation_buffer != nullptr) {
            params.accumulation_buffer[linear_idx] = make_float4(newColor.x, newColor.y, newColor.z, 1.0f);
        }
        params.output_buffer[linear_idx] = make_float4(newColor.x, newColor.y, newColor.z, 1.0f);
    }
}
