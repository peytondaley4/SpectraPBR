#include <optix.h>

// Launch parameters - must match LaunchParams struct in optix_engine.h
extern "C" {
__constant__ struct {
    float4* output_buffer;
    unsigned int width;
    unsigned int height;
    unsigned int frame_index;
} params;
}

extern "C" __global__ void __raygen__simple() {
    // Get the launch index (pixel coordinates)
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();

    // Calculate normalized coordinates [0, 1]
    const float u = static_cast<float>(idx.x) / static_cast<float>(dim.x);
    const float v = static_cast<float>(idx.y) / static_cast<float>(dim.y);

    // Generate a simple gradient:
    // Red increases left to right (u)
    // Green increases bottom to top (v)
    // Blue is constant at 0.2
    // This matches the spec: "color = (x/width, y/height, 0.5, 1.0)"
    // but with blue at 0.2 for better visual distinction
    float4 color = make_float4(u, v, 0.2f, 1.0f);

    // Write to output buffer
    // Linear index: y * width + x
    const unsigned int linear_idx = idx.y * params.width + idx.x;
    params.output_buffer[linear_idx] = color;
}
