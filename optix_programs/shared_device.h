#pragma once

#include <optix.h>
#include <cuda_runtime.h>

//------------------------------------------------------------------------------
// Ray Payload Structure
// Using 4 payload slots for color (RGB) and hit distance
//------------------------------------------------------------------------------
struct RayPayload {
    float3 color;
    float hitDistance;
};

//------------------------------------------------------------------------------
// Payload Packing/Unpacking
// OptiX uses 32-bit payload registers, so we pack/unpack floats
//------------------------------------------------------------------------------
__forceinline__ __device__ void setPayloadColor(const float3& color) {
    optixSetPayload_0(__float_as_uint(color.x));
    optixSetPayload_1(__float_as_uint(color.y));
    optixSetPayload_2(__float_as_uint(color.z));
}

__forceinline__ __device__ void setPayloadHitDistance(float dist) {
    optixSetPayload_3(__float_as_uint(dist));
}

__forceinline__ __device__ void setPayloadInstanceId(unsigned int instanceId) {
    optixSetPayload_4(instanceId);
}

__forceinline__ __device__ float3 getPayloadColor() {
    return make_float3(
        __uint_as_float(optixGetPayload_0()),
        __uint_as_float(optixGetPayload_1()),
        __uint_as_float(optixGetPayload_2())
    );
}

__forceinline__ __device__ float getPayloadHitDistance() {
    return __uint_as_float(optixGetPayload_3());
}

//------------------------------------------------------------------------------
// Barycentric Interpolation Helpers
//------------------------------------------------------------------------------
__forceinline__ __device__ float3 interpolate(
    const float3& a, const float3& b, const float3& c,
    float u, float v)
{
    float w = 1.0f - u - v;
    return make_float3(
        w * a.x + u * b.x + v * c.x,
        w * a.y + u * b.y + v * c.y,
        w * a.z + u * b.z + v * c.z
    );
}

__forceinline__ __device__ float4 interpolate(
    const float4& a, const float4& b, const float4& c,
    float u, float v)
{
    float w = 1.0f - u - v;
    return make_float4(
        w * a.x + u * b.x + v * c.x,
        w * a.y + u * b.y + v * c.y,
        w * a.z + u * b.z + v * c.z,
        w * a.w + u * b.w + v * c.w
    );
}

__forceinline__ __device__ float2 interpolate(
    const float2& a, const float2& b, const float2& c,
    float u, float v)
{
    float w = 1.0f - u - v;
    return make_float2(
        w * a.x + u * b.x + v * c.x,
        w * a.y + u * b.y + v * c.y
    );
}

//------------------------------------------------------------------------------
// Vector Math Helpers
//------------------------------------------------------------------------------
__forceinline__ __device__ float3 normalize(const float3& v) {
    float invLen = rsqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
    return make_float3(v.x * invLen, v.y * invLen, v.z * invLen);
}

__forceinline__ __device__ float dot(const float3& a, const float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__forceinline__ __device__ float3 cross(const float3& a, const float3& b) {
    return make_float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

__forceinline__ __device__ float length(const float3& v) {
    return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}

__forceinline__ __device__ float3 operator+(const float3& a, const float3& b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__forceinline__ __device__ float3 operator-(const float3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__forceinline__ __device__ float3 operator*(const float3& a, float s) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}

__forceinline__ __device__ float3 operator*(float s, const float3& a) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}

__forceinline__ __device__ float3 operator*(const float3& a, const float3& b) {
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

__forceinline__ __device__ float3 operator/(const float3& a, float s) {
    float invS = 1.0f / s;
    return make_float3(a.x * invS, a.y * invS, a.z * invS);
}

__forceinline__ __device__ float3 operator/(const float3& a, const float3& b) {
    return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}

__forceinline__ __device__ float3 operator-(const float3& v) {
    return make_float3(-v.x, -v.y, -v.z);
}

// float4 operators
__forceinline__ __device__ float4 operator*(float s, const float4& a) {
    return make_float4(a.x * s, a.y * s, a.z * s, a.w * s);
}

__forceinline__ __device__ float4 operator*(const float4& a, float s) {
    return make_float4(a.x * s, a.y * s, a.z * s, a.w * s);
}

__forceinline__ __device__ float4 operator+(const float4& a, const float4& b) {
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

__forceinline__ __device__ float4 operator-(const float4& a, const float4& b) {
    return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}

//------------------------------------------------------------------------------
// Transform helpers
//------------------------------------------------------------------------------
__forceinline__ __device__ float3 transformPoint(const float* matrix, const float3& p) {
    // matrix is 3x4 row-major
    return make_float3(
        matrix[0] * p.x + matrix[1] * p.y + matrix[2] * p.z + matrix[3],
        matrix[4] * p.x + matrix[5] * p.y + matrix[6] * p.z + matrix[7],
        matrix[8] * p.x + matrix[9] * p.y + matrix[10] * p.z + matrix[11]
    );
}

__forceinline__ __device__ float3 transformNormal(const float* matrix, const float3& n) {
    // For normals, we need the inverse transpose
    // For orthonormal rotation matrices, this is just the rotation part
    return normalize(make_float3(
        matrix[0] * n.x + matrix[4] * n.y + matrix[8] * n.z,
        matrix[1] * n.x + matrix[5] * n.y + matrix[9] * n.z,
        matrix[2] * n.x + matrix[6] * n.y + matrix[10] * n.z
    ));
}

//------------------------------------------------------------------------------
// Color helpers
//------------------------------------------------------------------------------
__forceinline__ __device__ float3 srgbToLinear(const float3& srgb) {
    // Simplified sRGB to linear conversion
    return make_float3(
        powf(srgb.x, 2.2f),
        powf(srgb.y, 2.2f),
        powf(srgb.z, 2.2f)
    );
}

__forceinline__ __device__ float3 linearToSrgb(const float3& linear) {
    // Simplified linear to sRGB conversion
    return make_float3(
        powf(linear.x, 1.0f / 2.2f),
        powf(linear.y, 1.0f / 2.2f),
        powf(linear.z, 1.0f / 2.2f)
    );
}

__forceinline__ __device__ float clamp(float x, float lo, float hi) {
    return fminf(hi, fmaxf(lo, x));
}

__forceinline__ __device__ float3 clamp(const float3& v, float lo, float hi) {
    return make_float3(
        clamp(v.x, lo, hi),
        clamp(v.y, lo, hi),
        clamp(v.z, lo, hi)
    );
}

//------------------------------------------------------------------------------
// Random Number Generation (PCG Hash)
//------------------------------------------------------------------------------

// Simple PCG hash for generating random numbers
__forceinline__ __device__ unsigned int pcgHash(unsigned int input) {
    unsigned int state = input * 747796405u + 2891336453u;
    unsigned int word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// Convert hash to float in [0, 1)
__forceinline__ __device__ float hashToFloat(unsigned int hash) {
    return (float)(hash & 0x00FFFFFFu) / (float)0x01000000u;
}

// Generate random float in [0, 1) and advance seed
__forceinline__ __device__ float randomFloat(unsigned int& seed) {
    seed = pcgHash(seed);
    return hashToFloat(seed);
}

// Generate two random numbers from a seed
__forceinline__ __device__ float2 randomFloat2(unsigned int& seed) {
    seed = pcgHash(seed);
    float u1 = hashToFloat(seed);
    seed = pcgHash(seed);
    float u2 = hashToFloat(seed);
    return make_float2(u1, u2);
}
