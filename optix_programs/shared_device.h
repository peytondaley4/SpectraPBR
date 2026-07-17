#pragma once

#include <optix.h>
#include <cuda_runtime.h>

//------------------------------------------------------------------------------
// Radiance Ray Payload (5 registers)
//
// The path tracer is iterative: closest-hit only reports WHERE the ray hit,
// and raygen does all shading. Payload layout:
//   p0 = hit distance t (float bits, negative = miss)
//   p1 = instance ID
//   p2 = primitive index
//   p3 = barycentric u (float bits)
//   p4 = barycentric v (float bits)
//
// Shadow rays use two registers:
//   p0 = occluded (init 1, miss writes 0)
//   p1 = isDelta (1 = point/directional target; __anyhit__shadow_alpha reads
//        it to let transmissive surfaces pass delta-light shadow rays)
//------------------------------------------------------------------------------

struct HitInfo {
    float t;                 // < 0 means miss
    unsigned int instanceId;
    unsigned int primIdx;
    float baryU;
    float baryV;
};

__forceinline__ __device__ void setPayloadHitInfo(
    float t, unsigned int instanceId, unsigned int primIdx, float baryU, float baryV)
{
    optixSetPayload_0(__float_as_uint(t));
    optixSetPayload_1(instanceId);
    optixSetPayload_2(primIdx);
    optixSetPayload_3(__float_as_uint(baryU));
    optixSetPayload_4(__float_as_uint(baryV));
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
// Instance transform helpers (3x4 row-major matrices stored as 12 floats)
//------------------------------------------------------------------------------
__forceinline__ __device__ float3 transformPoint(const float* m, const float3& p) {
    return make_float3(
        m[0] * p.x + m[1] * p.y + m[2]  * p.z + m[3],
        m[4] * p.x + m[5] * p.y + m[6]  * p.z + m[7],
        m[8] * p.x + m[9] * p.y + m[10] * p.z + m[11]
    );
}

// Linear part only (directions / tangents)
__forceinline__ __device__ float3 transformVector(const float* m, const float3& v) {
    return make_float3(
        m[0] * v.x + m[1] * v.y + m[2]  * v.z,
        m[4] * v.x + m[5] * v.y + m[6]  * v.z,
        m[8] * v.x + m[9] * v.y + m[10] * v.z
    );
}

//------------------------------------------------------------------------------
// Scalar helpers
//------------------------------------------------------------------------------
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

// Rec. 709 luminance — the single shared definition used by light selection,
// guide training, and RR. Host code (LightManager::syncToGpu) must match.
__forceinline__ __device__ float luminance3(const float3& c) {
    return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;
}

//------------------------------------------------------------------------------
// Random Number Generation (PCG Hash with improved mixing)
//------------------------------------------------------------------------------

__forceinline__ __device__ unsigned int pcgHash(unsigned int input) {
    unsigned int state = input * 747796405u + 2891336453u;
    unsigned int word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

__forceinline__ __device__ unsigned int wangHash(unsigned int seed) {
    seed = (seed ^ 61u) ^ (seed >> 16u);
    seed *= 9u;
    seed = seed ^ (seed >> 4u);
    seed *= 0x27d4eb2du;
    seed = seed ^ (seed >> 15u);
    return seed;
}

__forceinline__ __device__ unsigned int mixSeed(unsigned int x, unsigned int y, unsigned int frame, unsigned int sample) {
    unsigned int seed = x;
    seed = wangHash(seed + y * 0x9e3779b9u);
    seed = wangHash(seed + frame * 0x85ebca6bu);
    seed = wangHash(seed + sample * 0xc2b2ae35u);
    return seed | 1u;
}

__forceinline__ __device__ float hashToFloat(unsigned int hash) {
    return __uint_as_float((hash >> 9u) | 0x3f800000u) - 1.0f;
}

__forceinline__ __device__ float randomFloat(unsigned int& seed) {
    seed = pcgHash(seed);
    return hashToFloat(seed);
}

__forceinline__ __device__ float2 randomFloat2(unsigned int& seed) {
    seed = pcgHash(seed);
    float u1 = hashToFloat(seed);
    seed = pcgHash(seed);
    float u2 = hashToFloat(seed);
    return make_float2(u1, u2);
}

// R2 low-discrepancy sequence (Generalized Golden Ratio, Roberts 2018).
// Computed in 32-bit fixed point: n * frac(a) mod 2^32 wraps exactly, so the
// sequence never degrades for large n (float fmodf loses low bits past ~10^5
// frames, which matters for long accumulation runs).
//   a1 = 1/g  = 0.7548776662  -> round(a1 * 2^32) = 3242174889
//   a2 = 1/g^2 = 0.5698402910 -> round(a2 * 2^32) = 2447445414
// where g = 1.32471795724 (plastic constant, real root of x^3 = x + 1).
__forceinline__ __device__ float2 r2Sequence(unsigned int n) {
    unsigned int u = n * 3242174889u;
    unsigned int v = n * 2447445414u;
    return make_float2(hashToFloat(u), hashToFloat(v));
}
