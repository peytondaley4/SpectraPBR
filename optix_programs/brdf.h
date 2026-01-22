#pragma once

//------------------------------------------------------------------------------
// BRDF Utilities for PBR Rendering
//
// Contains implementations of:
// - Fresnel functions (Schlick, exact dielectric, conductor)
// - GGX microfacet distribution (D, G terms)
// - VNDF importance sampling
// - Helper math functions
//------------------------------------------------------------------------------

#include "shared_device.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

#define BRDF_EPSILON 1e-6f

//------------------------------------------------------------------------------
// Additional Math Helpers (not in shared_device.h)
//------------------------------------------------------------------------------

__forceinline__ __device__ float saturate(float x) {
    return clamp(x, 0.0f, 1.0f);
}

__forceinline__ __device__ float3 lerp(const float3& a, const float3& b, float t) {
    return a + (b - a) * t;
}

__forceinline__ __device__ float lerp(float a, float b, float t) {
    return a + (b - a) * t;
}

//------------------------------------------------------------------------------
// Random Number Generation (for importance sampling)
//------------------------------------------------------------------------------

// Simple PCG hash for generating random numbers
__forceinline__ __device__ uint32_t pcgHash(uint32_t input) {
    uint32_t state = input * 747796405u + 2891336453u;
    uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// Convert hash to float in [0, 1)
__forceinline__ __device__ float hashToFloat(uint32_t hash) {
    return (float)(hash & 0x00FFFFFF) / (float)0x01000000;
}

// Generate two random numbers from a seed
__forceinline__ __device__ float2 randomFloat2(uint32_t& seed) {
    seed = pcgHash(seed);
    float u1 = hashToFloat(seed);
    seed = pcgHash(seed);
    float u2 = hashToFloat(seed);
    return make_float2(u1, u2);
}

//------------------------------------------------------------------------------
// Fresnel Functions
//------------------------------------------------------------------------------

// Schlick approximation for Fresnel reflectance
// Fast and good for most dielectrics and metals
__forceinline__ __device__ float3 fresnelSchlick(float cosTheta, const float3& F0) {
    float t = 1.0f - saturate(cosTheta);
    float t2 = t * t;
    float t5 = t2 * t2 * t;
    return F0 + (make_float3(1.0f, 1.0f, 1.0f) - F0) * t5;
}

// Schlick with roughness for IBL (used in environment mapping)
__forceinline__ __device__ float3 fresnelSchlickRoughness(float cosTheta, const float3& F0, float roughness) {
    float t = 1.0f - saturate(cosTheta);
    float t2 = t * t;
    float t5 = t2 * t2 * t;
    float3 maxF0 = make_float3(
        fmaxf(1.0f - roughness, F0.x),
        fmaxf(1.0f - roughness, F0.y),
        fmaxf(1.0f - roughness, F0.z)
    );
    return F0 + (maxF0 - F0) * t5;
}

// Exact dielectric Fresnel equations
// Use for glass, water, and other transparent materials
// eta = eta_i / eta_t (ratio of indices of refraction)
__forceinline__ __device__ float fresnelDielectric(float cosThetaI, float eta) {
    cosThetaI = clamp(cosThetaI, -1.0f, 1.0f);

    // Handle entering vs exiting the medium
    bool entering = cosThetaI > 0.0f;
    if (!entering) {
        eta = 1.0f / eta;
        cosThetaI = -cosThetaI;
    }

    // Compute sin^2(theta_t) using Snell's law
    float sinThetaTSq = eta * eta * (1.0f - cosThetaI * cosThetaI);

    // Total internal reflection
    if (sinThetaTSq >= 1.0f) {
        return 1.0f;
    }

    float cosThetaT = sqrtf(1.0f - sinThetaTSq);

    // Fresnel equations for s and p polarization
    float rs = (eta * cosThetaI - cosThetaT) / (eta * cosThetaI + cosThetaT);
    float rp = (cosThetaI - eta * cosThetaT) / (cosThetaI + eta * cosThetaT);

    // Return average (unpolarized light)
    return 0.5f * (rs * rs + rp * rp);
}

// Conductor Fresnel for accurate metal reflectance
// eta = refractive index (n), k = extinction coefficient
// Reference: http://jcgt.org/published/0003/04/03/
__forceinline__ __device__ float3 fresnelConductor(float cosTheta, const float3& eta, const float3& k) {
    cosTheta = clamp(cosTheta, 0.0f, 1.0f);
    float cos2 = cosTheta * cosTheta;
    float sin2 = 1.0f - cos2;

    float3 eta2 = eta * eta;
    float3 k2 = k * k;

    float3 t0 = eta2 - k2 - make_float3(sin2, sin2, sin2);
    float3 a2b2 = make_float3(
        sqrtf(t0.x * t0.x + 4.0f * eta2.x * k2.x),
        sqrtf(t0.y * t0.y + 4.0f * eta2.y * k2.y),
        sqrtf(t0.z * t0.z + 4.0f * eta2.z * k2.z)
    );
    float3 t1 = a2b2 + make_float3(cos2, cos2, cos2);
    float3 a = make_float3(
        sqrtf(0.5f * (a2b2.x + t0.x)),
        sqrtf(0.5f * (a2b2.y + t0.y)),
        sqrtf(0.5f * (a2b2.z + t0.z))
    );
    float3 t2 = 2.0f * cosTheta * a;
    float3 rs = (t1 - t2) / (t1 + t2);

    float3 t3 = cos2 * a2b2 + make_float3(sin2 * sin2, sin2 * sin2, sin2 * sin2);
    float3 t4 = t2 * sin2;
    float3 rp = rs * (t3 - t4) / (t3 + t4);

    return 0.5f * (rs + rp);
}

// Preset complex IOR values for common metals
// Returns (eta, k) packed as float3 each
__forceinline__ __device__ void getMetalIOR(int metalType, float3& eta, float3& k) {
    switch (metalType) {
        case 0: // Gold
            eta = make_float3(0.18299f, 0.42108f, 1.37340f);
            k = make_float3(3.4242f, 2.3459f, 1.7704f);
            break;
        case 1: // Silver
            eta = make_float3(0.15943f, 0.14512f, 0.13547f);
            k = make_float3(3.9291f, 3.1900f, 2.3808f);
            break;
        case 2: // Copper
            eta = make_float3(0.27105f, 0.67693f, 1.31640f);
            k = make_float3(3.6092f, 2.6248f, 2.2921f);
            break;
        case 3: // Aluminum
            eta = make_float3(1.6574f, 0.8803f, 0.5212f);
            k = make_float3(9.2238f, 6.2694f, 4.8370f);
            break;
        default: // Iron
            eta = make_float3(2.9114f, 2.9497f, 2.5845f);
            k = make_float3(3.0893f, 2.9318f, 2.7670f);
            break;
    }
}

//------------------------------------------------------------------------------
// GGX Microfacet Distribution
//------------------------------------------------------------------------------

// GGX Normal Distribution Function (D term)
// alpha = roughness^2 (roughness squared)
__forceinline__ __device__ float D_GGX(float NdotH, float alpha) {
    float a2 = alpha * alpha;
    float d = NdotH * NdotH * (a2 - 1.0f) + 1.0f;
    return a2 / (M_PI * d * d + BRDF_EPSILON);
}

// GGX anisotropic distribution (for anisotropic materials)
__forceinline__ __device__ float D_GGX_Aniso(float NdotH, float HdotX, float HdotY, float ax, float ay) {
    float d = HdotX * HdotX / (ax * ax) + HdotY * HdotY / (ay * ay) + NdotH * NdotH;
    return 1.0f / (M_PI * ax * ay * d * d + BRDF_EPSILON);
}

// Smith G1 masking function for GGX
__forceinline__ __device__ float G1_GGX(float NdotV, float alpha) {
    float a2 = alpha * alpha;
    return 2.0f * NdotV / (NdotV + sqrtf(a2 + (1.0f - a2) * NdotV * NdotV) + BRDF_EPSILON);
}

// Smith height-correlated masking-shadowing function (G term)
// This is the "G2" term that accounts for correlation between masking and shadowing
__forceinline__ __device__ float G_SmithGGX(float NdotV, float NdotL, float alpha) {
    float a2 = alpha * alpha;
    float GGXV = NdotL * sqrtf(NdotV * NdotV * (1.0f - a2) + a2);
    float GGXL = NdotV * sqrtf(NdotL * NdotL * (1.0f - a2) + a2);
    return 0.5f / (GGXV + GGXL + BRDF_EPSILON);
}

// Lambda function for Smith G (used in some formulations)
__forceinline__ __device__ float Lambda_GGX(float NdotV, float alpha) {
    float a2 = alpha * alpha;
    float tan2 = (1.0f - NdotV * NdotV) / (NdotV * NdotV + BRDF_EPSILON);
    return (-1.0f + sqrtf(1.0f + a2 * tan2)) * 0.5f;
}

//------------------------------------------------------------------------------
// VNDF Importance Sampling
// Reference: http://jcgt.org/published/0007/04/01/
// Samples the visible normal distribution for lower variance
//------------------------------------------------------------------------------

__forceinline__ __device__ float3 sampleGGXVNDF(
    const float3& Ve,   // View direction in local (tangent) space
    float alpha,        // Roughness (already squared if using alpha = roughness^2)
    float u1, float u2) // Random numbers in [0, 1)
{
    // Transform view direction to hemisphere configuration
    float3 Vh = normalize(make_float3(alpha * Ve.x, alpha * Ve.y, Ve.z));

    // Build orthonormal basis around Vh
    float lensq = Vh.x * Vh.x + Vh.y * Vh.y;
    float3 T1 = lensq > 0.0f
        ? make_float3(-Vh.y, Vh.x, 0.0f) / sqrtf(lensq)
        : make_float3(1.0f, 0.0f, 0.0f);
    float3 T2 = cross(Vh, T1);

    // Parameterization of the projected area
    float r = sqrtf(u1);
    float phi = 2.0f * M_PI * u2;
    float t1 = r * cosf(phi);
    float t2 = r * sinf(phi);
    float s = 0.5f * (1.0f + Vh.z);
    t2 = (1.0f - s) * sqrtf(1.0f - t1 * t1) + s * t2;

    // Reprojection onto hemisphere
    float3 Nh = t1 * T1 + t2 * T2 + sqrtf(fmaxf(0.0f, 1.0f - t1 * t1 - t2 * t2)) * Vh;

    // Transform back to ellipsoid configuration
    return normalize(make_float3(alpha * Nh.x, alpha * Nh.y, fmaxf(0.0f, Nh.z)));
}

// PDF for VNDF sampling
__forceinline__ __device__ float pdfGGXVNDF(float D, float G1, float NdotV) {
    return D * G1 * NdotV / (4.0f * NdotV + BRDF_EPSILON);
}

//------------------------------------------------------------------------------
// Cosine-Weighted Hemisphere Sampling (for Lambertian)
//------------------------------------------------------------------------------

__forceinline__ __device__ float3 sampleCosineHemisphere(float u1, float u2) {
    float r = sqrtf(u1);
    float phi = 2.0f * M_PI * u2;
    return make_float3(r * cosf(phi), r * sinf(phi), sqrtf(1.0f - u1));
}

__forceinline__ __device__ float pdfCosineHemisphere(float cosTheta) {
    return cosTheta / M_PI;
}

//------------------------------------------------------------------------------
// Full BRDF Evaluation
//------------------------------------------------------------------------------

// Evaluate standard GGX BRDF (metallic-roughness workflow)
__forceinline__ __device__ float3 evaluateGGX_BRDF(
    const float3& V,        // View direction (toward camera)
    const float3& L,        // Light direction (toward light)
    const float3& N,        // Surface normal
    const float3& baseColor,// Albedo
    float metallic,
    float roughness)
{
    float3 H = normalize(V + L);

    float NdotV = fmaxf(dot(N, V), BRDF_EPSILON);
    float NdotL = fmaxf(dot(N, L), BRDF_EPSILON);
    float NdotH = fmaxf(dot(N, H), 0.0f);
    float VdotH = fmaxf(dot(V, H), 0.0f);

    // Roughness remapping (Disney/UE4 convention)
    float alpha = roughness * roughness;
    alpha = fmaxf(alpha, 0.001f); // Prevent div by zero

    // F0 (reflectance at normal incidence)
    // Dielectrics: 0.04 (approximately 4% reflectance)
    // Metals: base color IS the F0
    float3 F0 = lerp(make_float3(0.04f, 0.04f, 0.04f), baseColor, metallic);

    // Fresnel term
    float3 F = fresnelSchlick(VdotH, F0);

    // Distribution term
    float D = D_GGX(NdotH, alpha);

    // Geometry term (height-correlated Smith)
    float G = G_SmithGGX(NdotV, NdotL, alpha);

    // Specular BRDF: D * G * F (denominator is baked into G_SmithGGX)
    float3 specular = D * G * F;

    // Diffuse BRDF (energy conserving)
    // kD = (1 - F) * (1 - metallic)
    // Metals have no diffuse component
    float3 kD = (make_float3(1.0f, 1.0f, 1.0f) - F) * (1.0f - metallic);
    float3 diffuse = kD * baseColor / M_PI;

    return diffuse + specular;
}

// Evaluate BRDF with clearcoat layer
__forceinline__ __device__ float3 evaluateBRDF_Clearcoat(
    const float3& V,
    const float3& L,
    const float3& N,
    const float3& baseColor,
    float metallic,
    float roughness,
    float clearcoat,
    float clearcoatRoughness)
{
    // Base layer BRDF
    float3 baseBRDF = evaluateGGX_BRDF(V, L, N, baseColor, metallic, roughness);

    if (clearcoat <= 0.0f) {
        return baseBRDF;
    }

    // Clearcoat layer (always dielectric, IOR ~1.5)
    float3 H = normalize(V + L);
    float NdotV = fmaxf(dot(N, V), BRDF_EPSILON);
    float NdotL = fmaxf(dot(N, L), BRDF_EPSILON);
    float NdotH = fmaxf(dot(N, H), 0.0f);
    float VdotH = fmaxf(dot(V, H), 0.0f);

    float ccAlpha = clearcoatRoughness * clearcoatRoughness;
    ccAlpha = fmaxf(ccAlpha, 0.001f);

    // Clearcoat uses fixed F0 of 0.04 (IOR 1.5)
    float F_cc = 0.04f + 0.96f * powf(1.0f - VdotH, 5.0f);
    float D_cc = D_GGX(NdotH, ccAlpha);
    float G_cc = G_SmithGGX(NdotV, NdotL, ccAlpha);

    float3 clearcoatBRDF = make_float3(D_cc * G_cc * F_cc, D_cc * G_cc * F_cc, D_cc * G_cc * F_cc);

    // Blend: clearcoat on top absorbs some light from base layer
    return baseBRDF * (1.0f - clearcoat * F_cc) + clearcoatBRDF * clearcoat;
}

// Simple Lambertian diffuse (for Fast quality mode)
__forceinline__ __device__ float3 evaluateLambertian(const float3& baseColor) {
    return baseColor / M_PI;
}

//------------------------------------------------------------------------------
// Sheen BRDF (for cloth/velvet)
// Reference: https://blog.selfshadow.com/publications/s2017-shading-course/
//------------------------------------------------------------------------------

__forceinline__ __device__ float D_Charlie(float NdotH, float roughness) {
    float alpha = roughness * roughness;
    float invAlpha = 1.0f / alpha;
    float cos2h = NdotH * NdotH;
    float sin2h = 1.0f - cos2h;
    return (2.0f + invAlpha) * powf(sin2h, invAlpha * 0.5f) / (2.0f * M_PI);
}

__forceinline__ __device__ float V_Neubelt(float NdotV, float NdotL) {
    return 1.0f / (4.0f * (NdotL + NdotV - NdotL * NdotV) + BRDF_EPSILON);
}

__forceinline__ __device__ float3 evaluateSheen(
    const float3& V,
    const float3& L,
    const float3& N,
    const float3& sheenColor,
    float sheenRoughness)
{
    float3 H = normalize(V + L);
    float NdotV = fmaxf(dot(N, V), BRDF_EPSILON);
    float NdotL = fmaxf(dot(N, L), BRDF_EPSILON);
    float NdotH = fmaxf(dot(N, H), 0.0f);

    float D = D_Charlie(NdotH, sheenRoughness);
    float V_term = V_Neubelt(NdotV, NdotL);

    return sheenColor * D * V_term;
}

//------------------------------------------------------------------------------
// Refraction Utilities
//------------------------------------------------------------------------------

// Compute refraction direction using Snell's law
// Returns false if total internal reflection occurs
__forceinline__ __device__ bool refract(
    const float3& I,    // Incident direction (pointing toward surface)
    const float3& N,    // Surface normal
    float eta,          // Ratio of indices: eta_i / eta_t
    float3& T)          // Output: refracted direction
{
    float NdotI = dot(N, I);
    float k = 1.0f - eta * eta * (1.0f - NdotI * NdotI);

    if (k < 0.0f) {
        return false; // Total internal reflection
    }

    T = eta * I - (eta * NdotI + sqrtf(k)) * N;
    return true;
}

// Reflect direction around normal
__forceinline__ __device__ float3 reflect(const float3& I, const float3& N) {
    return I - 2.0f * dot(N, I) * N;
}

//------------------------------------------------------------------------------
// Environment Map Utilities
//------------------------------------------------------------------------------

// Convert direction to equirectangular UV coordinates
__forceinline__ __device__ float2 directionToEquirectangular(const float3& dir) {
    float u = atan2f(dir.z, dir.x) / (2.0f * M_PI) + 0.5f;
    float v = asinf(clamp(dir.y, -1.0f, 1.0f)) / M_PI + 0.5f;
    return make_float2(u, v);
}

// Convert equirectangular UV to direction
__forceinline__ __device__ float3 equirectangularToDirection(float u, float v) {
    float phi = (u - 0.5f) * 2.0f * M_PI;
    float theta = (v - 0.5f) * M_PI;
    float cosTheta = cosf(theta);
    return make_float3(cosTheta * cosf(phi), sinf(theta), cosTheta * sinf(phi));
}

//------------------------------------------------------------------------------
// Normal Mapping Utilities
//------------------------------------------------------------------------------

// Transform tangent-space normal to world space
__forceinline__ __device__ float3 applyNormalMap(
    const float3& tangentNormal,    // Normal from normal map (already remapped to [-1,1])
    const float3& worldNormal,      // Interpolated vertex normal
    const float3& worldTangent,     // Interpolated tangent
    float bitangentSign)            // Sign for bitangent (tangent.w)
{
    // Build TBN matrix
    float3 N = normalize(worldNormal);
    float3 T = normalize(worldTangent - N * dot(N, worldTangent)); // Gram-Schmidt orthogonalization
    float3 B = cross(N, T) * bitangentSign;

    // Transform from tangent space to world space
    return normalize(
        tangentNormal.x * T +
        tangentNormal.y * B +
        tangentNormal.z * N
    );
}

// Sample and unpack normal from normal map texture
__forceinline__ __device__ float3 unpackNormal(const float4& texSample) {
    return make_float3(
        texSample.x * 2.0f - 1.0f,
        texSample.y * 2.0f - 1.0f,
        texSample.z * 2.0f - 1.0f
    );
}

// Color space conversion functions are in shared_device.h
