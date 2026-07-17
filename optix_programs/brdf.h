#pragma once

//------------------------------------------------------------------------------
// BRDF Utilities for PBR Rendering
//
// Contains implementations of:
// - Fresnel functions (Schlick, exact dielectric, conductor)
// - GGX microfacet distribution (D, G terms) + multiple-scattering compensation
// - VNDF importance sampling + diffuse/specular lobe mixture sampling
// - Environment map importance sampling (continuous, sub-texel)
// - Helper math functions
//------------------------------------------------------------------------------

#include "shared_device.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

#define BRDF_EPSILON 1e-6f

// Default dielectric IOR for the metallic-roughness workflow (F0 = 0.04).
#define DIELECTRIC_IOR 1.5f

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

// Build an orthonormal tangent frame around unit vector N.
__forceinline__ __device__ void buildOrthonormalBasis(const float3& N, float3& T, float3& B) {
    float3 up = (fabsf(N.y) < 0.999f) ? make_float3(0.0f, 1.0f, 0.0f) : make_float3(1.0f, 0.0f, 0.0f);
    T = normalize(cross(up, N));
    B = cross(N, T);
}

//------------------------------------------------------------------------------
// Fresnel Functions
//------------------------------------------------------------------------------

// Schlick approximation for Fresnel reflectance.
// Used for cheap lobe-selection weights; the BSDF itself uses exact Fresnel.
__forceinline__ __device__ float3 fresnelSchlick(float cosTheta, const float3& F0) {
    float t = 1.0f - saturate(cosTheta);
    float t2 = t * t;
    float t5 = t2 * t2 * t;
    return F0 + (make_float3(1.0f, 1.0f, 1.0f) - F0) * t5;
}

// Exact dielectric Fresnel equations.
//
// IMPORTANT CONVENTION: eta = eta_i / eta_t — the RATIO of the incident
// medium's IOR to the transmitted medium's IOR, evaluated for the side the
// ray actually starts on (cosThetaI > 0).
//   * Air -> glass (entering):  eta = 1.0 / material_ior
//   * Glass -> air (exiting):   eta = material_ior
// Passing the material IOR directly for an entering ray flips the convention
// and produces spurious total internal reflection for incidence angles above
// ~41.8 deg (F saturates to 1, diffuse dies). Use 1/ior for entering rays.
__forceinline__ __device__ float fresnelDielectric(float cosThetaI, float eta) {
    cosThetaI = clamp(cosThetaI, -1.0f, 1.0f);

    // Handle entering vs exiting the medium
    bool entering = cosThetaI > 0.0f;
    if (!entering) {
        eta = 1.0f / eta;
        cosThetaI = -cosThetaI;
    }

    // Compute sin^2(theta_t) using Snell's law: sin_t = eta * sin_i
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

// Dielectric Fresnel for a surface seen from OUTSIDE (the common BRDF case):
// the incident medium is air, so eta_i/eta_t = 1/ior.
__forceinline__ __device__ float fresnelDielectricExternal(float cosThetaI, float ior) {
    return fresnelDielectric(cosThetaI, 1.0f / ior);
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

//------------------------------------------------------------------------------
// F0 → Conductor IOR Derivation
//
// The metallic-roughness PBR workflow provides F0 (normal-incidence reflectance)
// but not the complex IOR (n, k) needed for exact conductor Fresnel. We derive
// approximate (n, k) from F0 with n=1 and solve for k from the normal-incidence
// constraint: F0 = k² / (4 + k²), giving k = 2√(F0 / (1-F0)).
//
// This exactly reproduces F0 at normal incidence and gives a physically
// plausible conductor S-curve at grazing angles. For exact metals, the spectral
// rendering extension (Phase 4) will use measured per-wavelength (n, k) tables.
//------------------------------------------------------------------------------
__forceinline__ __device__ void F0ToConductorIOR(const float3& F0, float3& eta, float3& k) {
    // With n=1: F0 = k²/(4+k²), so k² = 4·F0/(1-F0)
    float3 r = clamp(F0, 0.02f, 0.99f);
    eta = make_float3(1.0f, 1.0f, 1.0f);
    k = make_float3(
        2.0f * sqrtf(r.x / (1.0f - r.x)),
        2.0f * sqrtf(r.y / (1.0f - r.y)),
        2.0f * sqrtf(r.z / (1.0f - r.z))
    );
}

// Fresnel dispatch shared by the BSDF evaluators: exact dielectric for
// non-metals, exact conductor with F0-derived (n,k) for metals, narrow blend
// zone in between (real materials rarely have intermediate metallic).
__forceinline__ __device__ float3 fresnelDispatch(float VdotH, const float3& baseColor, float metallic) {
    if (metallic < 0.05f) {
        float Fd = fresnelDielectricExternal(VdotH, DIELECTRIC_IOR);
        return make_float3(Fd, Fd, Fd);
    }
    if (metallic > 0.95f) {
        float3 eta, k;
        F0ToConductorIOR(baseColor, eta, k);
        return fresnelConductor(VdotH, eta, k);
    }
    float Fd = fresnelDielectricExternal(VdotH, DIELECTRIC_IOR);
    float3 F_dielectric = make_float3(Fd, Fd, Fd);
    float3 eta, k;
    F0ToConductorIOR(baseColor, eta, k);
    float3 F_conductor = fresnelConductor(VdotH, eta, k);
    return lerp(F_dielectric, F_conductor, metallic);
}

//------------------------------------------------------------------------------
// GGX Microfacet Distribution
//------------------------------------------------------------------------------

// GGX Normal Distribution Function (D term)
// alpha = roughness^2 (roughness squared)
// The denominator guard must stay far below the smallest reachable pi*d*d
// (callers clamp alpha >= 0.001, so d >= a2 >= 1e-6 and pi*d*d >= 3.1e-12):
// an additive epsilon comparable to pi*d*d flattens the peak of smooth lobes,
// and since sampleGGXVNDF draws from the TRUE analytic density, any distortion
// here desyncs pdfGGXVNDF from the sampler and biases every MIS weight.
__forceinline__ __device__ float D_GGX(float NdotH, float alpha) {
    float a2 = alpha * alpha;
    float d = NdotH * NdotH * (a2 - 1.0f) + 1.0f;
    return a2 / fmaxf(M_PI * d * d, 1e-12f);
}

// Smith G1 masking function for GGX.
// Max-guard, not additive: G1 feeds pdfGGXVNDF, so the same sampler-agreement
// argument as D_GGX applies (denominator >= alpha >= 0.001 at all call sites).
__forceinline__ __device__ float G1_GGX(float NdotV, float alpha) {
    float a2 = alpha * alpha;
    return 2.0f * NdotV / fmaxf(NdotV + sqrtf(a2 + (1.0f - a2) * NdotV * NdotV), 1e-8f);
}

// Smith height-correlated masking-shadowing VISIBILITY term:
// V = G2 / (4 * NdotV * NdotL). The 1/(4 NdotV NdotL) denominator of the
// microfacet BRDF is folded in, so specular = D * V * F.
__forceinline__ __device__ float V_SmithGGX(float NdotV, float NdotL, float alpha) {
    float a2 = alpha * alpha;
    float GGXV = NdotL * sqrtf(NdotV * NdotV * (1.0f - a2) + a2);
    float GGXL = NdotV * sqrtf(NdotL * NdotL * (1.0f - a2) + a2);
    return 0.5f / (GGXV + GGXL + BRDF_EPSILON);
}

//------------------------------------------------------------------------------
// GGX Multiple-Scattering Energy Compensation
//
// Single-scattering Smith GGX loses energy at high roughness (microfacet
// inter-reflection is ignored), failing the white furnace test — rough metals
// render too dark. Practical compensation (Turquin 2019 / Fdez-Agüera 2019):
// multiply the specular lobe by  1 + F0 * (1/E_ss - 1)  where E_ss is the
// directional albedo of the single-scattering lobe at F0 = 1.
//
// E_ss is evaluated with the analytic environment-BRDF fit from Lazarov,
// "Getting More Physical in Call of Duty: Black Ops II" (SIGGRAPH 2013 course):
// the (A, B) split-sum approximation with E_ss = A + B. Max error vs the
// tabulated DFG integral is a few percent — far smaller than the energy loss
// it corrects. The factor is >= 1, approaches 1 for smooth surfaces, and is
// intentionally excluded from the sampling PDF (PDF must match the sampler,
// not the BRDF).
//------------------------------------------------------------------------------
__forceinline__ __device__ float ggxDirectionalAlbedo(float NdotV, float roughness) {
    const float4 c0 = make_float4(-1.0f, -0.0275f, -0.572f, 0.022f);
    const float4 c1 = make_float4(1.0f, 0.0425f, 1.04f, -0.04f);
    float4 r = roughness * c0 + c1;
    float a004 = fminf(r.x * r.x, exp2f(-9.28f * NdotV)) * r.x + r.y;
    float A = -1.04f * a004 + r.z;
    float B = 1.04f * a004 + r.w;
    return saturate(A + B);
}

__forceinline__ __device__ float3 multiScatterCompensation(
    const float3& F0, float NdotV, float roughness)
{
    float Ess = fmaxf(ggxDirectionalAlbedo(NdotV, roughness), 0.05f);
    float k = 1.0f / Ess - 1.0f;
    return make_float3(1.0f + F0.x * k, 1.0f + F0.y * k, 1.0f + F0.z * k);
}

//------------------------------------------------------------------------------
// VNDF Importance Sampling
// Reference: Heitz, "Sampling the GGX Distribution of Visible Normals",
// JCGT 2018, http://jcgt.org/published/0007/04/01/
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
    float sinPhi, cosPhi;
    sincosf(phi, &sinPhi, &cosPhi);
    float t1 = r * cosPhi;
    float t2 = r * sinPhi;
    float s = 0.5f * (1.0f + Vh.z);
    t2 = (1.0f - s) * sqrtf(1.0f - t1 * t1) + s * t2;

    // Reprojection onto hemisphere
    float3 Nh = t1 * T1 + t2 * T2 + sqrtf(fmaxf(0.0f, 1.0f - t1 * t1 - t2 * t2)) * Vh;

    // Transform back to ellipsoid configuration
    return normalize(make_float3(alpha * Nh.x, alpha * Nh.y, fmaxf(0.0f, Nh.z)));
}

// PDF for VNDF sampling of reflected direction L.
// Derivation: p(L) = D_v(H) / (4*VdotH) = G1*VdotH*D / (NdotV*4*VdotH) = G1*D / (4*NdotV)
// Guard by max, not addition, for the same sampler-agreement reason as D_GGX.
__forceinline__ __device__ float pdfGGXVNDF(float D, float G1, float NdotV) {
    return D * G1 / fmaxf(4.0f * NdotV, 1e-8f);
}

//------------------------------------------------------------------------------
// Cosine-Weighted Hemisphere Sampling (for the diffuse lobe)
//------------------------------------------------------------------------------

__forceinline__ __device__ float3 sampleCosineHemisphere(float u1, float u2) {
    float r = sqrtf(u1);
    float phi = 2.0f * M_PI * u2;
    float sinPhi, cosPhi;
    sincosf(phi, &sinPhi, &cosPhi);
    return make_float3(r * cosPhi, r * sinPhi, sqrtf(fmaxf(1.0f - u1, 0.0f)));
}

__forceinline__ __device__ float pdfCosineHemisphere(float cosTheta) {
    return fmaxf(cosTheta, 0.0f) / M_PI;
}

//------------------------------------------------------------------------------
// Sheen BRDF (for cloth/velvet)
// Reference: https://blog.selfshadow.com/publications/s2017-shading-course/
//------------------------------------------------------------------------------

__forceinline__ __device__ float D_Charlie(float NdotH, float roughness) {
    float alpha = fmaxf(roughness * roughness, 1e-3f);
    float invAlpha = 1.0f / alpha;
    float cos2h = NdotH * NdotH;
    float sin2h = fmaxf(1.0f - cos2h, 1e-4f);
    return (2.0f + invAlpha) * powf(sin2h, invAlpha * 0.5f) / (2.0f * M_PI);
}

__forceinline__ __device__ float V_Neubelt(float NdotV, float NdotL) {
    return 1.0f / (4.0f * (NdotL + NdotV - NdotL * NdotV) + BRDF_EPSILON);
}

//------------------------------------------------------------------------------
// Clearcoat / Diffuse / Specular Lobe Mixture
//
// The indirect estimator samples a MIXTURE of the cosine (diffuse) lobe, the
// GGX VNDF (specular) lobe, and (at QUALITY_HIGH+) the clearcoat GGX VNDF
// lobe. Sampling VNDF only — the previous behavior — leaves the diffuse term
// effectively unsampled on smooth materials (the VNDF PDF is near-delta),
// which means unbounded variance or, with firefly clamps, lost diffuse GI.
// The mixture PDF below is shared by the sampler, the MIS weights, and the
// path-guide combination; all three MUST agree.
//------------------------------------------------------------------------------

// Probability of selecting the specular lobe when sampling this BSDF.
// Deterministic in (NdotV, material) so sampler/PDF/MIS recompute it
// identically. Clamped away from 0/1 when both lobes are present so neither
// lobe is starved (pure metals get exactly 1, FAST quality gets exactly 0).
__forceinline__ __device__ float specularSelectProb(
    float NdotV, const float3& baseColor, float metallic, unsigned int quality)
{
    if (quality == QUALITY_FAST) return 0.0f;  // FAST = Lambertian only
    if (metallic > 0.95f) return 1.0f;         // metals have no diffuse lobe

    float3 F0 = lerp(make_float3(0.04f, 0.04f, 0.04f),
                     make_float3(baseColor.x, baseColor.y, baseColor.z), metallic);
    float specW = luminance3(fresnelSchlick(NdotV, F0));
    float diffW = (1.0f - metallic) * (1.0f - specW) * luminance3(baseColor);
    float total = specW + diffW;
    if (total < 1e-6f) return 1.0f;  // black material: lobe choice is irrelevant
    return clamp(specW / total, 0.1f, 0.9f);
}

// Fraction of the NON-specular selection budget spent on the sheen leg — a
// uniform-hemisphere proposal for the Charlie sheen lobe. Charlie's energy
// sits at grazing angles where neither the cosine nor the VNDF proposal has
// mass, so strong-sheen materials (velvet/cloth) otherwise get their sheen
// only through badly-matched proposals — variance the firefly clamp then
// turns into visible energy loss. Uniform-hemisphere is the standard
// proposal for Charlie (Estevez & Kulla 2017): it keeps grazing directions
// reachable at O(1/2pi) density. Deterministic in the material, so
// sampler / mixture PDF / NEE MIS recompute it identically (same contract
// as specularSelectProb). Capped at 0.5 so the cosine leg is never starved.
__forceinline__ __device__ float sheenSelectProb(
    const float3& sheenColor, const float3& baseColor, float metallic,
    unsigned int quality)
{
    if (quality == QUALITY_FAST) return 0.0f;  // FAST evaluates pure Lambert
    float lumSheen = luminance3(sheenColor);
    if (lumSheen <= 0.0f) return 0.0f;
    float lumDiff = (1.0f - metallic) * luminance3(baseColor);
    return fminf(lumSheen / (lumSheen + lumDiff + 1e-4f), 0.5f);
}

// Probability of selecting the clearcoat lobe. The coat sits on top of every
// base lobe, so it takes the FIRST slice of the selection budget and the base
// machinery keeps the remainder. The weight is the coat's approximate
// directional albedo, clearcoat * F_cc(NdotV): with that weight the peak f/p
// of the coat lobe cancels to ~1, so a smooth coat over any base no longer
// arrives as rare huge-weight samples through the cosine/base-GGX tails.
// Deterministic in (NdotV, material, quality) — same contract as
// specularSelectProb — and capped at 0.5 so the base lobes are never starved.
// Zero below QUALITY_HIGH: evalPbrBSDF only contains the coat lobe at HIGH+,
// and the sampler must not propose a lobe the evaluator does not have.
__forceinline__ __device__ float clearcoatSelectProb(
    float NdotV, float clearcoat, unsigned int quality)
{
    if (quality < QUALITY_HIGH || clearcoat <= 0.0f) return 0.0f;
    return fminf(clearcoat * fresnelDielectricExternal(NdotV, DIELECTRIC_IOR), 0.5f);
}

// Uniform hemisphere sampling (the sheen leg's proposal)
__forceinline__ __device__ float3 sampleUniformHemisphere(float u1, float u2) {
    float z = u1;
    float r = sqrtf(fmaxf(1.0f - z * z, 0.0f));
    float phi = 2.0f * M_PI * u2;
    float sinPhi, cosPhi;
    sincosf(phi, &sinPhi, &cosPhi);
    return make_float3(r * cosPhi, r * sinPhi, z);
}

#define PDF_UNIFORM_HEMISPHERE (1.0f / (2.0f * M_PI))

// Mixture PDF for direction L given view V and shading normal N.
// pSpec / pSheen / pCC must come from specularSelectProb / sheenSelectProb /
// clearcoatSelectProb with the same inputs the sampler used.
__forceinline__ __device__ float pdfBSDFMixture(
    const float3& V, const float3& L, const float3& N,
    float roughness, float clearcoatRoughness,
    float pSpec, float pSheen, float pCC)
{
    float NdotL = dot(N, L);
    if (NdotL <= 0.0f) return 0.0f;

    // Non-specular leg: cosine diffuse blended with the uniform sheen proposal
    float pdfDiffuse = NdotL / M_PI;
    float pdfDiffLeg = (1.0f - pSheen) * pdfDiffuse + pSheen * PDF_UNIFORM_HEMISPHERE;
    if (pSpec <= 0.0f && pCC <= 0.0f) return pdfDiffLeg;

    float3 H = normalize(V + L);
    float NdotH = fmaxf(dot(N, H), 0.0f);
    float NdotV = fmaxf(dot(N, V), BRDF_EPSILON);

    float pdfBase = pdfDiffLeg;
    if (pSpec > 0.0f) {
        float alpha = fmaxf(roughness * roughness, 0.001f);
        float pdfSpec = pdfGGXVNDF(D_GGX(NdotH, alpha), G1_GGX(NdotV, alpha), NdotV);
        pdfBase = pSpec * pdfSpec + (1.0f - pSpec) * pdfDiffLeg;
    }
    if (pCC <= 0.0f) return pdfBase;

    // Clearcoat leg: VNDF density of the coat's own GGX lobe
    float ccAlpha = fmaxf(clearcoatRoughness * clearcoatRoughness, 0.001f);
    float pdfCC = pdfGGXVNDF(D_GGX(NdotH, ccAlpha), G1_GGX(NdotV, ccAlpha), NdotV);
    return pCC * pdfCC + (1.0f - pCC) * pdfBase;
}

// Sample a direction from the clearcoat/diffuse/specular mixture.
// Returns false if the sampled direction is below the shading hemisphere.
__forceinline__ __device__ bool sampleBSDFMixture(
    const float3& V, const float3& N,
    float roughness, float clearcoatRoughness,
    float pSpec, float pSheen, float pCC,
    float uSelect, float u1, float u2,
    float3& outL)
{
    float3 T, B;
    buildOrthonormalBasis(N, T, B);

    float3 L;
    if (uSelect < pCC) {
        // Clearcoat: GGX VNDF with the coat's own alpha
        float ccAlpha = fmaxf(clearcoatRoughness * clearcoatRoughness, 0.001f);
        float3 Ve = make_float3(dot(V, T), dot(V, B), dot(V, N));
        float3 Hl = sampleGGXVNDF(Ve, ccAlpha, u1, u2);
        float3 H = T * Hl.x + B * Hl.y + N * Hl.z;
        L = 2.0f * dot(V, H) * H - V;
    } else {
        // Base lobes get the leftover budget; remap the select variable
        // (uniform given uSelect >= pCC).
        float uBase = (uSelect - pCC) / fmaxf(1.0f - pCC, 1e-6f);
        if (uBase < pSpec) {
            // Specular: GGX VNDF (Heitz 2018) — visible normals only
            float alpha = fmaxf(roughness * roughness, 0.001f);
            float3 Ve = make_float3(dot(V, T), dot(V, B), dot(V, N));
            float3 Hl = sampleGGXVNDF(Ve, alpha, u1, u2);
            float3 H = T * Hl.x + B * Hl.y + N * Hl.z;
            L = 2.0f * dot(V, H) * H - V;
        } else {
            // Non-specular: remap again (uniform given uBase >= pSpec) to
            // choose sheen-uniform vs cosine diffuse.
            float rSel = (uBase - pSpec) / fmaxf(1.0f - pSpec, 1e-6f);
            float3 Ll = (rSel < pSheen) ? sampleUniformHemisphere(u1, u2)
                                        : sampleCosineHemisphere(u1, u2);
            L = T * Ll.x + B * Ll.y + N * Ll.z;
        }
    }

    L = normalize(L);
    if (dot(N, L) <= 0.0f) return false;
    outL = L;
    return true;
}

//------------------------------------------------------------------------------
// Full BSDF Evaluation (+ optional mixture PDF in the same pass)
//
// Lobes: Lambert diffuse * (1-F)(1-metallic), GGX specular with exact Fresnel
// and multiple-scattering compensation, clearcoat (QUALITY_HIGH+), sheen.
// QUALITY_FAST evaluates pure Lambert so the FAST sampler (cosine) and
// evaluator describe the same material — direct and indirect agree.
//------------------------------------------------------------------------------
__forceinline__ __device__ float3 evalPbrBSDF(
    const float3& V, const float3& L, const float3& N,
    const float3& baseColor, float metallic, float roughness,
    float clearcoat, float clearcoatRoughness,
    const float3& sheenColor, float sheenRoughness,
    unsigned int quality,
    float pSpec,        // from specularSelectProb (same inputs as the sampler)
    float pSheen,       // from sheenSelectProb (same inputs as the sampler)
    float pCC,          // from clearcoatSelectProb (same inputs as the sampler)
    float* outPdf)      // optional: mixture PDF of L (nullptr to skip)
{
    float NdotL = dot(N, L);
    float NdotV = fmaxf(dot(N, V), BRDF_EPSILON);
    if (NdotL <= 0.0f) {
        if (outPdf) *outPdf = 0.0f;
        return make_float3(0.0f, 0.0f, 0.0f);
    }
    // Single source of truth for the mixture density: delegate to
    // pdfBSDFMixture so this fused path can never drift from the sampler /
    // MIS formula (a hand-mirrored copy here had already diverged in its
    // NdotL clamping; the sampler's true density uses the raw NdotL).
    if (outPdf) {
        *outPdf = pdfBSDFMixture(V, L, N, roughness, clearcoatRoughness,
                                 pSpec, pSheen, pCC);
    }
    NdotL = fmaxf(NdotL, BRDF_EPSILON);

    if (quality == QUALITY_FAST) {
        return baseColor / M_PI;
    }

    float3 H = normalize(V + L);
    float NdotH = fmaxf(dot(N, H), 0.0f);
    float VdotH = fmaxf(dot(V, H), 0.0f);

    // Roughness remapping (Disney/UE4 convention)
    float alpha = fmaxf(roughness * roughness, 0.001f);

    // Shared microfacet terms
    float D = D_GGX(NdotH, alpha);
    float Vis = V_SmithGGX(NdotV, NdotL, alpha);

    // Exact Fresnel (dielectric / conductor dispatch)
    float3 F = fresnelDispatch(VdotH, baseColor, metallic);

    // Specular with multiple-scattering energy compensation
    float3 F0 = lerp(make_float3(0.04f, 0.04f, 0.04f), baseColor, metallic);
    float3 ems = multiScatterCompensation(F0, NdotV, roughness);
    float3 specular = (D * Vis) * F * ems;

    // Energy-conserving diffuse: metals have no diffuse component
    float3 kD = (make_float3(1.0f, 1.0f, 1.0f) - F) * (1.0f - metallic);
    float3 f = kD * baseColor / M_PI + specular;

    // Clearcoat layer (always dielectric, IOR 1.5) — HIGH quality and up
    if (clearcoat > 0.0f && quality >= QUALITY_HIGH) {
        float ccAlpha = fmaxf(clearcoatRoughness * clearcoatRoughness, 0.001f);
        float F_cc = fresnelDielectricExternal(VdotH, DIELECTRIC_IOR);
        float D_cc = D_GGX(NdotH, ccAlpha);
        float V_cc = V_SmithGGX(NdotV, NdotL, ccAlpha);
        float cc = D_cc * V_cc * F_cc;
        // Clearcoat on top absorbs some light from the base layer
        f = f * (1.0f - clearcoat * F_cc) + make_float3(cc, cc, cc) * clearcoat;
    }

    // Sheen (additive cloth lobe)
    if (sheenColor.x > 0.0f || sheenColor.y > 0.0f || sheenColor.z > 0.0f) {
        float Ds = D_Charlie(NdotH, sheenRoughness);
        float Vs = V_Neubelt(NdotV, NdotL);
        f = f + sheenColor * (Ds * Vs);
    }

    return f;
}

//------------------------------------------------------------------------------
// Refraction Utilities
//------------------------------------------------------------------------------

// Compute refraction direction using Snell's law.
// eta = eta_i / eta_t (same convention as fresnelDielectric).
// Returns false if total internal reflection occurs.
__forceinline__ __device__ bool refract(
    const float3& I,    // Incident direction (pointing toward surface)
    const float3& N,    // Surface normal (facing the incident side)
    float eta,
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
// Standard HDR format: U wraps horizontally (0-1), V goes from top (0) to bottom (1)
__forceinline__ __device__ float2 directionToEquirectangular(const float3& dir) {
    float3 d = normalize(dir);

    // Azimuthal angle (horizontal) - atan2 gives -PI to PI, map to 0-1
    float phi = atan2f(d.z, d.x);
    float u = phi / (2.0f * M_PI) + 0.5f;

    // Polar angle from +Y axis using acos - gives 0 (up) to PI (down)
    float theta = acosf(clamp(d.y, -1.0f, 1.0f));
    float v = theta / M_PI;

    return make_float2(u, v);
}

// Convert equirectangular UV to direction (inverse of directionToEquirectangular)
__forceinline__ __device__ float3 equirectangularToDirection(float u, float v) {
    float phi = (u - 0.5f) * 2.0f * M_PI;
    float theta = v * M_PI;

    float sinTheta, cosTheta, sinPhi, cosPhi;
    sincosf(theta, &sinTheta, &cosTheta);
    sincosf(phi, &sinPhi, &cosPhi);

    return make_float3(sinTheta * cosPhi, cosTheta, sinTheta * sinPhi);
}

//------------------------------------------------------------------------------
// Environment Map Importance Sampling — Walker/Vose alias table
//
// Sampling is O(1): one bucket pick + one accept/alias compare, replacing the
// old log2(W)+log2(H) DEPENDENT, point-filtered CDF binary searches (~22
// serialized texture fetches per sample on a 2K map). The sampled distribution
// is IDENTICAL — per-texel probability proportional to sin(theta)-weighted
// luminance, with a uniform sub-texel jitter — so variance/convergence are
// unchanged; this is purely a latency win. The host (EnvironmentMap) builds the
// alias tables (prob[], alias[]) and the per-texel pmf once at load.
//
// Unbiasedness: environmentPdf() (the MIS density) reads the SAME pmf the
// sampler uses, so the two stay in exact lockstep. The accept test takes its
// OWN random number (xiAccept): reusing frac(xi1*N) only works in exact
// arithmetic — a float xi1 has 24 mantissa bits, and for a 2K map (N = 2^21)
// the recovered fraction is quantized to ~3 bits in the upper buckets,
// skewing per-texel selection away from the pmf the MIS density divides by
// (a bias accumulation converges TO, growing with env resolution).
//------------------------------------------------------------------------------

// Sample a direction from the environment proportional to sin(theta)-weighted
// luminance via the alias table. Returns the direction and its solid-angle PDF.
// xi1 picks the bucket, xiAccept drives the accept/alias test; xi2, xi3
// jitter the sample uniformly inside the chosen texel.
__forceinline__ __device__ float3 sampleEnvironmentDirection(
    float xi1, float xiAccept, float xi2, float xi3,
    const float* aliasProb,             // [W*H] bucket accept-probabilities
    const unsigned int* aliasIdx,       // [W*H] bucket fallback texels
    const float* pmf,                   // [W*H] per-texel selection probability
    unsigned int envWidth,
    unsigned int envHeight,
    float& outPdf)
{
    unsigned int n = envWidth * envHeight;
    unsigned int bucket = (unsigned int)(xi1 * (float)n);
    if (bucket >= n) bucket = n - 1;
    unsigned int texel = (xiAccept < aliasProb[bucket]) ? bucket : aliasIdx[bucket];

    unsigned int col = texel % envWidth;
    unsigned int row = texel / envWidth;

    // Continuous UV inside the selected texel (uniform sub-texel jitter)
    float u = ((float)col + xi2) / (float)envWidth;
    float v = ((float)row + xi3) / (float)envHeight;

    float3 dir = equirectangularToDirection(u, v);

    // Solid-angle PDF: per-texel probability * (W*H) is the piecewise-constant
    // density over [0,1]^2; dividing by the equirectangular Jacobian
    // 2*pi^2*sin(theta) converts to solid angle. Identical to the old CDF path
    // (there pmf == marginalPdf * conditionalPdf).
    float theta = v * M_PI;
    float sinTheta = fmaxf(sinf(theta), 1e-6f);
    outPdf = pmf[texel] * (float)n / (2.0f * M_PI * M_PI * sinTheta);

    return dir;
}

// Solid-angle PDF of sampling a given direction from the environment.
// Must match sampleEnvironmentDirection exactly (used for MIS).
__forceinline__ __device__ float environmentPdf(
    const float3& dir,
    const float* pmf,
    unsigned int envWidth,
    unsigned int envHeight)
{
    float2 uv = directionToEquirectangular(dir);
    int col = clamp((int)(uv.x * envWidth), 0, (int)envWidth - 1);
    int row = clamp((int)(uv.y * envHeight), 0, (int)envHeight - 1);
    unsigned int texel = (unsigned int)row * envWidth + (unsigned int)col;

    float theta = uv.y * M_PI;
    float sinTheta = fmaxf(sinf(theta), 1e-6f);
    return pmf[texel] * (float)(envWidth * envHeight) / (2.0f * M_PI * M_PI * sinTheta);
}

// Sample environment map radiance for a given direction
__forceinline__ __device__ float3 sampleEnvironmentRadiance(
    const float3& dir,
    cudaTextureObject_t envMap,
    float intensity)
{
    float2 uv = directionToEquirectangular(dir);
    float4 envSample = tex2D<float4>(envMap, uv.x, uv.y);
    return make_float3(envSample.x, envSample.y, envSample.z) * intensity;
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
