#include <optix.h>
#include "gpu_types.h"
#include "shared_device.h"
#include "path_guide_grid_device.h"
#include "brdf.h"

__forceinline__ __device__ bool traceShadowRay(
    const float3& origin,
    const float3& normal,
    const float3& direction,
    float tmax)
{
    const float normalEps = 0.0001f;
    const float rayEps = 0.001f;

    float NdotD = dot(normal, direction);
    float3 offsetNormal = (NdotD > 0.0f) ? normal : -normal;
    float3 offsetOrigin = origin + offsetNormal * normalEps;

    unsigned int occluded = 0;
    float safeTmax = fmaxf(tmax - rayEps, rayEps * 2.0f);

    optixTrace(
        params.scene_handle,
        offsetOrigin,
        direction,
        rayEps,
        safeTmax,
        0.0f,
        0xFF,
        OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT | OPTIX_RAY_FLAG_DISABLE_ANYHIT,
        RAY_TYPE_SHADOW,
        RAY_TYPE_COUNT,
        RAY_TYPE_SHADOW,
        occluded
    );

    return occluded == 0;
}

__forceinline__ __device__ float calculateTextureLOD(float rayDistance) {
    float footprint = rayDistance * params.pixel_world_size;
    float lod = log2f(fmaxf(1.0f, footprint));
    return fminf(fmaxf(lod, 0.0f), 12.0f);
}

__forceinline__ __device__ float4 sampleTexture(cudaTextureObject_t tex, float2 uv, float4 fallback, float lod) {
    return tex != 0 ? tex2DLod<float4>(tex, uv.x, uv.y, lod) : fallback;
}

__forceinline__ __device__ float4 sampleTexture(cudaTextureObject_t tex, float2 uv, float4 fallback) {
    return tex != 0 ? tex2D<float4>(tex, uv.x, uv.y) : fallback;
}

//------------------------------------------------------------------------------
// Direct Lighting — one light sample per hit (luminance-weighted importance sampling)
//------------------------------------------------------------------------------

__forceinline__ __device__ float luminance3(float3 c) {
    return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;
}

__forceinline__ __device__ float3 evalBRDF(const float3& L, const float3& V, const float3& shadingNormal,
    const float3& baseColor, float metallic, float roughness, float clearcoat, float clearcoatRoughness,
    const float3& sheenColor, float sheenRoughness) {
    float3 brdf;
    if (params.quality_mode == QUALITY_FAST) {
        brdf = evaluateLambertian(baseColor);
    } else if (clearcoat > 0.0f && params.quality_mode >= QUALITY_HIGH) {
        brdf = evaluateBRDF_Clearcoat(V, L, shadingNormal, baseColor, metallic, roughness, clearcoat, clearcoatRoughness);
    } else {
        brdf = evaluateGGX_BRDF(V, L, shadingNormal, baseColor, metallic, roughness);
    }
    if (sheenColor.x > 0.0f || sheenColor.y > 0.0f || sheenColor.z > 0.0f) {
        brdf = brdf + evaluateSheen(V, L, shadingNormal, sheenColor, sheenRoughness);
    }
    return brdf;
}

__forceinline__ __device__ float3 computeStandardDirectLighting(
    const float3& hitPos,
    const float3& geomNormal,
    const float3& shadingNormal,
    const float3& V,
    const float3& baseColor,
    float metallic,
    float roughness,
    float clearcoat,
    float clearcoatRoughness,
    const float3& sheenColor,
    float sheenRoughness,
    unsigned int& seed,
    float3* out_lightDir = nullptr,
    float* out_contribLuminance = nullptr)
{
    float3 Lo = make_float3(0.0f, 0.0f, 0.0f);
    if (out_lightDir) *out_lightDir = make_float3(0.0f, 0.0f, 0.0f);
    if (out_contribLuminance) *out_contribLuminance = 0.0f;

    // One light sample per hit (luminance-weighted importance sampling) — one shadow ray
    if (params.total_light_luminance > 0.0f) {
        float target = randomFloat(seed) * params.total_light_luminance;
        float cumulative = 0.0f;
        bool lightSampled = false;

        for (unsigned int i = 0; i < params.point_light_count && !lightSampled; ++i) {
            float lum = luminance3(params.point_lights[i].intensity);
            cumulative += lum;
            if (cumulative >= target) {
                const GpuPointLight& light = params.point_lights[i];
                float3 lightVec = light.position - hitPos;
                float distance = fmaxf(length(lightVec), 0.001f);  // Prevent Inf from 1/d²
                float3 L = lightVec / distance;
                float NdotL = dot(shadingNormal, L);
                if (NdotL > 0.0f) {
                    bool visible = traceShadowRay(hitPos, geomNormal, L, distance);
                    float att = 1.0f / (distance * distance);
                    float3 brdf = evalBRDF(L, V, shadingNormal, baseColor, metallic, roughness, clearcoat, clearcoatRoughness, sheenColor, sheenRoughness);
                    float3 contrib = brdf * light.intensity * att * NdotL * (visible ? (params.total_light_luminance / lum) : 0.0f);
                    Lo = Lo + contrib;
                    if (out_lightDir) *out_lightDir = L;
                    if (out_contribLuminance) *out_contribLuminance = luminance3(contrib);
                }
                lightSampled = true;
            }
        }
        if (!lightSampled) {
            for (unsigned int i = 0; i < params.directional_light_count && !lightSampled; ++i) {
                float lum = luminance3(params.directional_lights[i].irradiance) * 10.0f;
                cumulative += lum;
                if (cumulative >= target) {
                    const GpuDirectionalLight& light = params.directional_lights[i];
                    float3 L = -normalize(light.direction);
                    float NdotL = dot(shadingNormal, L);
                    if (NdotL > 0.0f) {
                        bool visible = traceShadowRay(hitPos, geomNormal, L, 10000.0f);
                        float3 brdf = evalBRDF(L, V, shadingNormal, baseColor, metallic, roughness, clearcoat, clearcoatRoughness, sheenColor, sheenRoughness);
                        float3 contrib = brdf * light.irradiance * NdotL * (visible ? (params.total_light_luminance / lum) : 0.0f);
                        Lo = Lo + contrib;
                        if (out_lightDir) *out_lightDir = L;
                        if (out_contribLuminance) *out_contribLuminance = luminance3(contrib);
                    }
                    lightSampled = true;
                }
            }
        }
        if (!lightSampled) {
            for (unsigned int i = 0; i < params.area_light_count && !lightSampled; ++i) {
                float lum = luminance3(params.area_lights[i].emission);
                cumulative += lum;
                if (cumulative >= target) {
                    const GpuAreaLight& light = params.area_lights[i];

                    // Sample a random point on the rectangular light surface
                    float3 bitangent = cross(light.normal, light.tangent);
                    float u = randomFloat(seed) - 0.5f; // [-0.5, 0.5]
                    float v = randomFloat(seed) - 0.5f;
                    float3 samplePos = light.position
                        + light.tangent  * (u * light.size.x)
                        + bitangent      * (v * light.size.y);

                    float3 lightVec = samplePos - hitPos;
                    float distance = fmaxf(length(lightVec), 0.001f);
                    float3 L = lightVec / distance;
                    float NdotL = dot(shadingNormal, L);
                    float lightNdotL = -dot(light.normal, L);
                    if (NdotL > 0.0f && lightNdotL > 0.0f) {
                        bool visible = traceShadowRay(hitPos, geomNormal, L, distance);
                        // PDF of uniform rectangular sample = 1/area, so area cancels
                        // Solid angle measure: dA * cos(theta_light) / dist^2
                        float att = lightNdotL / (distance * distance);
                        float3 brdf = evalBRDF(L, V, shadingNormal, baseColor, metallic, roughness, clearcoat, clearcoatRoughness, sheenColor, sheenRoughness);
                        float3 contrib = brdf * light.emission * light.area * att * NdotL * (visible ? (params.total_light_luminance / lum) : 0.0f);
                        Lo = Lo + contrib;
                        if (out_lightDir) *out_lightDir = L;
                        if (out_contribLuminance) *out_contribLuminance = luminance3(contrib);
                    }
                    lightSampled = true;
                }
            }
        }
    }

    // Environment map (runs when env present; adds to Lo so lights + env both contribute)
    if (params.environment_map != 0 && params.env_conditional_cdf != 0 && params.env_marginal_cdf != 0) {
        float xi1 = randomFloat(seed);
        float xi2 = randomFloat(seed);

        float envPdf;
        float3 L = sampleEnvironmentDirection(
            xi1, xi2,
            params.env_marginal_cdf,
            params.env_conditional_cdf,
            params.env_width,
            params.env_height,
            params.env_total_luminance,
            envPdf
        );

        float NdotL = dot(shadingNormal, L);

        if (NdotL > 0.0f && envPdf > 0.0f) {
            bool visible = traceShadowRay(hitPos, geomNormal, L, 10000.0f);

            if (visible) {
                float3 envRadiance = sampleEnvironmentRadiance(L, params.environment_map, params.environment_intensity);

                float3 brdf = evalBRDF(L, V, shadingNormal, baseColor, metallic, roughness, clearcoat, clearcoatRoughness, sheenColor, sheenRoughness);

                float3 contrib = brdf * envRadiance * NdotL / envPdf;
                Lo = Lo + contrib;
                if (out_lightDir) *out_lightDir = L;
                if (out_contribLuminance) *out_contribLuminance = luminance3(contrib);
            }
        }
    }

    return Lo;
}

//------------------------------------------------------------------------------
// Path-Guided Indirect Lighting with One-Sample MIS
// Reference: Müller et al., "Practical Path Guiding", EGSR 2017
//
// === Design Philosophy ===
//
// Guiding distribution: We learn the incident radiance field Li(x, ω), NOT the
// full rendering integrand Li * f * cosθ ("product guiding"). Li-only guiding
// is correct for all BSDF types and naturally extends to additional sampling
// dimensions (wavelength, lens position) since the BSDF's spectral/spatial
// variation is handled separately via MIS. For rough/diffuse surfaces the BSDF
// is broad and Li-only is near-optimal; for glossy surfaces the VNDF BSDF
// sampling leg compensates via MIS.
//
// Training signal: Each hit deposits (direction, Li/p(ω)) — importance-weighted
// incident luminance. The 1/p(ω) factor makes deposits unbiased regardless of
// which sampling strategy produced the direction, preventing self-reinforcing
// feedback loops (Müller 2017 §4.2). Both primary and bounce hits train.
//
// Below-hemisphere: vMF covers the full sphere; directions with NdotL ≤ 0 are
// rejected (zero contribution). This is inherent to sphere-domain guiding and
// is correctly handled by the MIS weight (combinedPdf accounts for the guide's
// full-sphere PDF). The VNDF BSDF leg never wastes samples below the horizon,
// making it a natural complement.
//
// One-sample MIS (balance heuristic):
// 1. With probability alpha, sample from path guide (vMF)
// 2. With probability (1-alpha), sample from BSDF (VNDF)
// 3. Combined PDF = alpha*p_guide + (1-alpha)*p_bsdf
// 4. Estimator: f(L) * Li(L) * NdotL / combinedPdf
//------------------------------------------------------------------------------

// Debug stat indices
#define GUIDE_STAT_ATTEMPTS     0
#define GUIDE_STAT_CELL_FOUND   1
#define GUIDE_STAT_VALID_LOBE   2
#define GUIDE_STAT_BELOW_HORIZ  3
#define GUIDE_STAT_CONTRIBUTED  4
#define GUIDE_STAT_BSDF_SAMPLED 5
#define GUIDE_STAT_COUNT        6

__forceinline__ __device__ void incrementGuideStat(unsigned int statIdx) {
    if (params.path_guide_debug_enabled && params.path_guide_debug_stats != nullptr) {
        atomicAdd(&params.path_guide_debug_stats[statIdx], 1u);
    }
}

// VNDF PDF for reflected direction L.
// Uses Heitz 2018 visible normal distribution: p(L) = D(H) * G1(V) / (4 * NdotV).
// This matches sampleBSDFDirection below — sampler and PDF must always agree.
__forceinline__ __device__ float computeBSDFPdf(
    const float3& V, const float3& L, const float3& N, float roughness)
{
    float3 H = normalize(V + L);
    float NdotH = fmaxf(dot(N, H), 0.0f);
    float NdotV = fmaxf(dot(N, V), BRDF_EPSILON);

    float alpha = roughness * roughness;
    alpha = fmaxf(alpha, 0.001f);

    float D = D_GGX(NdotH, alpha);
    float G1 = G1_GGX(NdotV, alpha);

    return pdfGGXVNDF(D, G1, NdotV);
}

// Sample direction from GGX VNDF (Heitz 2018).
// Samples only visible microfacet normals → no below-hemisphere waste from BSDF leg,
// lower variance for glossy materials, and naturally complements vMF guiding in MIS.
__forceinline__ __device__ float3 sampleBSDFDirection(
    const float3& V, const float3& N, float roughness,
    float u1, float u2)
{
    float alpha = roughness * roughness;
    alpha = fmaxf(alpha, 0.001f);

    // Build tangent frame
    float3 up = (fabsf(N.y) < 0.999f) ? make_float3(0.0f, 1.0f, 0.0f) : make_float3(1.0f, 0.0f, 0.0f);
    float3 T = normalize(cross(up, N));
    float3 B = cross(N, T);

    // Transform V to tangent space for VNDF sampling
    float3 Ve = make_float3(dot(V, T), dot(V, B), dot(V, N));

    // Sample visible microfacet normal in tangent space
    float3 H_local = sampleGGXVNDF(Ve, alpha, u1, u2);

    // Transform H back to world space
    float3 H = T * H_local.x + B * H_local.y + N * H_local.z;

    // Reflect V around H to get L
    float VdotH = dot(V, H);
    float3 L = 2.0f * VdotH * H - V;

    return normalize(L);
}

// Build lightweight grid descriptor from launch params. Pointers only — no array copy.
// Level resolutions are read directly from __constant__ params by sparseResolutionAtLevel().
__forceinline__ __device__ SparsePathGuideDescriptorDevice buildGridDescriptor() {
    SparsePathGuideDescriptorDevice grid = {};
    grid.morton_codes = params.path_guide_morton_codes;
    grid.data = params.path_guide_data;
    grid.level_offsets = params.path_guide_level_offsets;
    grid.num_levels = params.path_guide_num_levels;
    grid.entry_stride = params.path_guide_entry_stride;
    grid.bounds_min[0] = params.path_guide_bounds_min[0];
    grid.bounds_min[1] = params.path_guide_bounds_min[1];
    grid.bounds_min[2] = params.path_guide_bounds_min[2];
    grid.bounds_max[0] = params.path_guide_bounds_max[0];
    grid.bounds_max[1] = params.path_guide_bounds_max[1];
    grid.bounds_max[2] = params.path_guide_bounds_max[2];
    grid.hash_keys = params.path_guide_hash_keys;
    grid.hash_values = params.path_guide_hash_values;
    grid.hash_table_size = params.path_guide_hash_table_size;
    grid.hash_shift = params.path_guide_hash_shift;
    return grid;
}

__forceinline__ __device__ float3 computeGuidedIndirectLighting(
    const SparsePathGuideDescriptorDevice& grid,
    unsigned int cellIdx,
    bool hasValidGuide,
    const TrilinearInfo& trilinear,
    const float3& hitPos,
    const float3& geomNormal,
    const float3& shadingNormal,
    const float3& V,
    const float3& baseColor,
    float metallic,
    float roughness,
    unsigned int& seed,
    float3* out_bounceDir = nullptr,
    float* out_bounceLuminance = nullptr,
    float* out_incidentLuminance = nullptr,
    float* out_samplingPdf = nullptr)
{
    if (out_bounceDir) *out_bounceDir = make_float3(0.0f, 0.0f, 0.0f);
    if (out_bounceLuminance) *out_bounceLuminance = 0.0f;
    if (out_incidentLuminance) *out_incidentLuminance = 0.0f;
    if (out_samplingPdf) *out_samplingPdf = 1.0f;
    // Early out if guiding is disabled
    if (params.path_guide_enabled == 0) {
        return make_float3(0.0f, 0.0f, 0.0f);
    }

    incrementGuideStat(GUIDE_STAT_ATTEMPTS);

    // MIS blend factor: probability of using guide vs BSDF
    float alpha = params.path_guide_mis_weight;  // e.g., 0.5 = equal probability

    if (cellIdx != 0xFFFFFFFFu) {
        incrementGuideStat(GUIDE_STAT_CELL_FOUND);
        if (hasValidGuide) {
            incrementGuideStat(GUIDE_STAT_VALID_LOBE);
        }
    }

    // If no valid guide, fall back to pure BSDF sampling for indirect lighting
    if (!hasValidGuide) {
        float u1 = randomFloat(seed);
        float u2 = randomFloat(seed);
        float3 L = sampleBSDFDirection(V, shadingNormal, roughness, u1, u2);
        float NdotL = dot(shadingNormal, L);
        if (NdotL <= 0.0f) return make_float3(0.0f, 0.0f, 0.0f);

        float brdfPdf = computeBSDFPdf(V, L, shadingNormal, roughness);
        if (brdfPdf <= 1e-8f) return make_float3(0.0f, 0.0f, 0.0f);

        const float rayEps = 0.001f;
        float NdotD = dot(geomNormal, L);
        float3 offsetNormal = (NdotD > 0.0f) ? geomNormal : -geomNormal;
        float3 offsetOrigin = hitPos + offsetNormal * rayEps;

        unsigned int p0, p1, p2, p3, p4, p5;
        p0 = __float_as_uint(0.0f);
        p1 = __float_as_uint(0.0f);
        p2 = __float_as_uint(0.0f);
        p3 = __float_as_uint(0.0f);
        p4 = 0xFFFFFFFFu;
        p5 = getPayloadDepth() + 1;

        optixTrace(
            params.scene_handle,
            offsetOrigin,
            L,
            rayEps,
            10000.0f,
            0.0f,
            0xFF,
            OPTIX_RAY_FLAG_NONE,
            RAY_TYPE_INDIRECT,
            RAY_TYPE_COUNT,
            RAY_TYPE_INDIRECT,
            p0, p1, p2, p3, p4, p5
        );

        float3 Li = make_float3(__uint_as_float(p0), __uint_as_float(p1), __uint_as_float(p2));
        float liLum = 0.2126f * Li.x + 0.7152f * Li.y + 0.0722f * Li.z;
        float3 brdf = evaluateGGX_BRDF(V, L, shadingNormal, baseColor, metallic, roughness);
        float3 contrib = brdf * Li * NdotL / brdfPdf;
        float3 result = clamp(contrib, 0.0f, 100.0f);
        float lum = 0.2126f * result.x + 0.7152f * result.y + 0.0722f * result.z;
        if (out_bounceDir && lum > 1e-6f) *out_bounceDir = L;
        if (out_bounceLuminance) *out_bounceLuminance = lum;
        if (out_incidentLuminance) *out_incidentLuminance = liLum;
        if (out_samplingPdf) *out_samplingPdf = brdfPdf;
        return result;
    }

    // One-sample MIS: choose sampling strategy
    float strategyRand = randomFloat(seed);
    bool useGuide = (strategyRand < alpha);

    float3 L;
    float guidePdf = 0.0f;
    float brdfPdf = 0.0f;

    if (useGuide) {
        // Sample from path guide (vMF distribution)
        float u_lobe = randomFloat(seed);
        float u1 = randomFloat(seed);
        float u2 = randomFloat(seed);

        float guideX, guideY, guideZ;
        if (!pathGuideSampleDirection(grid, cellIdx, u_lobe, u1, u2, guideX, guideY, guideZ)) {
            return make_float3(0.0f, 0.0f, 0.0f);
        }

        L = make_float3(guideX, guideY, guideZ);
        guidePdf = trilinearGuidePdf(grid, trilinear, guideX, guideY, guideZ);
        brdfPdf = computeBSDFPdf(V, L, shadingNormal, roughness);

    } else {
        // Sample from BSDF (GGX distribution)
        incrementGuideStat(GUIDE_STAT_BSDF_SAMPLED);
        float u1 = randomFloat(seed);
        float u2 = randomFloat(seed);

        L = sampleBSDFDirection(V, shadingNormal, roughness, u1, u2);
        brdfPdf = computeBSDFPdf(V, L, shadingNormal, roughness);
        guidePdf = trilinearGuidePdf(grid, trilinear, L.x, L.y, L.z);
    }

    // Check if direction is valid
    float NdotL = dot(shadingNormal, L);
    if (NdotL <= 0.0f) {
        incrementGuideStat(GUIDE_STAT_BELOW_HORIZ);
        return make_float3(0.0f, 0.0f, 0.0f);
    }

    // Balance heuristic MIS weight
    // Combined PDF = alpha * guidePdf + (1 - alpha) * brdfPdf
    float combinedPdf = alpha * guidePdf + (1.0f - alpha) * brdfPdf;
    if (combinedPdf <= 1e-8f) {
        return make_float3(0.0f, 0.0f, 0.0f);
    }

    // Trace secondary ray (uses lightweight bounce shader via RAY_TYPE_INDIRECT)
    const float rayEps = 0.001f;
    float NdotD = dot(geomNormal, L);
    float3 offsetNormal = (NdotD > 0.0f) ? geomNormal : -geomNormal;
    float3 offsetOrigin = hitPos + offsetNormal * rayEps;

    unsigned int p0, p1, p2, p3, p4, p5;
    p0 = __float_as_uint(0.0f);
    p1 = __float_as_uint(0.0f);
    p2 = __float_as_uint(0.0f);
    p3 = __float_as_uint(0.0f);
    p4 = 0xFFFFFFFFu;
    p5 = getPayloadDepth() + 1;

    optixTrace(
        params.scene_handle,
        offsetOrigin,
        L,
        rayEps,
        10000.0f,
        0.0f,
        0xFF,
        OPTIX_RAY_FLAG_NONE,
        RAY_TYPE_INDIRECT,
        RAY_TYPE_COUNT,
        RAY_TYPE_INDIRECT,
        p0, p1, p2, p3, p4, p5
    );

    float3 incomingRadiance = make_float3(__uint_as_float(p0), __uint_as_float(p1), __uint_as_float(p2));

    // Evaluate BRDF at sampled direction
    float3 brdf = evaluateGGX_BRDF(V, L, shadingNormal, baseColor, metallic, roughness);

    // One-sample MIS estimator: f(x) * Li * cos / combinedPdf
    // No additional MIS weight needed - the combinedPdf already accounts for it
    float3 contrib = brdf * incomingRadiance * NdotL / combinedPdf;

    incrementGuideStat(GUIDE_STAT_CONTRIBUTED);

    float3 result = clamp(contrib, 0.0f, 100.0f);
    float lum = 0.2126f * result.x + 0.7152f * result.y + 0.0722f * result.z;
    // Training uses importance-weighted Li to make training unbiased regardless
    // of sampling strategy. Without this, guide-sampled directions oversample the
    // current lobe mean, creating a self-reinforcing feedback loop where cells
    // converge to different modes. Li/p(ω) makes each deposit an unbiased
    // estimate of ∫Li(ω)dω, breaking the feedback loop.
    // Müller 2017 avoids this with iterative rebuilds (fresh tree each iteration);
    // in our online setting, importance weighting is the correct substitute.
    float liLum = 0.2126f * incomingRadiance.x + 0.7152f * incomingRadiance.y + 0.0722f * incomingRadiance.z;
    if (out_bounceDir && lum > 1e-6f) *out_bounceDir = L;
    if (out_bounceLuminance) *out_bounceLuminance = lum;
    if (out_incidentLuminance) *out_incidentLuminance = liLum;
    if (out_samplingPdf) *out_samplingPdf = combinedPdf;
    return result;
}

//------------------------------------------------------------------------------
// Main Closest Hit Shader
//------------------------------------------------------------------------------

extern "C" __global__ void __closesthit__radiance() {
    const HitGroupData* sbtData = reinterpret_cast<HitGroupData*>(optixGetSbtDataPointer());
    const GpuMaterial& material = sbtData->material;
    const unsigned int instanceId = optixGetInstanceId();
    const unsigned int primIdx = optixGetPrimitiveIndex();
    const float2 barycentrics = optixGetTriangleBarycentrics();
    const float baryU = barycentrics.x;
    const float baryV = barycentrics.y;
    const float baryW = 1.0f - baryU - baryV;

    const GpuVertex* vertices = reinterpret_cast<const GpuVertex*>(params.vertex_buffers[instanceId]);
    const unsigned int* indices = reinterpret_cast<const unsigned int*>(params.index_buffers[instanceId]);

    const unsigned int i0 = indices[primIdx * 3 + 0];
    const unsigned int i1 = indices[primIdx * 3 + 1];
    const unsigned int i2 = indices[primIdx * 3 + 2];

    const GpuVertex& vert0 = vertices[i0];
    const GpuVertex& vert1 = vertices[i1];
    const GpuVertex& vert2 = vertices[i2];

    float3 objectPos = baryW * vert0.position + baryU * vert1.position + baryV * vert2.position;
    float3 hitPos = optixTransformPointFromObjectToWorldSpace(objectPos);

    // In pick mode, return world position via color payload and skip lighting
    if (params.pick_mode) {
        setPayloadColor(hitPos);
        setPayloadInstanceId(instanceId);
        return;
    }

    float3 objectNormal = normalize(baryW * vert0.normal + baryU * vert1.normal + baryV * vert2.normal);
    float3 geomNormal = normalize(optixTransformNormalFromObjectToWorldSpace(objectNormal));

    float4 tangent = baryW * vert0.tangent + baryU * vert1.tangent + baryV * vert2.tangent;
    float3 objectTangent = make_float3(tangent.x, tangent.y, tangent.z);
    float3 worldTangent = normalize(optixTransformVectorFromObjectToWorldSpace(objectTangent));
    float bitangentSign = tangent.w;

    float2 texCoord = make_float2(
        baryW * vert0.u + baryU * vert1.u + baryV * vert2.u,
        baryW * vert0.v + baryU * vert1.v + baryV * vert2.v
    );

    float rayDistance = optixGetRayTmax();
    float texLOD = calculateTextureLOD(rayDistance);

    float4 baseColorTex = sampleTexture(material.baseColorTex, texCoord, make_float4(1.0f, 1.0f, 1.0f, 1.0f), texLOD);
    float4 baseColor = make_float4(
        material.baseColor.x * baseColorTex.x,
        material.baseColor.y * baseColorTex.y,
        material.baseColor.z * baseColorTex.z,
        material.baseColor.w * baseColorTex.w
    );

    float metallic = material.metallic;
    float roughness = material.roughness;
    if (material.metallicRoughnessTex != 0) {
        float4 mrSample = tex2DLod<float4>(material.metallicRoughnessTex, texCoord.x, texCoord.y, texLOD);
        roughness = material.roughness * mrSample.y;
        metallic = material.metallic * mrSample.z;
    }
    roughness = fmaxf(roughness, 0.04f);

    float3 emissive = material.emissive;
    if (material.emissiveTex != 0) {
        float4 emissiveTex = tex2DLod<float4>(material.emissiveTex, texCoord.x, texCoord.y, texLOD);
        emissive = make_float3(
            material.emissive.x * emissiveTex.x,
            material.emissive.y * emissiveTex.y,
            material.emissive.z * emissiveTex.z
        );
    }

    float3 shadingNormal = geomNormal;
    if (material.normalTex != 0) {
        float4 normalSample = tex2DLod<float4>(material.normalTex, texCoord.x, texCoord.y, texLOD);
        float3 tangentNormal = unpackNormal(normalSample);
        shadingNormal = applyNormalMap(tangentNormal, geomNormal, worldTangent, bitangentSign);
    }

    // Double-sided: use unperturbed vertex normal for front/back check
    // Save original geometric normal before flip for glass entering/exiting detection
    float3 rayDir = optixGetWorldRayDirection();
    float3 origGeomNormal = geomNormal;
    float3 origShadingNormal = shadingNormal;
    if (material.doubleSided && dot(geomNormal, rayDir) > 0.0f) {
        shadingNormal = -shadingNormal;
        geomNormal = -geomNormal;
    }

    // V = incoming ray direction reversed (correct for both primary and bounce rays)
    float3 V = -normalize(rayDir);

    float clearcoat = material.clearcoat;
    float clearcoatRoughness = material.clearcoatRoughness;
    if (material.clearcoatTex != 0) {
        clearcoat *= tex2DLod<float4>(material.clearcoatTex, texCoord.x, texCoord.y, texLOD).x;
    }
    if (material.clearcoatRoughnessTex != 0) {
        clearcoatRoughness *= tex2DLod<float4>(material.clearcoatRoughnessTex, texCoord.x, texCoord.y, texLOD).y;
    }

    float3 sheenColor = material.sheenColor;
    float sheenRoughness = material.sheenRoughness;
    if (material.sheenColorTex != 0) {
        float4 sheenTex = tex2DLod<float4>(material.sheenColorTex, texCoord.x, texCoord.y, texLOD);
        sheenColor = make_float3(sheenColor.x * sheenTex.x, sheenColor.y * sheenTex.y, sheenColor.z * sheenTex.z);
    }
    if (material.sheenRoughnessTex != 0) {
        sheenRoughness *= tex2DLod<float4>(material.sheenRoughnessTex, texCoord.x, texCoord.y, texLOD).w;
    }

    float3 baseColorRGB = make_float3(baseColor.x, baseColor.y, baseColor.z);

    // Generate seed for random sampling (moved before glass block so randomFloat is available)
    unsigned int pixelIdx = optixGetLaunchIndex().y * params.width + optixGetLaunchIndex().x;
    unsigned int dirHash = __float_as_uint(rayDir.x) ^ __float_as_uint(rayDir.y) ^ __float_as_uint(rayDir.z);
    unsigned int seed = pixelIdx ^ (params.frame_index * 0x9E3779B9u) ^ dirHash;

    // ── Glass / Transmission ──
    if (material.transmission > 0.0f) {
        unsigned int depth = getPayloadDepth();
        if (depth < params.max_bounce_depth) {
            float3 I = rayDir;
            // Use original (pre-double-sided-flip) geometric normal for entering/exiting.
            // The double-sided flip makes both normals face the ray, which would make
            // every hit look like "entering" — trapping rays inside with the wrong eta.
            bool entering = dot(origGeomNormal, I) < 0.0f;

            float eta = entering ? (1.0f / material.ior) : material.ior;
            float3 N  = entering ? origShadingNormal : -origShadingNormal;
            float3 gN = entering ? origGeomNormal    : -origGeomNormal;

            float F = fresnelDielectric(dot(N, -I), eta);

            float3 newDir;
            bool reflected;
            if (randomFloat(seed) < F) {
                newDir = reflect(I, N);
                reflected = true;
            } else {
                if (!refract(I, N, eta, newDir)) {
                    newDir = reflect(I, N);  // total internal reflection
                    reflected = true;
                } else {
                    reflected = false;
                }
            }

            // Glass bounces use lightweight indirect shader to avoid full path guide overhead
            float3 origin = hitPos + (reflected ? gN : -gN) * 0.001f;
            unsigned int gp0, gp1, gp2, gp3, gp4, gp5;
            gp0 = gp1 = gp2 = __float_as_uint(0.0f);
            gp3 = __float_as_uint(0.0f);
            gp4 = 0xFFFFFFFFu;
            gp5 = depth + 1;

            optixTrace(params.scene_handle, origin, normalize(newDir),
                       0.001f, 10000.0f, 0.0f, 0xFF, OPTIX_RAY_FLAG_NONE,
                       RAY_TYPE_INDIRECT, RAY_TYPE_COUNT, RAY_TYPE_INDIRECT,
                       gp0, gp1, gp2, gp3, gp4, gp5);

            float3 result = make_float3(
                __uint_as_float(gp0), __uint_as_float(gp1), __uint_as_float(gp2));

            // Tint refracted rays by base color (colored glass)
            if (!reflected)
                result = result * baseColorRGB;

            result = result + emissive;
            setPayloadColor(clamp(result, 0.0f, 1000.0f));
            setPayloadHitDistance(optixGetRayTmax());
            setPayloadInstanceId(instanceId);
            setPayloadDepth(depth);
            return;
        }
        // depth exceeded → fall through to opaque shading
    }

    float3 lightDir = make_float3(0.0f, 0.0f, 0.0f);
    float contribLuminance = 0.0f;
    float3 Lo = computeStandardDirectLighting(
        hitPos, geomNormal, shadingNormal, V,
        baseColorRGB, metallic, roughness,
        clearcoat, clearcoatRoughness,
        sheenColor, sheenRoughness, seed,
        &lightDir, &contribLuminance);

    Lo = Lo + emissive;

    // Build grid descriptor once, shared by guiding and training.
    // Hierarchical lookup finds the exact cell; trilinear interpolation smooths
    // sampling boundaries. Training uses exact cell only (no trilinear) to
    // prevent spatial coupling that creates large-scale mode boundaries.
    SparsePathGuideDescriptorDevice grid;
    unsigned int exactCellIdx = 0xFFFFFFFFu;   // exact cell (for training + seeding)
    unsigned int guideCellIdx = 0xFFFFFFFFu;   // stochastic trilinear cell for sampling
    unsigned int guideFoundLevel = 0;
    bool hasValidGuide = false;
    TrilinearInfo allNeighbors;                // all trilinear neighbors (for training via stochastic box filter)
    allNeighbors.weightSum = 0.0f;
    TrilinearInfo guideTrilinear;              // filtered: cells with valid lobes (for guiding)
    guideTrilinear.weightSum = 0.0f;

    if (params.path_guide_num_levels > 0) {
        grid = buildGridDescriptor();

        if (params.path_guide_morton_codes != nullptr &&
            params.path_guide_data != nullptr &&
            params.path_guide_level_offsets != nullptr) {
            exactCellIdx = hierarchicalCellLookup(
                grid, hitPos.x, hitPos.y, hitPos.z,
                params.path_guide_max_level,
                params.path_guide_min_level,
                &guideFoundLevel);
            if (exactCellIdx != 0xFFFFFFFFu) {
                // Compute trilinear neighbors for smooth sampling/PDF at boundaries.
                // Kept in outer scope: reused for training (stochastic box filter).
                allNeighbors = computeTrilinearNeighbors(
                    grid, hitPos.x, hitPos.y, hitPos.z, guideFoundLevel);
                // Filter for guiding (cells with valid vMF lobes only)
                guideTrilinear = filterTrilinearByValidLobes(grid, allNeighbors);
                // Stochastically select one cell for sampling
                float interpRand = randomFloat(seed);
                guideCellIdx = stochasticSelectCell(guideTrilinear, interpRand);
                if (guideCellIdx != 0xFFFFFFFFu) {
                    hasValidGuide = true;
                }
            }
        }
    }

    // Add path-guided indirect lighting (one-bounce GI with learned direction distribution)
    // Secondary bounces use __closesthit__radiance_bounce via RAY_TYPE_INDIRECT, so no recursion guard needed
    float3 indirectBounceDir = make_float3(0.0f, 0.0f, 0.0f);
    float indirectBounceLuminance = 0.0f;
    float indirectIncidentLuminance = 0.0f;
    float indirectSamplingPdf = 1.0f;
    {
        float3 guidedIndirect = computeGuidedIndirectLighting(
            grid, guideCellIdx, hasValidGuide,
            guideTrilinear,
            hitPos, geomNormal, shadingNormal, V,
            baseColorRGB, metallic, roughness, seed,
            &indirectBounceDir, &indirectBounceLuminance, &indirectIncidentLuminance,
            &indirectSamplingPdf);
        Lo = Lo + guidedIndirect;
    }

    if (params.selected_instance_id == instanceId) {
        float3 selectionTint = make_float3(1.1f, 1.15f, 1.4f);
        Lo = Lo * selectionTint;
        float rim = 1.0f - fmaxf(0.0f, dot(shadingNormal, V));
        rim = powf(rim, 2.0f);
        Lo = Lo + make_float3(0.2f, 0.4f, 1.0f) * rim * 0.5f;
    }

    Lo = clamp(Lo, 0.0f, 1000.0f);

    // Path guiding: collect occupancy and train direction distributions.
    //
    // Two key techniques from the literature ensure correct online training:
    //
    // 1. Importance-weighted training (Li / p(ω)): Makes each deposit an
    //    unbiased estimate of ∫Li(ω)dω regardless of sampling strategy.
    //
    // 2. Stochastic box filter via trilinear neighbor reuse (Müller 2017):
    //    Stochastically selecting one trilinear neighbor (weighted by
    //    trilinear weights) is equivalent to jittering by ±0.5 cells.
    //    Reuses the already-computed allNeighbors, avoiding an extra
    //    hierarchical binary search through global memory.
    if (params.path_guide_num_levels > 0) {
        if (exactCellIdx == 0xFFFFFFFFu) {
            // No cell exists — seed one at start level so all geometry gets coverage.
            if (params.path_guide_staging_buffer != nullptr && params.path_guide_staging_count != nullptr &&
                params.path_guide_staging_capacity > 0) {
                float nx, ny, nz;
                worldToNormalized(grid, hitPos.x, hitPos.y, hitPos.z, nx, ny, nz);
                int ix, iy, iz;
                normalizedToCell(nx, ny, nz, params.path_guide_start_level, grid, ix, iy, iz);

                PathGuideStagingDevice staging = {};
                staging.buffer = params.path_guide_staging_buffer;
                staging.count = params.path_guide_staging_count;
                staging.capacity = params.path_guide_staging_capacity;
                pathGuideStagingAppend(staging, params.path_guide_start_level, ix, iy, iz);
            }
        } else if (indirectIncidentLuminance > 1e-6f) {
            // Stochastic box filter via trilinear neighbor reuse (Müller 2017):
            // The trilinear weights from computeTrilinearNeighbors are exactly
            // the box filter kernel overlap probabilities. Stochastically selecting
            // one neighbor (weighted by trilinear weights) is mathematically
            // equivalent to jittering the position by ±0.5 cells and looking up
            // the cell — but avoids an extra hierarchical binary search.
            unsigned int trainCellIdx = stochasticSelectCell(allNeighbors, randomFloat(seed));

            // Fall back to exact cell if no trilinear neighbor found
            if (trainCellIdx == 0xFFFFFFFFu) trainCellIdx = exactCellIdx;

            float* cell = sparseCellDataPtr(grid, trainCellIdx);
            if (cell != nullptr) {
                float blen = sqrtf(indirectBounceDir.x*indirectBounceDir.x +
                                   indirectBounceDir.y*indirectBounceDir.y +
                                   indirectBounceDir.z*indirectBounceDir.z);
                if (blen > 1e-6f) {
                    float bx = indirectBounceDir.x / blen;
                    float by = indirectBounceDir.y / blen;
                    float bz = indirectBounceDir.z / blen;
                    // Importance-weighted training: Li / p(ω)
                    // Clamp to prevent extreme weights from corrupting cell sums.
                    float trainWeight = indirectIncidentLuminance / fmaxf(indirectSamplingPdf, 1e-4f);
                    trainWeight = fminf(trainWeight, 500.0f);
                    pathGuideTrainCell(cell, bx, by, bz, trainWeight, params.frame_index);
                }
            }
        }
    }

    setPayloadColor(Lo);
    setPayloadHitDistance(optixGetRayTmax());
    setPayloadInstanceId(instanceId);
}

//------------------------------------------------------------------------------
// Lightweight Closest Hit for Secondary Bounces (indirect rays)
// No path guiding, no training, no debug stats — just material eval + direct lighting
//------------------------------------------------------------------------------

extern "C" __global__ void __closesthit__radiance_bounce() {
    const HitGroupData* sbtData = reinterpret_cast<HitGroupData*>(optixGetSbtDataPointer());
    const GpuMaterial& material = sbtData->material;
    const unsigned int instanceId = optixGetInstanceId();
    const unsigned int primIdx = optixGetPrimitiveIndex();
    const float2 barycentrics = optixGetTriangleBarycentrics();
    const float baryU = barycentrics.x;
    const float baryV = barycentrics.y;
    const float baryW = 1.0f - baryU - baryV;

    const GpuVertex* vertices = reinterpret_cast<const GpuVertex*>(params.vertex_buffers[instanceId]);
    const unsigned int* indices = reinterpret_cast<const unsigned int*>(params.index_buffers[instanceId]);

    const unsigned int i0 = indices[primIdx * 3 + 0];
    const unsigned int i1 = indices[primIdx * 3 + 1];
    const unsigned int i2 = indices[primIdx * 3 + 2];

    const GpuVertex& vert0 = vertices[i0];
    const GpuVertex& vert1 = vertices[i1];
    const GpuVertex& vert2 = vertices[i2];

    float3 objectPos = baryW * vert0.position + baryU * vert1.position + baryV * vert2.position;
    float3 hitPos = optixTransformPointFromObjectToWorldSpace(objectPos);

    float3 objectNormal = normalize(baryW * vert0.normal + baryU * vert1.normal + baryV * vert2.normal);
    float3 geomNormal = normalize(optixTransformNormalFromObjectToWorldSpace(objectNormal));

    float4 tangent = baryW * vert0.tangent + baryU * vert1.tangent + baryV * vert2.tangent;
    float3 objectTangent = make_float3(tangent.x, tangent.y, tangent.z);
    float3 worldTangent = normalize(optixTransformVectorFromObjectToWorldSpace(objectTangent));
    float bitangentSign = tangent.w;

    float2 texCoord = make_float2(
        baryW * vert0.u + baryU * vert1.u + baryV * vert2.u,
        baryW * vert0.v + baryU * vert1.v + baryV * vert2.v
    );

    float rayDistance = optixGetRayTmax();
    float texLOD = calculateTextureLOD(rayDistance);

    // Material evaluation (same as primary but no clearcoat/sheen for perf)
    float4 baseColorTex = sampleTexture(material.baseColorTex, texCoord, make_float4(1.0f, 1.0f, 1.0f, 1.0f), texLOD);
    float4 baseColor = make_float4(
        material.baseColor.x * baseColorTex.x,
        material.baseColor.y * baseColorTex.y,
        material.baseColor.z * baseColorTex.z,
        material.baseColor.w * baseColorTex.w
    );

    float metallic = material.metallic;
    float roughness = material.roughness;
    if (material.metallicRoughnessTex != 0) {
        float4 mrSample = tex2DLod<float4>(material.metallicRoughnessTex, texCoord.x, texCoord.y, texLOD);
        roughness = material.roughness * mrSample.y;
        metallic = material.metallic * mrSample.z;
    }
    roughness = fmaxf(roughness, 0.04f);

    float3 emissive = material.emissive;
    if (material.emissiveTex != 0) {
        float4 emissiveTex = tex2DLod<float4>(material.emissiveTex, texCoord.x, texCoord.y, texLOD);
        emissive = make_float3(
            material.emissive.x * emissiveTex.x,
            material.emissive.y * emissiveTex.y,
            material.emissive.z * emissiveTex.z
        );
    }

    float3 shadingNormal = geomNormal;
    if (material.normalTex != 0) {
        float4 normalSample = tex2DLod<float4>(material.normalTex, texCoord.x, texCoord.y, texLOD);
        float3 tangentNormal = unpackNormal(normalSample);
        shadingNormal = applyNormalMap(tangentNormal, geomNormal, worldTangent, bitangentSign);
    }

    float3 rayDir = optixGetWorldRayDirection();
    float3 origGeomNormal = geomNormal;
    float3 origShadingNormal = shadingNormal;
    if (material.doubleSided && dot(geomNormal, rayDir) > 0.0f) {
        shadingNormal = -shadingNormal;
        geomNormal = -geomNormal;
    }

    // V = incoming ray direction reversed (correct for bounce rays)
    float3 V = -normalize(rayDir);
    float3 baseColorRGB = make_float3(baseColor.x, baseColor.y, baseColor.z);

    // Generate seed for random sampling (moved before glass block)
    unsigned int pixelIdx = optixGetLaunchIndex().y * params.width + optixGetLaunchIndex().x;
    unsigned int dirHash = __float_as_uint(rayDir.x) ^ __float_as_uint(rayDir.y) ^ __float_as_uint(rayDir.z);
    unsigned int seed = pixelIdx ^ (params.frame_index * 0x9E3779B9u) ^ dirHash;

    // ── Glass / Transmission ──
    if (material.transmission > 0.0f) {
        unsigned int depth = getPayloadDepth();
        if (depth < params.max_bounce_depth) {
            float3 I = rayDir;
            // Use original (pre-double-sided-flip) geometric normal for entering/exiting
            bool entering = dot(origGeomNormal, I) < 0.0f;

            float eta = entering ? (1.0f / material.ior) : material.ior;
            float3 N  = entering ? origShadingNormal : -origShadingNormal;
            float3 gN = entering ? origGeomNormal    : -origGeomNormal;

            float F = fresnelDielectric(dot(N, -I), eta);

            float3 newDir;
            bool reflected;
            if (randomFloat(seed) < F) {
                newDir = reflect(I, N);
                reflected = true;
            } else {
                if (!refract(I, N, eta, newDir)) {
                    newDir = reflect(I, N);  // total internal reflection
                    reflected = true;
                } else {
                    reflected = false;
                }
            }

            // Glass bounces stay on lightweight indirect path
            float3 origin = hitPos + (reflected ? gN : -gN) * 0.001f;
            unsigned int gp0, gp1, gp2, gp3, gp4, gp5;
            gp0 = gp1 = gp2 = __float_as_uint(0.0f);
            gp3 = __float_as_uint(0.0f);
            gp4 = 0xFFFFFFFFu;
            gp5 = depth + 1;

            optixTrace(params.scene_handle, origin, normalize(newDir),
                       0.001f, 10000.0f, 0.0f, 0xFF, OPTIX_RAY_FLAG_NONE,
                       RAY_TYPE_INDIRECT, RAY_TYPE_COUNT, RAY_TYPE_INDIRECT,
                       gp0, gp1, gp2, gp3, gp4, gp5);

            float3 result = make_float3(
                __uint_as_float(gp0), __uint_as_float(gp1), __uint_as_float(gp2));

            // Tint refracted rays by base color (colored glass)
            if (!reflected)
                result = result * baseColorRGB;

            result = result + emissive;
            setPayloadColor(clamp(result, 0.0f, 1000.0f));
            setPayloadHitDistance(optixGetRayTmax());
            setPayloadInstanceId(instanceId);
            setPayloadDepth(depth);
            return;
        }
        // depth exceeded → fall through to opaque shading
    }

    // Direct lighting only (no indirect bounce from here — avoids recursion)
    float3 lightDir = make_float3(0.0f, 0.0f, 0.0f);
    float contribLuminance = 0.0f;
    float3 Lo = computeStandardDirectLighting(
        hitPos, geomNormal, shadingNormal, V,
        baseColorRGB, metallic, roughness,
        material.clearcoat, material.clearcoatRoughness,
        material.sheenColor, material.sheenRoughness, seed,
        &lightDir, &contribLuminance);

    Lo = Lo + emissive;
    Lo = clamp(Lo, 0.0f, 1000.0f);

    // Train path guide at bounce hits too — multi-bounce light transport paths
    // contribute to the guiding distribution, improving convergence for complex
    // indirect lighting (Müller 2017: every path vertex should train the guide).
    if (params.path_guide_num_levels > 0 && contribLuminance > 1e-6f) {
        SparsePathGuideDescriptorDevice grid = buildGridDescriptor();
        // Incoming ray direction = outgoing light direction for training
        float3 inRay = -normalize(rayDir);
        float blen = length(inRay);
        if (blen > 1e-6f) {
            pathGuideTrainAtomic(grid, hitPos.x, hitPos.y, hitPos.z,
                inRay.x, inRay.y, inRay.z, contribLuminance,
                params.frame_index,
                params.path_guide_min_level, params.path_guide_max_level);
        }
    }

    setPayloadColor(Lo);
    setPayloadHitDistance(optixGetRayTmax());
    setPayloadInstanceId(instanceId);
}

extern "C" __global__ void __closesthit__shadow() {
    optixSetPayload_0(1);
}
