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
    float tanHalfFov = tanf(params.camera.fovY * 0.5f);
    float pixelWorldSize = (2.0f * tanHalfFov) / static_cast<float>(params.height);
    float footprint = rayDistance * pixelWorldSize;
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
                float distance = length(lightVec);
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
                    float3 lightVec = light.position - hitPos;
                    float distance = length(lightVec);
                    float3 L = lightVec / distance;
                    float NdotL = dot(shadingNormal, L);
                    float lightNdotL = -dot(light.normal, L);
                    if (NdotL > 0.0f && lightNdotL > 0.0f) {
                        bool visible = traceShadowRay(hitPos, geomNormal, L, distance);
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
// Implements proper one-sample MIS:
// 1. With probability alpha, sample from path guide (vMF)
// 2. With probability (1-alpha), sample from BSDF (GGX)
// 3. Weight using balance heuristic: w = p_chosen / (alpha*p_guide + (1-alpha)*p_bsdf)
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

// Compute BRDF PDF for GGX importance sampling
__forceinline__ __device__ float computeGGXPdf(
    const float3& V, const float3& L, const float3& N, float roughness)
{
    float3 H = normalize(V + L);
    float NdotH = fmaxf(dot(N, H), 0.001f);
    float VdotH = fmaxf(dot(V, H), 0.001f);
    float alpha = roughness * roughness;
    float alpha2 = alpha * alpha;
    float denom = NdotH * NdotH * (alpha2 - 1.0f) + 1.0f;
    float D = alpha2 / (3.14159265f * denom * denom);
    return D * NdotH / (4.0f * VdotH);
}

// Sample direction from GGX distribution (VNDF sampling)
__forceinline__ __device__ float3 sampleGGXDirection(
    const float3& V, const float3& N, float roughness,
    float u1, float u2)
{
    // Build tangent frame
    float3 up = (fabsf(N.y) < 0.999f) ? make_float3(0.0f, 1.0f, 0.0f) : make_float3(1.0f, 0.0f, 0.0f);
    float3 T = normalize(cross(up, N));
    float3 B = cross(N, T);

    // GGX importance sampling (simplified, samples microfacet normal)
    float alpha = roughness * roughness;
    float phi = 2.0f * 3.14159265f * u1;
    float cosTheta = sqrtf((1.0f - u2) / (1.0f + (alpha * alpha - 1.0f) * u2));
    float sinTheta = sqrtf(1.0f - cosTheta * cosTheta);

    // Microfacet normal in tangent space
    float3 H_local = make_float3(sinTheta * cosf(phi), sinTheta * sinf(phi), cosTheta);

    // Transform to world space
    float3 H = T * H_local.x + B * H_local.y + N * H_local.z;

    // Reflect V around H to get L
    float VdotH = dot(V, H);
    float3 L = 2.0f * VdotH * H - V;

    return normalize(L);
}

__forceinline__ __device__ float3 computeGuidedIndirectLighting(
    const float3& hitPos,
    const float3& geomNormal,
    const float3& shadingNormal,
    const float3& V,
    const float3& baseColor,
    float metallic,
    float roughness,
    unsigned int& seed,
    float3* out_bounceDir = nullptr,
    float* out_bounceLuminance = nullptr)
{
    if (out_bounceDir) *out_bounceDir = make_float3(0.0f, 0.0f, 0.0f);
    if (out_bounceLuminance) *out_bounceLuminance = 0.0f;
    // Early out if guiding is disabled
    if (params.path_guide_enabled == 0) {
        return make_float3(0.0f, 0.0f, 0.0f);
    }

    incrementGuideStat(GUIDE_STAT_ATTEMPTS);

    // MIS blend factor: probability of using guide vs BSDF
    float alpha = params.path_guide_mis_weight;  // e.g., 0.5 = equal probability

    // Build grid descriptor
    SparsePathGuideDescriptorDevice grid = {};
    grid.morton_codes = params.path_guide_morton_codes;
    grid.data = params.path_guide_data;
    grid.level_offsets = params.path_guide_level_offsets;
    grid.num_levels = params.path_guide_num_levels;
    grid.entry_stride = params.path_guide_entry_stride;
    grid.base_resolution = params.path_guide_base_resolution;
    grid.per_level_scale = params.path_guide_per_level_scale;
    grid.bounds_min[0] = params.path_guide_bounds_min[0];
    grid.bounds_min[1] = params.path_guide_bounds_min[1];
    grid.bounds_min[2] = params.path_guide_bounds_min[2];
    grid.bounds_max[0] = params.path_guide_bounds_max[0];
    grid.bounds_max[1] = params.path_guide_bounds_max[1];
    grid.bounds_max[2] = params.path_guide_bounds_max[2];

    // Try to find a cell for this position
    unsigned int foundLevel = 0;
    unsigned int cellIdx = 0xFFFFFFFFu;
    bool hasValidGuide = false;

    if (params.path_guide_morton_codes != nullptr &&
        params.path_guide_data != nullptr &&
        params.path_guide_level_offsets != nullptr) {

        cellIdx = hierarchicalCellLookup(
            grid, hitPos.x, hitPos.y, hitPos.z,
            params.path_guide_max_level,
            params.path_guide_min_level,
            &foundLevel);

        if (cellIdx != 0xFFFFFFFFu) {
            incrementGuideStat(GUIDE_STAT_CELL_FOUND);
            // Check if cell has valid lobes (kappa > 0)
            float* cell = sparseCellDataPtr(grid, cellIdx);
            if (cell != nullptr && (cell[2] > 1e-6f || cell[5] > 1e-6f)) {
                hasValidGuide = true;
                incrementGuideStat(GUIDE_STAT_VALID_LOBE);
            }
        }
    }

    // If no valid guide, fall back to pure BSDF sampling for indirect lighting
    if (!hasValidGuide) {
        float u1 = randomFloat(seed);
        float u2 = randomFloat(seed);
        float3 L = sampleGGXDirection(V, shadingNormal, roughness, u1, u2);
        float NdotL = dot(shadingNormal, L);
        if (NdotL <= 0.0f) return make_float3(0.0f, 0.0f, 0.0f);

        float brdfPdf = computeGGXPdf(V, L, shadingNormal, roughness);
        if (brdfPdf <= 1e-8f) return make_float3(0.0f, 0.0f, 0.0f);

        const float rayEps = 0.001f;
        float NdotD = dot(geomNormal, L);
        float3 offsetNormal = (NdotD > 0.0f) ? geomNormal : -geomNormal;
        float3 offsetOrigin = hitPos + offsetNormal * rayEps;

        unsigned int p0, p1, p2, p3, p4;
        p0 = __float_as_uint(0.0f);
        p1 = __float_as_uint(0.0f);
        p2 = __float_as_uint(0.0f);
        p3 = __float_as_uint(0.0f);
        p4 = 0xFFFFFFFFu;

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
            p0, p1, p2, p3, p4
        );

        float3 Li = make_float3(__uint_as_float(p0), __uint_as_float(p1), __uint_as_float(p2));
        float3 brdf = evaluateGGX_BRDF(V, L, shadingNormal, baseColor, metallic, roughness);
        float3 contrib = brdf * Li * NdotL / brdfPdf;
        float3 result = clamp(contrib, 0.0f, 100.0f);
        float lum = 0.2126f * result.x + 0.7152f * result.y + 0.0722f * result.z;
        if (out_bounceDir && lum > 1e-6f) *out_bounceDir = L;
        if (out_bounceLuminance) *out_bounceLuminance = lum;
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
        guidePdf = pathGuidePdfDirection(grid, cellIdx, guideX, guideY, guideZ);
        brdfPdf = computeGGXPdf(V, L, shadingNormal, roughness);

    } else {
        // Sample from BSDF (GGX distribution)
        incrementGuideStat(GUIDE_STAT_BSDF_SAMPLED);
        float u1 = randomFloat(seed);
        float u2 = randomFloat(seed);

        L = sampleGGXDirection(V, shadingNormal, roughness, u1, u2);
        brdfPdf = computeGGXPdf(V, L, shadingNormal, roughness);
        guidePdf = pathGuidePdfDirection(grid, cellIdx, L.x, L.y, L.z);
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

    unsigned int p0, p1, p2, p3, p4;
    p0 = __float_as_uint(0.0f);
    p1 = __float_as_uint(0.0f);
    p2 = __float_as_uint(0.0f);
    p3 = __float_as_uint(0.0f);
    p4 = 0xFFFFFFFFu;

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
        p0, p1, p2, p3, p4
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
    if (out_bounceDir && lum > 1e-6f) *out_bounceDir = L;
    if (out_bounceLuminance) *out_bounceLuminance = lum;
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

    float3 rayDir = optixGetWorldRayDirection();
    if (material.doubleSided && dot(shadingNormal, rayDir) > 0.0f) {
        shadingNormal = -shadingNormal;
        geomNormal = -geomNormal;
    }

    float3 V = normalize(params.camera.position - hitPos);

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

    // Generate seed for random sampling
    unsigned int pixelIdx = optixGetLaunchIndex().y * params.width + optixGetLaunchIndex().x;
    unsigned int dirHash = __float_as_uint(rayDir.x) ^ __float_as_uint(rayDir.y) ^ __float_as_uint(rayDir.z);
    unsigned int seed = pixelIdx ^ (params.frame_index * 0x9E3779B9u) ^ dirHash;

    float3 lightDir = make_float3(0.0f, 0.0f, 0.0f);
    float contribLuminance = 0.0f;
    float3 Lo = computeStandardDirectLighting(
        hitPos, geomNormal, shadingNormal, V,
        baseColorRGB, metallic, roughness,
        clearcoat, clearcoatRoughness,
        sheenColor, sheenRoughness, seed,
        &lightDir, &contribLuminance);

    Lo = Lo + emissive;

    // Add path-guided indirect lighting (one-bounce GI with learned direction distribution)
    // Secondary bounces use __closesthit__radiance_bounce via RAY_TYPE_INDIRECT, so no recursion guard needed
    float3 indirectBounceDir = make_float3(0.0f, 0.0f, 0.0f);
    float indirectBounceLuminance = 0.0f;
    {
        float3 guidedIndirect = computeGuidedIndirectLighting(
            hitPos, geomNormal, shadingNormal, V,
            baseColorRGB, metallic, roughness, seed,
            &indirectBounceDir, &indirectBounceLuminance);
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

    // Path guiding: collect occupancy for debug visualization and train direction distributions
    // Reference: Müller et al., "Practical Path Guiding for Efficient Light-Transport Simulation", EGSR 2017
    if (params.path_guide_num_levels > 0) {
        SparsePathGuideDescriptorDevice grid = {};
        grid.morton_codes = params.path_guide_morton_codes;
        grid.data = params.path_guide_data;
        grid.level_offsets = params.path_guide_level_offsets;
        grid.num_levels = params.path_guide_num_levels;
        grid.entry_stride = params.path_guide_entry_stride;
        grid.base_resolution = params.path_guide_base_resolution;
        grid.per_level_scale = params.path_guide_per_level_scale;
        grid.bounds_min[0] = params.path_guide_bounds_min[0];
        grid.bounds_min[1] = params.path_guide_bounds_min[1];
        grid.bounds_min[2] = params.path_guide_bounds_min[2];
        grid.bounds_max[0] = params.path_guide_bounds_max[0];
        grid.bounds_max[1] = params.path_guide_bounds_max[1];
        grid.bounds_max[2] = params.path_guide_bounds_max[2];

        // NOTE: Debug-level staging was removed. It wrote 2M entries per frame
        // (one per pixel) into the staging buffer, inflating buildFromStaging's
        // sort from ~500K to 2.5M entries and adding ~150ms of CPU overhead.
        // The wireframe visualization now uses cells created by cell seeding
        // and training, which cover all visible surfaces via generateEdgeVerticesAllLevels().

        // Cell seeding: ensure every visible surface gets a cell, even in shadow.
        // Decoupled from training — cells are created for ALL hits (stochastically),
        // while training data is only collected where there's light to learn from.
        if (params.path_guide_training_probability > 0.0f &&
            params.path_guide_staging_buffer != nullptr && params.path_guide_staging_count != nullptr &&
            params.path_guide_staging_capacity > 0 &&
            randomFloat(seed) < params.path_guide_training_probability) {

            unsigned int foundLevel = 0;
            unsigned int cellIdx = hierarchicalCellLookup(
                grid, hitPos.x, hitPos.y, hitPos.z,
                params.path_guide_max_level,
                params.path_guide_min_level,
                &foundLevel);

            if (cellIdx == 0xFFFFFFFFu) {
                // No cell exists — seed one at start level so all geometry gets coverage
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
        }

        // Training: stochastic subsampling per Müller et al. 2017 §4.1
        // Train from direct lighting AND/OR indirect bounce — either source is valid.
        if (params.path_guide_training_buffer != nullptr && params.path_guide_training_count != nullptr &&
            params.path_guide_training_capacity > 0 &&
            (contribLuminance > 0.0f || indirectBounceLuminance > 1e-6f) &&
            randomFloat(seed) < params.path_guide_training_probability) {

            // Find finest existing cell for this position
            unsigned int foundLevel = 0;
            unsigned int cellIdx = hierarchicalCellLookup(
                grid, hitPos.x, hitPos.y, hitPos.z,
                params.path_guide_max_level,
                params.path_guide_min_level,
                &foundLevel);

            float nx, ny, nz;
            worldToNormalized(grid, hitPos.x, hitPos.y, hitPos.z, nx, ny, nz);
            int ix, iy, iz;

            unsigned int trainLevel;
            if (cellIdx != 0xFFFFFFFFu) {
                trainLevel = foundLevel;
                normalizedToCell(nx, ny, nz, trainLevel, grid, ix, iy, iz);
            } else {
                trainLevel = params.path_guide_start_level;
                normalizedToCell(nx, ny, nz, trainLevel, grid, ix, iy, iz);
            }

            PathGuideTrainingStagingDevice trainStaging = {};
            trainStaging.buffer = params.path_guide_training_buffer;
            trainStaging.count = params.path_guide_training_count;
            trainStaging.capacity = params.path_guide_training_capacity;

            // Train from direct lighting direction
            if (contribLuminance > 0.0f) {
                float len = sqrtf(lightDir.x*lightDir.x + lightDir.y*lightDir.y + lightDir.z*lightDir.z);
                if (len > 1e-6f) {
                    float ux = lightDir.x / len, uy = lightDir.y / len, uz = lightDir.z / len;
                    pathGuideTrainingAppend(trainStaging, trainLevel, ix, iy, iz, ux, uy, uz, contribLuminance, params.frame_index);
                }
            }

            // Train from indirect bounce direction
            float blen = sqrtf(indirectBounceDir.x*indirectBounceDir.x +
                               indirectBounceDir.y*indirectBounceDir.y +
                               indirectBounceDir.z*indirectBounceDir.z);
            if (blen > 1e-6f && indirectBounceLuminance > 1e-6f) {
                float bx = indirectBounceDir.x / blen;
                float by = indirectBounceDir.y / blen;
                float bz = indirectBounceDir.z / blen;
                float bounceNdotL = fmaxf(dot(shadingNormal, make_float3(bx, by, bz)), 0.0f);
                float bounceWeight = indirectBounceLuminance * bounceNdotL;
                if (bounceWeight > 1e-6f) {
                    pathGuideTrainingAppend(trainStaging, trainLevel, ix, iy, iz,
                        bx, by, bz, bounceWeight, params.frame_index);
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
    if (material.doubleSided && dot(shadingNormal, rayDir) > 0.0f) {
        shadingNormal = -shadingNormal;
        geomNormal = -geomNormal;
    }

    float3 V = normalize(params.camera.position - hitPos);
    float3 baseColorRGB = make_float3(baseColor.x, baseColor.y, baseColor.z);

    // Generate seed for random sampling
    unsigned int pixelIdx = optixGetLaunchIndex().y * params.width + optixGetLaunchIndex().x;
    unsigned int dirHash = __float_as_uint(rayDir.x) ^ __float_as_uint(rayDir.y) ^ __float_as_uint(rayDir.z);
    unsigned int seed = pixelIdx ^ (params.frame_index * 0x9E3779B9u) ^ dirHash;

    // Direct lighting only (no indirect bounce, no path guiding)
    float3 Lo = computeStandardDirectLighting(
        hitPos, geomNormal, shadingNormal, V,
        baseColorRGB, metallic, roughness,
        material.clearcoat, material.clearcoatRoughness,
        material.sheenColor, material.sheenRoughness, seed);

    Lo = Lo + emissive;
    Lo = clamp(Lo, 0.0f, 1000.0f);

    setPayloadColor(Lo);
    setPayloadHitDistance(optixGetRayTmax());
    setPayloadInstanceId(instanceId);
}

extern "C" __global__ void __closesthit__shadow() {
    optixSetPayload_0(1);
}
