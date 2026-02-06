#include <optix.h>
#include "gpu_types.h"
#include "shared_device.h"
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
    unsigned int& seed)
{
    float3 Lo = make_float3(0.0f, 0.0f, 0.0f);

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
                    Lo = Lo + brdf * light.intensity * att * NdotL * (visible ? (params.total_light_luminance / lum) : 0.0f);
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
                        Lo = Lo + brdf * light.irradiance * NdotL * (visible ? (params.total_light_luminance / lum) : 0.0f);
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
                        Lo = Lo + brdf * light.emission * light.area * att * NdotL * (visible ? (params.total_light_luminance / lum) : 0.0f);
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

                Lo = Lo + brdf * envRadiance * NdotL / envPdf;
            }
        }
    }
    // Fallback ambient when no lights
    else if (params.environment_map == 0 &&
             params.point_light_count == 0 &&
             params.directional_light_count == 0 &&
             params.area_light_count == 0) {
        float ambient = 0.1f;
        Lo = Lo + baseColor * ambient;
        float3 defaultLightDir = normalize(make_float3(1.0f, 1.0f, 1.0f));
        float NdotL = fmaxf(0.0f, dot(shadingNormal, defaultLightDir));
        Lo = Lo + baseColor * NdotL * 0.8f;
    }

    return Lo;
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

    float3 Lo = computeStandardDirectLighting(
        hitPos, geomNormal, shadingNormal, V,
        baseColorRGB, metallic, roughness,
        clearcoat, clearcoatRoughness,
        sheenColor, sheenRoughness, seed);

    Lo = Lo + emissive;

    if (params.selected_instance_id == instanceId) {
        float3 selectionTint = make_float3(1.1f, 1.15f, 1.4f);
        Lo = Lo * selectionTint;
        float rim = 1.0f - fmaxf(0.0f, dot(shadingNormal, V));
        rim = powf(rim, 2.0f);
        Lo = Lo + make_float3(0.2f, 0.4f, 1.0f) * rim * 0.5f;
    }

    Lo = clamp(Lo, 0.0f, 1000.0f);

    setPayloadColor(Lo);
    setPayloadHitDistance(optixGetRayTmax());
    setPayloadInstanceId(instanceId);
}

extern "C" __global__ void __closesthit__shadow() {
    optixSetPayload_0(1);
}
