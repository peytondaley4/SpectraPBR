#include <optix.h>
#include "gpu_types.h"
#include "shared_device.h"
#include "brdf.h"

//------------------------------------------------------------------------------
// Phase 3: PBR Closest Hit Shader
//
// Implements:
// - Full GGX microfacet BRDF
// - Normal mapping
// - Multiple light types (point, directional, area)
// - Shadow ray tracing
// - Quality modes (Fast/Balanced/High/Accurate)
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// Shadow Ray Tracing
//------------------------------------------------------------------------------

__forceinline__ __device__ bool traceShadowRay(
    const float3& origin,
    const float3& normal,
    const float3& direction,
    float tmax)
{
    // Robust self-intersection avoidance:
    // Offset along both normal (to get off surface) and ray direction
    // Use larger epsilon for better robustness
    const float normalEps = 0.0001f;
    const float rayEps = 0.001f;

    // Offset in normal direction (ensure we're on the correct side of surface)
    float NdotD = dot(normal, direction);
    float3 offsetNormal = (NdotD > 0.0f) ? normal : -normal;
    float3 offsetOrigin = origin + offsetNormal * normalEps;

    // Shadow ray uses a simple 0/1 payload: 0 = visible, 1 = occluded
    unsigned int occluded = 0;

    // Ensure tmax is valid
    float safeTmax = fmaxf(tmax - rayEps, rayEps * 2.0f);

    optixTrace(
        params.scene_handle,
        offsetOrigin,
        direction,
        rayEps,                                     // tmin - start slightly away from origin
        safeTmax,                                   // tmax
        0.0f,                                       // rayTime
        0xFF,                                       // visibility mask
        OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT |
        OPTIX_RAY_FLAG_DISABLE_ANYHIT,              // For opaque shadows; enable anyhit for alpha
        RAY_TYPE_SHADOW,                            // SBT offset
        RAY_TYPE_COUNT,                             // SBT stride
        RAY_TYPE_SHADOW,                            // miss SBT index
        occluded
    );

    return occluded == 0;  // Return true if light is visible
}

//------------------------------------------------------------------------------
// Mipmap LOD Calculation
//------------------------------------------------------------------------------

// Calculate texture LOD based on ray distance and screen pixel size
// This provides automatic mipmap selection for distant objects
__forceinline__ __device__ float calculateTextureLOD(float rayDistance) {
    // Calculate the size of a pixel in world space at distance 1
    // pixelAngle ≈ 2 * tan(fovY/2) / screenHeight
    float tanHalfFov = tanf(params.camera.fovY * 0.5f);
    float pixelWorldSize = (2.0f * tanHalfFov) / static_cast<float>(params.height);

    // At the hit distance, a pixel covers this much world space
    float footprint = rayDistance * pixelWorldSize;

    // LOD = log2(footprint * texelsPerUnit)
    // We use a base scale factor assuming ~1 texel per world unit at LOD 0
    // Adjust the 1.0f multiplier if textures are denser/sparser
    float lod = log2f(fmaxf(1.0f, footprint * 1.0f));

    // Clamp to valid range (0 to ~12 for most textures)
    return fminf(fmaxf(lod, 0.0f), 12.0f);
}

//------------------------------------------------------------------------------
// Texture Sampling Helpers (with LOD support)
//------------------------------------------------------------------------------

__forceinline__ __device__ float4 sampleTexture(cudaTextureObject_t tex, float2 uv, float4 fallback, float lod) {
    if (tex != 0) {
        return tex2DLod<float4>(tex, uv.x, uv.y, lod);
    }
    return fallback;
}

// Legacy version without LOD (for compatibility)
__forceinline__ __device__ float4 sampleTexture(cudaTextureObject_t tex, float2 uv, float4 fallback) {
    if (tex != 0) {
        return tex2D<float4>(tex, uv.x, uv.y);
    }
    return fallback;
}

__forceinline__ __device__ float sampleTextureChannel(cudaTextureObject_t tex, float2 uv, int channel, float fallback, float lod) {
    if (tex != 0) {
        float4 sample = tex2DLod<float4>(tex, uv.x, uv.y, lod);
        switch (channel) {
            case 0: return sample.x;
            case 1: return sample.y;
            case 2: return sample.z;
            case 3: return sample.w;
        }
    }
    return fallback;
}

// Legacy version without LOD
__forceinline__ __device__ float sampleTextureChannel(cudaTextureObject_t tex, float2 uv, int channel, float fallback) {
    if (tex != 0) {
        float4 sample = tex2D<float4>(tex, uv.x, uv.y);
        switch (channel) {
            case 0: return sample.x;
            case 1: return sample.y;
            case 2: return sample.z;
            case 3: return sample.w;
        }
    }
    return fallback;
}

//------------------------------------------------------------------------------
// Main Closest Hit Shader
//------------------------------------------------------------------------------

extern "C" __global__ void __closesthit__radiance() {
    // Get SBT data
    const HitGroupData* sbtData = reinterpret_cast<HitGroupData*>(optixGetSbtDataPointer());
    const GpuMaterial& material = sbtData->material;

    // Get instance ID for buffer lookup
    const unsigned int instanceId = optixGetInstanceId();

    // Get primitive index and barycentrics
    const unsigned int primIdx = optixGetPrimitiveIndex();
    const float2 barycentrics = optixGetTriangleBarycentrics();
    const float baryU = barycentrics.x;
    const float baryV = barycentrics.y;
    const float baryW = 1.0f - baryU - baryV;

    // Get vertex/index buffers
    const GpuVertex* vertices = reinterpret_cast<const GpuVertex*>(params.vertex_buffers[instanceId]);
    const unsigned int* indices = reinterpret_cast<const unsigned int*>(params.index_buffers[instanceId]);

    // Get triangle vertices
    const unsigned int i0 = indices[primIdx * 3 + 0];
    const unsigned int i1 = indices[primIdx * 3 + 1];
    const unsigned int i2 = indices[primIdx * 3 + 2];

    const GpuVertex& vert0 = vertices[i0];
    const GpuVertex& vert1 = vertices[i1];
    const GpuVertex& vert2 = vertices[i2];

    // Interpolate position in object space, then transform to world space
    float3 objectPos = baryW * vert0.position + baryU * vert1.position + baryV * vert2.position;
    float3 hitPos = optixTransformPointFromObjectToWorldSpace(objectPos);

    // Interpolate normal in object space, then transform to world space
    float3 objectNormal = normalize(baryW * vert0.normal + baryU * vert1.normal + baryV * vert2.normal);
    float3 geomNormal = normalize(optixTransformNormalFromObjectToWorldSpace(objectNormal));

    // Interpolate tangent in object space, then transform to world space
    float4 tangent = baryW * vert0.tangent + baryU * vert1.tangent + baryV * vert2.tangent;
    float3 objectTangent = make_float3(tangent.x, tangent.y, tangent.z);
    float3 worldTangent = normalize(optixTransformVectorFromObjectToWorldSpace(objectTangent));
    float bitangentSign = tangent.w;

    // Interpolate texture coordinates
    float2 texCoord = make_float2(
        baryW * vert0.u + baryU * vert1.u + baryV * vert2.u,
        baryW * vert0.v + baryU * vert1.v + baryV * vert2.v
    );

    //--------------------------------------------------------------------------
    // Calculate Mipmap LOD based on ray distance
    //--------------------------------------------------------------------------
    float rayDistance = optixGetRayTmax();
    float texLOD = calculateTextureLOD(rayDistance);

    //--------------------------------------------------------------------------
    // Sample Material Textures (with automatic mipmap selection)
    //--------------------------------------------------------------------------

    // Base color (sRGB texture, factor in linear)
    float4 baseColorTex = sampleTexture(material.baseColorTex, texCoord, make_float4(1.0f, 1.0f, 1.0f, 1.0f), texLOD);
    float4 baseColor = make_float4(
        material.baseColor.x * baseColorTex.x,
        material.baseColor.y * baseColorTex.y,
        material.baseColor.z * baseColorTex.z,
        material.baseColor.w * baseColorTex.w
    );

    // Metallic-roughness (packed: G = roughness, B = metallic)
    float metallic = material.metallic;
    float roughness = material.roughness;
    if (material.metallicRoughnessTex != 0) {
        float4 mrSample = tex2DLod<float4>(material.metallicRoughnessTex, texCoord.x, texCoord.y, texLOD);
        roughness = material.roughness * mrSample.y;  // G channel
        metallic = material.metallic * mrSample.z;    // B channel
    }

    // Clamp roughness to avoid division issues
    roughness = fmaxf(roughness, 0.04f);

    // Emissive
    float3 emissive = material.emissive;
    if (material.emissiveTex != 0) {
        float4 emissiveTex = tex2DLod<float4>(material.emissiveTex, texCoord.x, texCoord.y, texLOD);
        emissive = make_float3(
            material.emissive.x * emissiveTex.x,
            material.emissive.y * emissiveTex.y,
            material.emissive.z * emissiveTex.z
        );
    }

    //--------------------------------------------------------------------------
    // Normal Mapping
    //--------------------------------------------------------------------------

    float3 shadingNormal = geomNormal;
    if (material.normalTex != 0) {
        float4 normalSample = tex2DLod<float4>(material.normalTex, texCoord.x, texCoord.y, texLOD);
        float3 tangentNormal = unpackNormal(normalSample);
        shadingNormal = applyNormalMap(tangentNormal, geomNormal, worldTangent, bitangentSign);
    }

    // Handle double-sided materials
    float3 rayDir = optixGetWorldRayDirection();
    if (material.doubleSided && dot(shadingNormal, rayDir) > 0.0f) {
        shadingNormal = -shadingNormal;
        geomNormal = -geomNormal;
    }

    //--------------------------------------------------------------------------
    // View Direction
    //--------------------------------------------------------------------------

    float3 V = normalize(params.camera.position - hitPos);
    float NdotV = fmaxf(dot(shadingNormal, V), BRDF_EPSILON);

    //--------------------------------------------------------------------------
    // Clearcoat Parameters (if enabled)
    //--------------------------------------------------------------------------

    float clearcoat = material.clearcoat;
    float clearcoatRoughness = material.clearcoatRoughness;
    if (material.clearcoatTex != 0) {
        clearcoat *= tex2DLod<float4>(material.clearcoatTex, texCoord.x, texCoord.y, texLOD).x;
    }
    if (material.clearcoatRoughnessTex != 0) {
        clearcoatRoughness *= tex2DLod<float4>(material.clearcoatRoughnessTex, texCoord.x, texCoord.y, texLOD).y;
    }

    //--------------------------------------------------------------------------
    // Sheen Parameters (if enabled)
    //--------------------------------------------------------------------------

    float3 sheenColor = material.sheenColor;
    float sheenRoughness = material.sheenRoughness;
    if (material.sheenColorTex != 0) {
        float4 sheenTex = tex2DLod<float4>(material.sheenColorTex, texCoord.x, texCoord.y, texLOD);
        sheenColor = make_float3(
            sheenColor.x * sheenTex.x,
            sheenColor.y * sheenTex.y,
            sheenColor.z * sheenTex.z
        );
    }
    if (material.sheenRoughnessTex != 0) {
        sheenRoughness *= tex2DLod<float4>(material.sheenRoughnessTex, texCoord.x, texCoord.y, texLOD).w;
    }

    //--------------------------------------------------------------------------
    // Transmission Parameters (KHR_materials_transmission)
    //--------------------------------------------------------------------------

    float transmission = material.transmission;
    if (material.transmissionTex != 0) {
        transmission *= tex2DLod<float4>(material.transmissionTex, texCoord.x, texCoord.y, texLOD).x;
    }

    //--------------------------------------------------------------------------
    // Specular Parameters (KHR_materials_specular)
    //--------------------------------------------------------------------------

    float specularFactor = material.specularFactor;
    float3 specularColorFactor = material.specularColorFactor;
    if (material.specularTex != 0) {
        specularFactor *= tex2DLod<float4>(material.specularTex, texCoord.x, texCoord.y, texLOD).w;  // A channel
    }
    if (material.specularColorTex != 0) {
        float4 specColorTex = tex2DLod<float4>(material.specularColorTex, texCoord.x, texCoord.y, texLOD);
        specularColorFactor = make_float3(
            specularColorFactor.x * specColorTex.x,
            specularColorFactor.y * specColorTex.y,
            specularColorFactor.z * specColorTex.z
        );
    }

    //--------------------------------------------------------------------------
    // Occlusion (affects ambient lighting)
    //--------------------------------------------------------------------------

    float ao = 1.0f;  // Default: no occlusion
    if (material.occlusionTex != 0) {
        ao = tex2DLod<float4>(material.occlusionTex, texCoord.x, texCoord.y, texLOD).x;  // R channel
        ao = 1.0f + material.occlusionStrength * (ao - 1.0f);  // Apply strength
    }

    //--------------------------------------------------------------------------
    // Accumulate Lighting
    //--------------------------------------------------------------------------

    float3 Lo = make_float3(0.0f, 0.0f, 0.0f);
    float3 baseColorRGB = make_float3(baseColor.x, baseColor.y, baseColor.z);

    // Process Point Lights
    for (unsigned int i = 0; i < params.point_light_count; ++i) {
        const GpuPointLight& light = params.point_lights[i];

        float3 lightVec = light.position - hitPos;
        float distance = length(lightVec);
        float3 L = lightVec / distance;

        float NdotL = dot(shadingNormal, L);
        if (NdotL <= 0.0f) continue;

        // Shadow ray
        bool visible = traceShadowRay(hitPos, geomNormal, L, distance);
        if (!visible) continue;

        // Attenuation (inverse square falloff)
        float attenuation = 1.0f / (distance * distance);

        // BRDF evaluation based on quality mode
        float3 brdf;
        if (params.quality_mode == QUALITY_FAST) {
            // Fast: Simple Lambertian + basic specular
            brdf = evaluateLambertian(baseColorRGB);
        } else if (clearcoat > 0.0f && params.quality_mode >= QUALITY_HIGH) {
            // High/Accurate with clearcoat
            brdf = evaluateBRDF_Clearcoat(V, L, shadingNormal, baseColorRGB, metallic, roughness, clearcoat, clearcoatRoughness);
        } else {
            // Balanced: Full GGX
            brdf = evaluateGGX_BRDF(V, L, shadingNormal, baseColorRGB, metallic, roughness);
        }

        // Add sheen contribution (for cloth)
        if (sheenColor.x > 0.0f || sheenColor.y > 0.0f || sheenColor.z > 0.0f) {
            brdf = brdf + evaluateSheen(V, L, shadingNormal, sheenColor, sheenRoughness);
        }

        Lo = Lo + brdf * light.intensity * attenuation * NdotL;
    }

    // Process Directional Lights
    for (unsigned int i = 0; i < params.directional_light_count; ++i) {
        const GpuDirectionalLight& light = params.directional_lights[i];

        // Light direction points toward the light source
        float3 L = -normalize(light.direction);

        float NdotL = dot(shadingNormal, L);
        if (NdotL <= 0.0f) continue;

        // Shadow ray (infinite distance for directional lights)
        bool visible = traceShadowRay(hitPos, geomNormal, L, 10000.0f);
        if (!visible) continue;

        // BRDF evaluation
        float3 brdf;
        if (params.quality_mode == QUALITY_FAST) {
            brdf = evaluateLambertian(baseColorRGB);
        } else if (clearcoat > 0.0f && params.quality_mode >= QUALITY_HIGH) {
            brdf = evaluateBRDF_Clearcoat(V, L, shadingNormal, baseColorRGB, metallic, roughness, clearcoat, clearcoatRoughness);
        } else {
            brdf = evaluateGGX_BRDF(V, L, shadingNormal, baseColorRGB, metallic, roughness);
        }

        // Add sheen
        if (sheenColor.x > 0.0f || sheenColor.y > 0.0f || sheenColor.z > 0.0f) {
            brdf = brdf + evaluateSheen(V, L, shadingNormal, sheenColor, sheenRoughness);
        }

        Lo = Lo + brdf * light.irradiance * NdotL;
    }

    // Process Area Lights (simplified: treat center as point light with area factor)
    for (unsigned int i = 0; i < params.area_light_count; ++i) {
        const GpuAreaLight& light = params.area_lights[i];

        float3 lightVec = light.position - hitPos;
        float distance = length(lightVec);
        float3 L = lightVec / distance;

        float NdotL = dot(shadingNormal, L);
        if (NdotL <= 0.0f) continue;

        // Check if we're on the front side of the area light
        float lightNdotL = -dot(light.normal, L);
        if (lightNdotL <= 0.0f) continue;

        // Shadow ray
        bool visible = traceShadowRay(hitPos, geomNormal, L, distance);
        if (!visible) continue;

        // Area light attenuation includes projected solid angle
        float attenuation = lightNdotL / (distance * distance);

        // BRDF evaluation
        float3 brdf;
        if (params.quality_mode == QUALITY_FAST) {
            brdf = evaluateLambertian(baseColorRGB);
        } else {
            brdf = evaluateGGX_BRDF(V, L, shadingNormal, baseColorRGB, metallic, roughness);
        }

        Lo = Lo + brdf * light.emission * light.area * attenuation * NdotL;
    }

    //--------------------------------------------------------------------------
    // Environment Map Direct Lighting (Importance Sampled)
    //--------------------------------------------------------------------------

    if (params.environment_map != 0 && params.env_conditional_cdf != 0 && params.env_marginal_cdf != 0) {
        // Generate random numbers for environment sampling
        // Use pixel position, frame index, AND ray direction to ensure unique samples per SPP
        unsigned int pixelIdx = optixGetLaunchIndex().y * params.width + optixGetLaunchIndex().x;
        float3 rayDir = optixGetWorldRayDirection();
        unsigned int dirHash = __float_as_uint(rayDir.x) ^ __float_as_uint(rayDir.y) ^ __float_as_uint(rayDir.z);
        unsigned int seed = pixelIdx ^ (params.frame_index * 0x9E3779B9u) ^ dirHash;
        
        // Generate two random numbers
        float xi1 = randomFloat(seed);
        float xi2 = randomFloat(seed);
        
        // Sample direction from environment map using importance sampling
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
        
        // Only contribute if light direction is on correct hemisphere
        if (NdotL > 0.0f && envPdf > 0.0f) {
            // Check visibility (shadow ray toward environment)
            bool visible = traceShadowRay(hitPos, geomNormal, L, 10000.0f);
            
            if (visible) {
                // Get environment radiance
                float3 envRadiance = sampleEnvironmentRadiance(L, params.environment_map, params.environment_intensity);
                
                // Evaluate BRDF
                float3 brdf;
                if (params.quality_mode == QUALITY_FAST) {
                    brdf = evaluateLambertian(baseColorRGB);
                } else if (clearcoat > 0.0f && params.quality_mode >= QUALITY_HIGH) {
                    brdf = evaluateBRDF_Clearcoat(V, L, shadingNormal, baseColorRGB, metallic, roughness, clearcoat, clearcoatRoughness);
                } else {
                    brdf = evaluateGGX_BRDF(V, L, shadingNormal, baseColorRGB, metallic, roughness);
                }
                
                // Add sheen
                if (sheenColor.x > 0.0f || sheenColor.y > 0.0f || sheenColor.z > 0.0f) {
                    brdf = brdf + evaluateSheen(V, L, shadingNormal, sheenColor, sheenRoughness);
                }
                
                // Monte Carlo estimator: (brdf * Li * cos_theta) / pdf
                // Apply MIS weight (balance heuristic) - for single-strategy sampling, weight = 1
                float3 envContrib = brdf * envRadiance * NdotL / envPdf;
                
                // Soft clamp to reduce fireflies while preserving energy
                // Using a higher threshold to avoid biasing the accumulated result
                float maxVal = fmaxf(fmaxf(envContrib.x, envContrib.y), envContrib.z);
                if (maxVal > 100.0f) {
                    envContrib = envContrib * (100.0f / maxVal);
                }
                
                Lo = Lo + envContrib;
            }
        }
    }
    // Fallback ambient if no environment map and no lights
    else if (params.environment_map == 0 && 
             params.point_light_count == 0 && 
             params.directional_light_count == 0 && 
             params.area_light_count == 0) {
        // Fallback: Simple ambient lighting (modulated by AO)
        float ambient = 0.1f * ao;  // Apply ambient occlusion
        Lo = Lo + baseColorRGB * ambient;

        // Add simple directional light for visibility
        float3 defaultLightDir = normalize(make_float3(1.0f, 1.0f, 1.0f));
        float NdotL = fmaxf(0.0f, dot(shadingNormal, defaultLightDir));
        Lo = Lo + baseColorRGB * NdotL * 0.8f;
    }

    //--------------------------------------------------------------------------
    // Add Emissive
    //--------------------------------------------------------------------------

    Lo = Lo + emissive;

    //--------------------------------------------------------------------------
    // Selection Highlighting
    //--------------------------------------------------------------------------

    if (params.selected_instance_id == instanceId) {
        // Add blue tint to selected object
        float3 selectionTint = make_float3(1.1f, 1.15f, 1.4f);  // Subtle blue tint
        Lo = Lo * selectionTint;

        // Add rim highlight effect
        float rim = 1.0f - fmaxf(0.0f, dot(shadingNormal, V));
        rim = powf(rim, 2.0f);
        Lo = Lo + make_float3(0.2f, 0.4f, 1.0f) * rim * 0.5f;  // Blue rim light
    }

    //--------------------------------------------------------------------------
    // Output
    //--------------------------------------------------------------------------

    // Clamp to prevent fireflies
    Lo = clamp(Lo, 0.0f, 100.0f);

    setPayloadColor(Lo);
    setPayloadHitDistance(optixGetRayTmax());
    setPayloadInstanceId(instanceId);
}

//------------------------------------------------------------------------------
// Shadow Ray Closest Hit
// This is called when a shadow ray hits geometry - marks the ray as occluded
//------------------------------------------------------------------------------

extern "C" __global__ void __closesthit__shadow() {
    // Set payload to indicate occlusion (1 = blocked)
    optixSetPayload_0(1);
}
