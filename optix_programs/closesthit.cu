#include <optix.h>
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

// Quality mode constants
constexpr unsigned int QUALITY_FAST     = 0;
constexpr unsigned int QUALITY_BALANCED = 1;
constexpr unsigned int QUALITY_HIGH     = 2;
constexpr unsigned int QUALITY_ACCURATE = 3;

// Ray type constants
constexpr unsigned int RAY_TYPE_RADIANCE = 0;
constexpr unsigned int RAY_TYPE_SHADOW   = 1;
constexpr unsigned int RAY_TYPE_COUNT    = 2;

// Vertex structure matching GpuVertex from shared_types.h
struct GpuVertex {
    float3 position;
    float u;
    float3 normal;
    float v;
    float4 tangent;
};

// Material structure matching GpuMaterial from shared_types.h
struct GpuMaterial {
    float4 baseColor;
    float metallic;
    float roughness;
    float2 _pad0;
    float3 emissive;
    float _pad1;

    cudaTextureObject_t baseColorTex;
    cudaTextureObject_t normalTex;
    cudaTextureObject_t metallicRoughnessTex;
    cudaTextureObject_t emissiveTex;

    // KHR_materials_transmission
    float transmission;
    float ior;
    float2 _pad2;
    cudaTextureObject_t transmissionTex;

    // KHR_materials_volume
    float3 attenuationColor;
    float attenuationDistance;
    float thickness;
    float3 _pad3;

    // KHR_materials_clearcoat
    float clearcoat;
    float clearcoatRoughness;
    float2 _pad4;
    cudaTextureObject_t clearcoatTex;
    cudaTextureObject_t clearcoatRoughnessTex;
    cudaTextureObject_t clearcoatNormalTex;

    // KHR_materials_sheen
    float3 sheenColor;
    float sheenRoughness;
    cudaTextureObject_t sheenColorTex;
    cudaTextureObject_t sheenRoughnessTex;

    // KHR_materials_specular
    float specularFactor;
    float3 _pad5;
    float3 specularColorFactor;
    float _pad6;
    cudaTextureObject_t specularTex;
    cudaTextureObject_t specularColorTex;

    // Occlusion
    cudaTextureObject_t occlusionTex;
    float occlusionStrength;

    // Alpha settings
    unsigned int alphaMode;
    float alphaCutoff;
    unsigned int doubleSided;
};

// Light structures
struct GpuPointLight {
    float3 position;
    float radius;
    float3 intensity;
    float _pad;
};

struct GpuDirectionalLight {
    float3 direction;
    float angularDiameter;
    float3 irradiance;
    float _pad;
};

struct GpuAreaLight {
    float3 position;
    float _pad0;
    float3 normal;
    float _pad1;
    float3 tangent;
    float _pad2;
    float3 emission;
    float area;
    float2 size;
    float2 _pad3;
};

// HitGroupRecord data
struct HitGroupData {
    GpuMaterial material;
    unsigned int geometryIndex;
};

// Launch parameters
extern "C" {
__constant__ struct {
    float4* output_buffer;
    float4* accumulation_buffer;
    unsigned int width;
    unsigned int height;
    unsigned int frame_index;
    unsigned int accumulated_frames;

    struct {
        float3 position;
        float _pad0;
        float3 forward;
        float _pad1;
        float3 right;
        float _pad2;
        float3 up;
        float _pad3;
        float fovY;
        float aspectRatio;
        float nearPlane;
        float farPlane;
    } camera;

    OptixTraversableHandle scene_handle;

    CUdeviceptr* vertex_buffers;
    CUdeviceptr* index_buffers;

    unsigned int* instance_material_indices;

    // Lighting
    GpuPointLight* point_lights;
    unsigned int point_light_count;
    unsigned int _pad_lights0;
    GpuDirectionalLight* directional_lights;
    unsigned int directional_light_count;
    unsigned int _pad_lights1;
    GpuAreaLight* area_lights;
    unsigned int area_light_count;
    unsigned int _pad_lights2;

    cudaTextureObject_t environment_map;
    float environment_intensity;
    float _pad_env;

    unsigned int quality_mode;
    unsigned int random_seed;

    // UI selection (UINT32_MAX = no selection)
    unsigned int selected_instance_id;
    unsigned int _pad_selection;

    // Picking mode
    unsigned int* pick_result;
    unsigned int pick_x;
    unsigned int pick_y;
    unsigned int pick_mode;
} params;
}

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
// Texture Sampling Helpers
//------------------------------------------------------------------------------

__forceinline__ __device__ float4 sampleTexture(cudaTextureObject_t tex, float2 uv, float4 fallback) {
    if (tex != 0) {
        return tex2D<float4>(tex, uv.x, uv.y);
    }
    return fallback;
}

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
    // Sample Material Textures
    //--------------------------------------------------------------------------

    // Base color (sRGB texture, factor in linear)
    float4 baseColorTex = sampleTexture(material.baseColorTex, texCoord, make_float4(1.0f, 1.0f, 1.0f, 1.0f));
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
        float4 mrSample = tex2D<float4>(material.metallicRoughnessTex, texCoord.x, texCoord.y);
        roughness = material.roughness * mrSample.y;  // G channel
        metallic = material.metallic * mrSample.z;    // B channel
    }

    // Clamp roughness to avoid division issues
    roughness = fmaxf(roughness, 0.04f);

    // Emissive
    float3 emissive = material.emissive;
    if (material.emissiveTex != 0) {
        float4 emissiveTex = tex2D<float4>(material.emissiveTex, texCoord.x, texCoord.y);
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
        float4 normalSample = tex2D<float4>(material.normalTex, texCoord.x, texCoord.y);
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
        clearcoat *= tex2D<float4>(material.clearcoatTex, texCoord.x, texCoord.y).x;
    }
    if (material.clearcoatRoughnessTex != 0) {
        clearcoatRoughness *= tex2D<float4>(material.clearcoatRoughnessTex, texCoord.x, texCoord.y).y;
    }

    //--------------------------------------------------------------------------
    // Sheen Parameters (if enabled)
    //--------------------------------------------------------------------------

    float3 sheenColor = material.sheenColor;
    float sheenRoughness = material.sheenRoughness;
    if (material.sheenColorTex != 0) {
        float4 sheenTex = tex2D<float4>(material.sheenColorTex, texCoord.x, texCoord.y);
        sheenColor = make_float3(
            sheenColor.x * sheenTex.x,
            sheenColor.y * sheenTex.y,
            sheenColor.z * sheenTex.z
        );
    }
    if (material.sheenRoughnessTex != 0) {
        sheenRoughness *= tex2D<float4>(material.sheenRoughnessTex, texCoord.x, texCoord.y).w;
    }

    //--------------------------------------------------------------------------
    // Transmission Parameters (KHR_materials_transmission)
    //--------------------------------------------------------------------------

    float transmission = material.transmission;
    if (material.transmissionTex != 0) {
        transmission *= tex2D<float4>(material.transmissionTex, texCoord.x, texCoord.y).x;
    }

    //--------------------------------------------------------------------------
    // Specular Parameters (KHR_materials_specular)
    //--------------------------------------------------------------------------

    float specularFactor = material.specularFactor;
    float3 specularColorFactor = material.specularColorFactor;
    if (material.specularTex != 0) {
        specularFactor *= tex2D<float4>(material.specularTex, texCoord.x, texCoord.y).w;  // A channel
    }
    if (material.specularColorTex != 0) {
        float4 specColorTex = tex2D<float4>(material.specularColorTex, texCoord.x, texCoord.y);
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
        ao = tex2D<float4>(material.occlusionTex, texCoord.x, texCoord.y).x;  // R channel
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
    // Environment Lighting (if no lights defined, use simple ambient)
    //--------------------------------------------------------------------------

    if (params.point_light_count == 0 && params.directional_light_count == 0 && params.area_light_count == 0) {
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
