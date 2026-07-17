#include <optix.h>
#include "gpu_types.h"
#include "shared_device.h"
#include "path_guide_grid_device.h"
#include "brdf.h"

//------------------------------------------------------------------------------
// Iterative Path Tracer (single raygen loop)
//
// Closest-hit only reports hit info; ALL shading happens here. Compared to the
// previous recursive design (closest-hit tracing from closest-hit, trace depth
// 11) this:
//   - cuts OptiX trace depth to 2 (radiance + shadow), shrinking the
//     continuation stack and register pressure,
//   - removes the duplicated primary/bounce shading code paths,
//   - gives every path vertex (not just the first) guided sampling, NEE,
//     and training,
//   - makes proper NEE <-> BSDF multiple importance sampling possible: the
//     loop knows the sampling PDF of the previous bounce when a ray hits an
//     emissive surface or escapes to the environment.
//
// Estimator summary (per path vertex):
//   - One light sample (NEE) chosen from {point, directional, area, env} by
//     luminance-weighted selection; mesh lights and env are MIS-weighted
//     against the path sampler (balance heuristic), virtual rects and delta
//     lights take full weight (the path sampler cannot reach them).
//   - Emission picked up by the path sampler carries the complementary MIS
//     weight. Together both halves sum to 1 — no double counting.
//   - Direction sampling is one-sample MIS between the path-guide vMF mixture
//     (probability alpha) and a diffuse/specular BSDF mixture (1 - alpha);
//     the combined PDF alpha*p_guide + (1-alpha)*p_bsdf is used directly.
//   - Russian roulette on throughput after a few bounces; division by the
//     survival probability keeps the estimator unbiased.
//
// Path-guide training (Müller et al., EGSR 2017, adapted to online fitting):
// the loop records one entry per vertex and runs a backward pass at path end,
// reconstructing the incident radiance L_o(k+1) each vertex saw along its
// sampled direction. Deposits are importance-weighted (Li / pdf) so training
// is unbiased w.r.t. the sampling distribution that produced the direction.
//------------------------------------------------------------------------------

#define RAY_EPS              0.001f
#define SHADOW_NORMAL_EPS    0.0001f
#define MAX_TRAIN_VERTICES   8
#define TRAIN_WEIGHT_CLAMP   500.0f

// Training deposits are subsampled per path with compensating weight: the
// learned distribution is unchanged in expectation, the contended per-cell
// atomic traffic drops by 1/PG_TRAIN_PROB. The probability also sets how fast
// every cell matures: at 0.25 only a quarter of paths ever train the guide, so
// cells accumulate evidence 4x slower than the renderer samples — the dominant
// limit on how fast the guide subdivides and its mixtures converge (a wide view
// gives each world-cell little traffic, so cells stay coarse/noisy unless the
// camera concentrates rays on them). Raised to 0.5 to halve maturity time at
// the cost of ~2x the training atomic traffic. WEIGHT_SCALE is derived so it
// always stays 1/PROB.
#define PG_TRAIN_PROB        0.5f
#define PG_TRAIN_WEIGHT_SCALE (1.0f / PG_TRAIN_PROB)

// Max share of the continuation budget the guide may claim on a surface the
// BSDF-selection logic considers fully specular (pSpec == 1, i.e. metallic
// metals). Gated by roughness so it only applies to ROUGH metals, where the
// incident-radiance guide is the only variance reduction for caustics /
// emissive-via-glossy transport; smooth metals keep it at ~0 (the guide vMF
// barely overlaps a near-mirror lobe).
#define PG_SPEC_GUIDE_FLOOR  0.5f

// Hermite smoothstep on [e0, e1].
__forceinline__ __device__ float smoothstep01(float e0, float e1, float x) {
    float t = clamp((x - e0) / fmaxf(e1 - e0, 1e-8f), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

// Far distance for secondary rays and unbounded shadow rays. Tied to the
// camera far plane (which the host scales to the scene size) so very large
// scenes don't get clipped by a hardcoded constant.
__forceinline__ __device__ float sceneFarDistance() {
    return fmaxf(10000.0f, params.camera.farPlane);
}

// Debug stat indices (UI panel)
#define GUIDE_STAT_ATTEMPTS     0
#define GUIDE_STAT_CELL_FOUND   1
#define GUIDE_STAT_VALID_LOBE   2
#define GUIDE_STAT_BELOW_HORIZ  3
#define GUIDE_STAT_CONTRIBUTED  4
#define GUIDE_STAT_BSDF_SAMPLED 5

// Stats are SAMPLED from 1/256 of pixels. Counting from every vertex fired
// 3-4 global atomics per vertex on six hot counters (~20M serialized atomics
// per frame at 1080p) — a measurable permanent tax whenever guiding is on.
// The UI panel reads these as ratios, so uniform subsampling preserves them.
__forceinline__ __device__ void incrementGuideStat(unsigned int statIdx) {
    if (params.path_guide_debug_enabled && params.path_guide_debug_stats != nullptr) {
        const uint3 li = optixGetLaunchIndex();
        if (((li.x & 15u) | (li.y & 15u)) == 0u) {
            atomicAdd(&params.path_guide_debug_stats[statIdx], 1u);
        }
    }
}

//------------------------------------------------------------------------------
// Ray casting helpers
//------------------------------------------------------------------------------

__forceinline__ __device__ HitInfo traceRadianceRay(
    const float3& origin, const float3& dir, float tmin, float tmax)
{
    unsigned int p0 = __float_as_uint(-1.0f);
    unsigned int p1 = 0xFFFFFFFFu;
    unsigned int p2 = 0u;
    unsigned int p3 = __float_as_uint(0.0f);
    unsigned int p4 = __float_as_uint(0.0f);

    optixTrace(
        params.scene_handle,
        origin, dir,
        tmin, tmax, 0.0f,
        0xFF,
        OPTIX_RAY_FLAG_NONE,            // anyhit enabled for alpha-masked geometry
        RAY_TYPE_RADIANCE,
        RAY_TYPE_COUNT,
        RAY_TYPE_RADIANCE,
        p0, p1, p2, p3, p4);

    HitInfo hit;
    hit.t = __uint_as_float(p0);
    hit.instanceId = p1;
    hit.primIdx = p2;
    hit.baryU = __uint_as_float(p3);
    hit.baryV = __uint_as_float(p4);
    return hit;
}

// deltaLight: the shadow ray targets a point/directional light — the ONLY
// case where transmissive surfaces pass the ray (see __anyhit__shadow_alpha:
// no path-sampled alternative exists for delta lights, so occluding would
// render glass interiors black). Area/mesh/env shadow rays are occluded by
// glass: the transmitted energy arrives via refracted path transport (the
// caustic), and letting NEE through would both erase the glass shadow and
// double-count on top of the caustic.
__forceinline__ __device__ bool traceShadowRay(
    const float3& origin, const float3& geomNormal, const float3& direction, float tmax,
    bool deltaLight = false)
{
    float NdotD = dot(geomNormal, direction);
    float3 offsetNormal = (NdotD > 0.0f) ? geomNormal : -geomNormal;
    float3 offsetOrigin = origin + offsetNormal * SHADOW_NORMAL_EPS;

    // Init to 1 (occluded). Miss program sets 0 (visible).
    // DISABLE_CLOSESTHIT: no shader runs on a confirmed hit.
    // ENFORCE_ANYHIT: ray flags override the per-GAS DISABLE_ANYHIT, so the
    // shadow anyhit runs even for geometry whose GAS was built opaque — this
    // is what lets transmissive (glass) materials pass shadow rays without a
    // GAS rebuild when a material is toggled to glass at runtime. Materials
    // bound to the no-anyhit shadow hit group (opaque) still block with no
    // program invocation; alpha-masked and transmissive materials carry
    // __anyhit__shadow_alpha (cut-out shadows / transparent glass shadows).
    unsigned int occluded = 1;
    unsigned int isDelta = deltaLight ? 1u : 0u;
    float safeTmax = fmaxf(tmax - RAY_EPS, RAY_EPS * 2.0f);

    optixTrace(
        params.scene_handle,
        offsetOrigin, direction,
        RAY_EPS, safeTmax, 0.0f,
        0xFF,
        OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT |
            OPTIX_RAY_FLAG_ENFORCE_ANYHIT,
        RAY_TYPE_SHADOW,
        RAY_TYPE_COUNT,
        RAY_TYPE_SHADOW,
        occluded, isDelta);

    return occluded == 0;
}

//------------------------------------------------------------------------------
// Surface loading (vertex fetch + material textures) — raygen-side, using the
// per-instance transform buffers (optixTransform* intrinsics are CH-only).
//------------------------------------------------------------------------------

struct Surface {
    float3 pos;
    float3 faceNormal;       // true geometric (face) normal, front side of winding
    float3 geomNormal;       // face normal flipped to oppose the incoming ray (offsets)
    float3 shadingNormal;    // interpolated + normal-mapped, flipped for double-sided
    float3 origShadingNormal;// pre-flip shading normal (glass entering/exiting)
    float3 baseColor;
    float  metallic;
    float  roughness;
    float3 emissive;
    float  clearcoat;
    float  clearcoatRoughness;
    float3 sheenColor;
    float  sheenRoughness;
    float  transmission;
    float  ior;
    float3 attenuationColor;
    float  attenuationDistance;
    unsigned int instanceId;
    unsigned int lightIndex;  // area light index or 0xFFFFFFFF
};

__forceinline__ __device__ float calculateTextureLOD(float footprintDistance) {
    float footprint = footprintDistance * params.pixel_world_size;
    float lod = log2f(fmaxf(1.0f, footprint));
    return fminf(fmaxf(lod, 0.0f), 12.0f);
}

__forceinline__ __device__ float4 sampleTexture(
    cudaTextureObject_t tex, float2 uv, float4 fallback, float lod)
{
    return tex != 0 ? tex2DLod<float4>(tex, uv.x, uv.y, lod) : fallback;
}

__forceinline__ __device__ Surface loadSurface(
    const HitInfo& hit, const float3& rayOrigin, const float3& rayDir,
    float pathFootprintDist)
{
    Surface s;
    s.instanceId = hit.instanceId;

    const GpuVertex* vertices = reinterpret_cast<const GpuVertex*>(params.vertex_buffers[hit.instanceId]);
    const unsigned int* indices = reinterpret_cast<const unsigned int*>(params.index_buffers[hit.instanceId]);
    const float* xf  = params.instance_transforms + (size_t)hit.instanceId * 12;
    const float* nxf = params.instance_normal_transforms + (size_t)hit.instanceId * 12;

    const unsigned int matIdx = params.instance_material_indices[hit.instanceId];
    const GpuMaterial& material = params.materials[matIdx];

    const unsigned int i0 = indices[hit.primIdx * 3 + 0];
    const unsigned int i1 = indices[hit.primIdx * 3 + 1];
    const unsigned int i2 = indices[hit.primIdx * 3 + 2];
    const GpuVertex& v0 = vertices[i0];
    const GpuVertex& v1 = vertices[i1];
    const GpuVertex& v2 = vertices[i2];

    const float bu = hit.baryU, bv = hit.baryV;
    const float bw = 1.0f - bu - bv;

    // Hit position from the ray (avoids a point transform and exactly matches
    // the traversal's notion of the hit).
    s.pos = rayOrigin + rayDir * hit.t;

    // True geometric face normal (object-space cross product through the
    // inverse-transpose). Used for ray offsets, entering/exiting tests, and
    // mesh-light PDFs — more robust than the interpolated vertex normal.
    float3 e1 = v1.position - v0.position;
    float3 e2 = v2.position - v0.position;
    float3 faceN = normalize(transformVector(nxf, cross(e1, e2)));
    s.faceNormal = faceN;

    float3 objectNormal = normalize(bw * v0.normal + bu * v1.normal + bv * v2.normal);
    float3 worldNormal = normalize(transformVector(nxf, objectNormal));

    float4 tangent = bw * v0.tangent + bu * v1.tangent + bv * v2.tangent;
    // Meshes without authored tangents interpolate to ~zero; normalize(0) is
    // NaN (rsqrtf(0)), which would ride through applyNormalMap into the
    // shading normal and get every affected sample discarded by the NaN
    // guard. Detect it and disable normal mapping for the hit instead.
    float3 tangentVec = transformVector(xf, make_float3(tangent.x, tangent.y, tangent.z));
    float tangentLen2 = dot(tangentVec, tangentVec);
    bool hasTangent = tangentLen2 > 1e-12f;
    float3 worldTangent = hasTangent ? tangentVec * rsqrtf(tangentLen2)
                                     : make_float3(1.0f, 0.0f, 0.0f);

    float2 texCoord = make_float2(
        bw * v0.u + bu * v1.u + bv * v2.u,
        bw * v0.v + bu * v1.v + bv * v2.v);

    float texLOD = calculateTextureLOD(pathFootprintDist);

    float4 baseColorTex = sampleTexture(material.baseColorTex, texCoord,
        make_float4(1.0f, 1.0f, 1.0f, 1.0f), texLOD);
    s.baseColor = make_float3(
        material.baseColor.x * baseColorTex.x,
        material.baseColor.y * baseColorTex.y,
        material.baseColor.z * baseColorTex.z);

    s.metallic = material.metallic;
    s.roughness = material.roughness;
    if (material.metallicRoughnessTex != 0) {
        float4 mr = tex2DLod<float4>(material.metallicRoughnessTex, texCoord.x, texCoord.y, texLOD);
        s.roughness = material.roughness * mr.y;
        s.metallic = material.metallic * mr.z;
    }
    s.roughness = fmaxf(s.roughness, 0.04f);

    s.emissive = material.emissive;
    if (material.emissiveTex != 0) {
        float4 em = tex2DLod<float4>(material.emissiveTex, texCoord.x, texCoord.y, texLOD);
        s.emissive = make_float3(
            material.emissive.x * em.x,
            material.emissive.y * em.y,
            material.emissive.z * em.z);
    }

    float3 shadingNormal = worldNormal;
    if (material.normalTex != 0 && hasTangent) {
        float4 normalSample = tex2DLod<float4>(material.normalTex, texCoord.x, texCoord.y, texLOD);
        shadingNormal = applyNormalMap(unpackNormal(normalSample), worldNormal, worldTangent, tangent.w);
    }
    s.origShadingNormal = shadingNormal;

    // Flip normals to face the incoming ray for shading/offsets. The original
    // (unflipped) shading normal is kept for glass entering/exiting logic.
    bool backface = dot(faceN, rayDir) > 0.0f;
    s.geomNormal = backface ? -faceN : faceN;
    if (material.doubleSided && backface) {
        shadingNormal = -shadingNormal;
    }
    // Interpolation/normal mapping can push the shading normal below the view
    // horizon (backfaces of single-sided meshes, grazing silhouettes). The
    // VNDF sampler and pdfBSDFMixture are only valid for dot(N, V) > 0, so
    // fall back to the ray-facing geometric normal there — sampler, PDF, and
    // NEE MIS must all see the same valid frame. The transmission event reads
    // origShadingNormal/faceNormal and resolves its own orientation.
    if (dot(shadingNormal, rayDir) >= 0.0f) {
        shadingNormal = s.geomNormal;
    }
    s.shadingNormal = shadingNormal;

    // Clearcoat only contributes at QUALITY_HIGH and up — skip its texture
    // fetches below that.
    s.clearcoat = material.clearcoat;
    s.clearcoatRoughness = material.clearcoatRoughness;
    if (s.clearcoat > 0.0f && params.quality_mode >= QUALITY_HIGH) {
        if (material.clearcoatTex != 0) {
            s.clearcoat *= tex2DLod<float4>(material.clearcoatTex, texCoord.x, texCoord.y, texLOD).x;
        }
        if (material.clearcoatRoughnessTex != 0) {
            s.clearcoatRoughness *= tex2DLod<float4>(material.clearcoatRoughnessTex, texCoord.x, texCoord.y, texLOD).y;
        }
    }
    // Same floor as base roughness: bounds the coat lobe's peak D (and f/p).
    s.clearcoatRoughness = fmaxf(s.clearcoatRoughness, 0.04f);

    s.sheenColor = material.sheenColor;
    s.sheenRoughness = material.sheenRoughness;
    if (material.sheenColorTex != 0) {
        float4 sheenTex = tex2DLod<float4>(material.sheenColorTex, texCoord.x, texCoord.y, texLOD);
        s.sheenColor = make_float3(
            s.sheenColor.x * sheenTex.x, s.sheenColor.y * sheenTex.y, s.sheenColor.z * sheenTex.z);
    }
    if (material.sheenRoughnessTex != 0) {
        s.sheenRoughness *= tex2DLod<float4>(material.sheenRoughnessTex, texCoord.x, texCoord.y, texLOD).w;
    }

    s.transmission = material.transmission;
    if (material.transmissionTex != 0) {
        s.transmission *= tex2DLod<float4>(material.transmissionTex, texCoord.x, texCoord.y, texLOD).x;
    }
    s.ior = material.ior;
    s.attenuationColor = material.attenuationColor;
    s.attenuationDistance = material.attenuationDistance;

    s.lightIndex = (params.instance_light_indices != nullptr)
        ? params.instance_light_indices[hit.instanceId] : 0xFFFFFFFFu;

    return s;
}

//------------------------------------------------------------------------------
// Light selection + Next Event Estimation
//
// One light sample per vertex. Selection weights MUST match the host total
// (LightManager::syncToGpu): point = lum(intensity), directional =
// lum(irradiance)*10, area = lum(emission)*area; environment weight comes
// precomputed in params.env_selection_weight. The grand total is the
// denominator of every selection probability used in PDFs and MIS weights —
// keep host and device formulas in lockstep or the estimator biases.
//------------------------------------------------------------------------------

// LIGHT_KIND_* constants come from gpu_types.h (shared with the host-built
// alias table).

__forceinline__ __device__ float areaLightSelectionWeight(const GpuAreaLight& l) {
    return luminance3(l.emission) * l.area;
}

// Selection probability (relative to grand total) of the area light owning a
// surface the path sampler just hit — used for the BSDF-side MIS weight.
__forceinline__ __device__ float areaLightSelectionProb(const GpuAreaLight& l, float grandTotal) {
    return (grandTotal > 0.0f) ? areaLightSelectionWeight(l) / grandTotal : 0.0f;
}

// Pick one light: env-vs-scene split, then O(1) Walker/Vose alias-table pick
// among the scene lights (host-built by LightManager::syncToGpu from the same
// selection weights). Replaces the linear luminance-CDF walk — that walk was
// O(N-lights) of divergent global loads per path vertex, dominating the
// shadow ray itself in many-light scenes. Selection probabilities are
// IDENTICAL (P(light) = weight / grandTotal), so every selProb/MIS formula
// downstream is unchanged. Draws its own randoms from the path seed.
__forceinline__ __device__ unsigned int selectLight(
    unsigned int& seed, float grandTotal, unsigned int& outIndex)
{
    // Environment leg: P(env) = env_selection_weight / grandTotal, exactly as
    // the CDF walk realized it (env took the tail past the scene lights).
    if (params.env_selection_weight > 0.0f) {
        if (randomFloat(seed) * grandTotal >= params.total_light_luminance) {
            outIndex = 0;
            return LIGHT_KIND_ENV;
        }
    }

    unsigned int n = params.light_alias_count;
    if (n == 0 || params.light_alias_prob == nullptr) {
        outIndex = 0;
        return LIGHT_KIND_NONE;
    }
    unsigned int bucket = (unsigned int)(randomFloat(seed) * (float)n);
    if (bucket >= n) bucket = n - 1;
    // Fresh random for the accept test — the bucket residual is too
    // quantized to reuse (same reasoning as the env alias sampler).
    unsigned int slot = (randomFloat(seed) < params.light_alias_prob[bucket])
                      ? bucket : params.light_alias_idx[bucket];
    unsigned int packed = params.light_alias_entries[slot];
    outIndex = packed & 0x00FFFFFFu;
    return packed >> 24;
}

// Sample a point uniformly (by area) on a mesh light's triangles.
__forceinline__ __device__ bool sampleMeshLightPoint(
    const GpuAreaLight& light, float xiTri, float xi1, float xi2,
    float3& outPos, float3& outNormal)
{
    if (light.triCount == 0 || params.area_light_tris == nullptr) return false;

    // Binary search the per-light cumulative area CDF stored in tri[0].w
    unsigned int lo = 0, hi = light.triCount - 1;
    while (lo < hi) {
        unsigned int mid = (lo + hi) / 2;
        float cdf = params.area_light_tris[(size_t)(light.triOffset + mid) * 3].w;
        if (cdf < xiTri) lo = mid + 1; else hi = mid;
    }

    const float4* tri = params.area_light_tris + (size_t)(light.triOffset + lo) * 3;
    float3 p0 = make_float3(tri[0].x, tri[0].y, tri[0].z);
    float3 p1 = make_float3(tri[1].x, tri[1].y, tri[1].z);
    float3 p2 = make_float3(tri[2].x, tri[2].y, tri[2].z);

    // Uniform barycentric sample
    float su = sqrtf(xi1);
    float b0 = 1.0f - su;
    float b1 = xi2 * su;
    float b2 = 1.0f - b0 - b1;
    outPos = p0 * b0 + p1 * b1 + p2 * b2;

    float3 n = cross(p1 - p0, p2 - p0);
    float nl = length(n);
    if (nl < 1e-12f) return false;
    outNormal = n / nl;
    return true;
}

// Per-vertex guide context: ONE cell's vMF lobe, cached in registers.
// Müller-style single-cell conditioning — the position is jittered by ±0.5
// cell, the cell at the jittered position becomes THE guide distribution for
// this vertex, and the sampler, the continuation PDF, and the NEE MIS pdf all
// condition on it. The jitter is drawn independently of any direction, so the
// estimator stays unbiased while touching one cell instead of an 8-cell
// trilinear mixture (and zero cell memory per PDF evaluation).
struct GuideLobe {
    float mux, muy, muz;
    float kappa;
    float expNeg2K;   // exp(-2*kappa), cached by the refit kernel
};

__forceinline__ __device__ float guideLobePdf(const GuideLobe& lobe, const float3& L) {
    float cosT = lobe.mux * L.x + lobe.muy * L.y + lobe.muz * L.z;
    return fmaxf(vmfPdfCached(lobe.kappa, lobe.expNeg2K, cosT), 1e-10f);
}

// Next event estimation at a shading vertex. Returns the (selection- and
// MIS-weighted) radiance contribution, NOT multiplied by path throughput.
// pathPdfAt(L) for the MIS weight is alpha*p_guide(L) + (1-alpha)*p_bsdf(L),
// evaluated with the same guide lobe the path sampler uses.
// finalVertex: the path will NOT continue past this vertex (depth cap), so
// the BSDF-side technique does not exist and NEE must take full weight —
// MIS-weighting it down here would lose direct light.
__forceinline__ __device__ float3 sampleDirectLight(
    const Surface& s, const float3& V,
    float pSpec, float pSheen, float pCC, float guideAlpha, const GuideLobe& guideLobe,
    unsigned int& seed, bool finalVertex)
{
    float grandTotal = params.total_light_luminance + params.env_selection_weight;
    if (grandTotal <= 0.0f) return make_float3(0.0f, 0.0f, 0.0f);

    unsigned int lightIdx = 0;
    unsigned int kind = selectLight(seed, grandTotal, lightIdx);
    if (kind == LIGHT_KIND_NONE) return make_float3(0.0f, 0.0f, 0.0f);

    const float3& N = s.shadingNormal;
    float3 contrib = make_float3(0.0f, 0.0f, 0.0f);

    if (kind == LIGHT_KIND_POINT) {
        const GpuPointLight& light = params.point_lights[lightIdx];
        float selProb = luminance3(light.intensity) / grandTotal;
        float3 lightVec = light.position - s.pos;
        float distance = fmaxf(length(lightVec), 0.001f);
        float3 L = lightVec / distance;
        float NdotL = dot(N, L);
        if (NdotL > 0.0f && selProb > 0.0f) {
            if (traceShadowRay(s.pos, s.geomNormal, L, distance, /*deltaLight=*/true)) {
                float3 f = evalPbrBSDF(V, L, N, s.baseColor, s.metallic, s.roughness,
                    s.clearcoat, s.clearcoatRoughness, s.sheenColor, s.sheenRoughness,
                    params.quality_mode, pSpec, pSheen, pCC, nullptr);
                // Delta light: no MIS (path sampler cannot hit it)
                contrib = f * light.intensity * (NdotL / (distance * distance * selProb));
            }
        }
    } else if (kind == LIGHT_KIND_DIR) {
        const GpuDirectionalLight& light = params.directional_lights[lightIdx];
        float selProb = luminance3(light.irradiance) * 10.0f / grandTotal;
        float3 L = -normalize(light.direction);
        float NdotL = dot(N, L);
        if (NdotL > 0.0f && selProb > 0.0f) {
            if (traceShadowRay(s.pos, s.geomNormal, L, sceneFarDistance(), /*deltaLight=*/true)) {
                float3 f = evalPbrBSDF(V, L, N, s.baseColor, s.metallic, s.roughness,
                    s.clearcoat, s.clearcoatRoughness, s.sheenColor, s.sheenRoughness,
                    params.quality_mode, pSpec, pSheen, pCC, nullptr);
                contrib = f * light.irradiance * (NdotL / selProb);
            }
        }
    } else if (kind == LIGHT_KIND_AREA) {
        const GpuAreaLight& light = params.area_lights[lightIdx];
        float selProb = areaLightSelectionWeight(light) / grandTotal;
        if (selProb > 0.0f && light.area > 1e-8f) {
            float3 samplePos, lightNormal;
            bool haveSample;
            bool isMeshLight = (light.triCount > 0);
            if (isMeshLight) {
                haveSample = sampleMeshLightPoint(light,
                    randomFloat(seed), randomFloat(seed), randomFloat(seed),
                    samplePos, lightNormal);
            } else {
                // Virtual rectangle (no geometry in the BVH)
                float3 bitangent = cross(light.normal, light.tangent);
                float u = randomFloat(seed) - 0.5f;
                float v = randomFloat(seed) - 0.5f;
                samplePos = light.position
                    + light.tangent * (u * light.size.x)
                    + bitangent * (v * light.size.y);
                lightNormal = light.normal;
                haveSample = true;
            }

            if (haveSample) {
                float3 lightVec = samplePos - s.pos;
                float distance = fmaxf(length(lightVec), 0.001f);
                float3 L = lightVec / distance;
                float NdotL = dot(N, L);
                // Mesh lights emit from both faces (matching the emissive-on-hit
                // path); virtual rects are one-sided as before.
                float cosLight = isMeshLight ? fabsf(dot(lightNormal, L))
                                             : -dot(lightNormal, L);
                if (NdotL > 0.0f && cosLight > 1e-6f) {
                    if (traceShadowRay(s.pos, s.geomNormal, L, distance)) {
                        // Uniform-area sampling: pdf_area = 1/area, so the
                        // solid-angle pdf is d^2 / (cosLight * area).
                        float pdfLightSA = (distance * distance) / (cosLight * light.area);
                        float pLight = selProb * pdfLightSA;

                        float misW = 1.0f;
                        if (isMeshLight && !finalVertex) {
                            // BSDF/guide sampling can also hit this geometry — MIS.
                            float pPath = pdfBSDFMixture(V, L, N, s.roughness,
                                s.clearcoatRoughness, pSpec, pSheen, pCC);
                            if (guideAlpha > 0.0f) {
                                float pGuide = guideLobePdf(guideLobe, L);
                                pPath = guideAlpha * pGuide + (1.0f - guideAlpha) * pPath;
                            }
                            misW = pLight / (pLight + pPath);
                        }

                        float3 f = evalPbrBSDF(V, L, N, s.baseColor, s.metallic, s.roughness,
                            s.clearcoat, s.clearcoatRoughness, s.sheenColor, s.sheenRoughness,
                            params.quality_mode, pSpec, pSheen, pCC, nullptr);
                        contrib = f * light.emission * (NdotL * misW / pLight);
                    }
                }
            }
        }
    } else if (kind == LIGHT_KIND_ENV) {
        float selProb = params.env_selection_weight / grandTotal;
        if (selProb > 0.0f && params.environment_map != 0 &&
            params.env_alias_prob != nullptr && params.env_pmf != nullptr) {
            float envPdf;
            float3 L = sampleEnvironmentDirection(
                randomFloat(seed), randomFloat(seed), randomFloat(seed), randomFloat(seed),
                params.env_alias_prob, params.env_alias_idx, params.env_pmf,
                params.env_width, params.env_height, envPdf);
            float NdotL = dot(N, L);
            if (NdotL > 0.0f && envPdf > 1e-12f) {
                if (traceShadowRay(s.pos, s.geomNormal, L, sceneFarDistance())) {
                    float pLight = selProb * envPdf;

                    // Path sampler reaches the env on miss — MIS.
                    float misW = 1.0f;
                    if (!finalVertex) {
                        float pPath = pdfBSDFMixture(V, L, N, s.roughness,
                            s.clearcoatRoughness, pSpec, pSheen, pCC);
                        if (guideAlpha > 0.0f) {
                            float pGuide = guideLobePdf(guideLobe, L);
                            pPath = guideAlpha * pGuide + (1.0f - guideAlpha) * pPath;
                        }
                        misW = pLight / (pLight + pPath);
                    }

                    float3 envRadiance = sampleEnvironmentRadiance(L, params.environment_map,
                        params.environment_intensity);
                    float3 f = evalPbrBSDF(V, L, N, s.baseColor, s.metallic, s.roughness,
                        s.clearcoat, s.clearcoatRoughness, s.sheenColor, s.sheenRoughness,
                        params.quality_mode, pSpec, pSheen, pCC, nullptr);
                    contrib = f * envRadiance * (NdotL * misW / pLight);
                }
            }
        }
    }

    return contrib;
}

//------------------------------------------------------------------------------
// Per-vertex training record for the backward pass
//------------------------------------------------------------------------------
struct TrainRecord {
    unsigned int cellIdx;   // deposit target (0xFFFFFFFF = chain-only entry)
    // Exact-cell visit record for the split statistics: exact cell index in
    // bits [0..27], per-axis half-cell signs of the EXACT vertex position in
    // bits [29..31]. 0xFFFFFFFF = no guided lookup at this vertex.
    unsigned int exactPacked;
    float dirX, dirY, dirZ; // sampled continuation direction
    float pdf;              // combined sampling pdf of that direction
    // RGB (not luminance) so the backward recurrence reconstructs the true
    // incident radiance: lum(prod beta_i) != prod lum(beta_i) for chromatic
    // bounces (red wall then green wall over-estimates a direction's importance
    // ~3-4x), which skews the learned guide in colored interreflection — the
    // common photoreal case. Costs ~4 extra floats/record vs the old scalar
    // pair; the deposit is still one scalar (lum of the RGB Lo).
    float3 local;           // RGB radiance emitted from this vertex (emission + NEE)
    float3 beta;            // RGB local throughput factor f*cos/pdf (incl. 1/RR)
    float dist;             // distance to the next vertex along dir (parallax fit);
                            // patched from the next segment's hit.t after the trace
    float relX, relY, relZ; // deposit position within the training cell, [-1,1]
                            // from cell center; feeds the deposit centroid and
                            // spread (parallax pivot + kappa cap) — NOT
                            // subdivision, which uses exactPacked
};

//------------------------------------------------------------------------------
// CONSISTENT firefly clamp: the threshold GROWS with accumulation,
// limit = firefly_clamp * sqrt(accumulated_frames + 1), so early frames get
// firefly suppression but the clamp bias provably vanishes as the image
// accumulates (average bias ~ sum b_i / N -> 0 when per-frame bias b_i -> 0).
//
// A FIXED clamp is not just biased — with guiding it is CELL-STRUCTURED bias:
// near a small bright light essentially every sample exceeds the threshold,
// so every sample is scaled to the same luminance and the pixel CONVERGES to
// a flat clamped value. alpha, lobe fit, and the NEE MIS weights vary per
// guide cell, so the clamp bites differently per cell — the visible artifact
// is noise-free, wrong-brightness, cell-shaped patches at the lit pool that
// accumulation never removes (it converges TO them). Growing the threshold
// restores the invariant the whole engine is built on: any long-accumulated
// image is correct, regardless of proposal quality.
//
// FLT_MAX still disables entirely (QUALITY_ACCURATE: limit = inf).
//------------------------------------------------------------------------------
__forceinline__ __device__ float3 clampContribution(const float3& c) {
    float lum = luminance3(c);
    if (lum <= 0.0f) return c;
    float limit = params.firefly_clamp
                * sqrtf((float)(params.accumulated_frames + 1u));
    if (lum <= limit) return c;
    return c * (limit / lum);
}

//------------------------------------------------------------------------------
// Trace one full path and return its radiance estimate.
//------------------------------------------------------------------------------
__forceinline__ __device__ float3 tracePath(
    float3 rayOrigin, float3 rayDir, unsigned int& seed,
    unsigned int* outFirstInstance, float* outSelectionRim,
    float3* outFirstAlbedo, float3* outFirstNormal)
{
    float3 radiance = make_float3(0.0f, 0.0f, 0.0f);
    float3 throughput = make_float3(1.0f, 1.0f, 1.0f);

    // MIS state from the previous bounce
    bool prevDelta = true;     // camera vertex counts as delta (emission weight 1)
    float prevPdf = 0.0f;      // combined solid-angle pdf of the previous direction

    // Participating-medium state for glass absorption (Beer–Lambert).
    // Single-medium tracking: nested dielectrics are not modeled.
    bool inMedium = false;
    float3 mediumSigma = make_float3(0.0f, 0.0f, 0.0f);

    float pathFootprintDist = 0.0f;   // accumulated distance for texture LOD
    float tmin = params.camera.nearPlane;
    float tmax = params.camera.farPlane;

    // Training records (backward pass at path end)
    const bool guidingActive = (params.path_guide_enabled != 0) &&
                               (params.path_guide_hash_table_size > 0);
    SparsePathGuideDescriptorDevice grid = {};
    if (guidingActive) {
        grid.table.hash_keys = params.path_guide_hash_keys;
        grid.table.hash_values = params.path_guide_hash_values;
        grid.table.hash_table_size = params.path_guide_hash_table_size;
        grid.table.hash_shift = params.path_guide_hash_shift;
        grid.table.cell_keys = params.path_guide_cell_keys;
        grid.table.cell_counter = params.path_guide_cell_counter;
        grid.table.cell_capacity = params.path_guide_cell_capacity;
        grid.table.data = params.path_guide_data;
        grid.table.entry_stride = params.path_guide_entry_stride;
        grid.bounds_min[0] = params.path_guide_bounds_min[0];
        grid.bounds_min[1] = params.path_guide_bounds_min[1];
        grid.bounds_min[2] = params.path_guide_bounds_min[2];
        grid.bounds_max[0] = params.path_guide_bounds_max[0];
        grid.bounds_max[1] = params.path_guide_bounds_max[1];
        grid.bounds_max[2] = params.path_guide_bounds_max[2];
    }
    TrainRecord train[MAX_TRAIN_VERTICES];
    int trainCount = 0;
    // Set when the path continues past a vertex that could not be recorded
    // (train[] saturated): the recurrence below train[last] is then missing
    // that vertex's local/beta, so splicing terminalEnv onto train[last]
    // would attribute radiance to it un-attenuated (orders of magnitude too
    // large after several unrecorded bounces). A broken chain seeds lo = 0
    // instead — the recorded vertices still train on their own local terms.
    bool trainChainBroken = false;
    float3 terminalEnv = make_float3(0.0f, 0.0f, 0.0f);  // (unweighted) RGB radiance entering the last segment
    // Index of the continuation deposit awaiting its parallax distance — the
    // distance to where its radiance came from is the NEXT segment's hit.t,
    // only known after the next trace, so we patch it retroactively.
    int pendingDistRecord = -1;

    const unsigned int maxDepth = params.max_bounce_depth;

    for (unsigned int depth = 0; ; ++depth) {
        HitInfo hit = traceRadianceRay(rayOrigin, rayDir, tmin, tmax);

        // Fill in the previous continuation deposit's parallax distance: hit.t
        // for a surface hit, or a far distance on env miss (distant => no
        // reprojection). Training reads this only for cellIdx != INVALID.
        if (pendingDistRecord >= 0) {
            train[pendingDistRecord].dist = (hit.t >= 0.0f) ? hit.t : sceneFarDistance();
            pendingDistRecord = -1;
        }

        //── Miss: environment (MIS-weighted against env NEE) ────────────────
        if (hit.t < 0.0f) {
            if (params.environment_map != 0) {
                float3 envRadiance = sampleEnvironmentRadiance(
                    rayDir, params.environment_map, params.environment_intensity);
                float w = 1.0f;
                if (!prevDelta && params.env_selection_weight > 0.0f &&
                    params.env_pmf != nullptr) {
                    float grandTotal = params.total_light_luminance + params.env_selection_weight;
                    float selProbEnv = params.env_selection_weight / grandTotal;
                    float pLight = selProbEnv * environmentPdf(rayDir,
                        params.env_pmf, params.env_width, params.env_height);
                    w = prevPdf / (prevPdf + pLight);
                }
                // Depth 0 is deterministic (throughput = 1, w = 1): clamping a
                // zero-variance term has no firefly benefit and only dims the
                // visible background until the growing threshold releases it.
                float3 envContrib = throughput * envRadiance * w;
                radiance = radiance + (depth == 0 ? envContrib
                                                  : clampContribution(envContrib));
                terminalEnv = envRadiance;
            }
            break;
        }

        //── Hit: load surface ───────────────────────────────────────────────
        pathFootprintDist += hit.t;
        Surface s = loadSurface(hit, rayOrigin, rayDir, pathFootprintDist);

        // Absorption through the medium we just crossed (Beer–Lambert)
        if (inMedium) {
            throughput = throughput * make_float3(
                expf(-mediumSigma.x * hit.t),
                expf(-mediumSigma.y * hit.t),
                expf(-mediumSigma.z * hit.t));
        }

        if (depth == 0) {
            if (outFirstInstance) {
                *outFirstInstance = s.instanceId;
                if (s.instanceId == params.selected_instance_id && outSelectionRim) {
                    float3 V0 = -rayDir;
                    float rim = 1.0f - fmaxf(0.0f, dot(s.shadingNormal, V0));
                    *outSelectionRim = rim * rim * 0.5f;
                }
            }
            // Denoiser AOVs: first-hit albedo and CAMERA-space normal — the
            // OptiX denoiser's normal guide layer is specified in camera
            // space; a world-space normal makes the guide inconsistent under
            // camera rotation. Right-handed view basis, looking down -Z.
            if (outFirstAlbedo)
                *outFirstAlbedo = s.baseColor;
            if (outFirstNormal) {
                const float3& n = s.shadingNormal;
                *outFirstNormal = make_float3(
                    dot(n, params.camera.right),
                    dot(n, params.camera.up),
                    -dot(n, params.camera.forward));
            }
        }

        float3 V = -rayDir;

        //── Emission (MIS-weighted against mesh-light NEE) ──────────────────
        float3 emissionContrib = make_float3(0.0f, 0.0f, 0.0f);
        if (s.emissive.x > 0.0f || s.emissive.y > 0.0f || s.emissive.z > 0.0f) {
            float w = 1.0f;
            if (!prevDelta && s.lightIndex != 0xFFFFFFFFu &&
                s.lightIndex < params.area_light_count) {
                const GpuAreaLight& light = params.area_lights[s.lightIndex];
                float grandTotal = params.total_light_luminance + params.env_selection_weight;
                float selProb = areaLightSelectionProb(light, grandTotal);
                float cosLight = fabsf(dot(s.faceNormal, rayDir));
                if (cosLight > 1e-6f && light.area > 1e-8f && selProb > 0.0f) {
                    float pLight = selProb * (hit.t * hit.t) / (cosLight * light.area);
                    w = prevPdf / (prevPdf + pLight);
                }
            }
            emissionContrib = s.emissive * w;
            // Camera-visible emission (depth 0) is deterministic — same
            // zero-variance reasoning as the depth-0 env miss above.
            float3 pathEmission = throughput * emissionContrib;
            radiance = radiance + (depth == 0 ? pathEmission
                                              : clampContribution(pathEmission));
        }

        //── Depth cap ────────────────────────────────────────────────────────
        // Path truncation past max_bounce_depth is biased, so the cap scales
        // with quality (setQualityMode: 8 FAST/BALANCED, 16 HIGH, 32 ACCURATE)
        // — high enough in ACCURATE that RR (from depth 3) terminates paths
        // first and the cap only backstops pathological mirror chains. The cap
        // vertex still gets NEE (with full weight — see sampleDirectLight's
        // finalVertex), only the continuation is dropped.
        const bool atCap = (depth >= maxDepth);

        //── Transmission (stochastic dielectric event, delta lobes) ─────────
        if (!atCap && s.transmission > 0.0f && randomFloat(seed) < s.transmission) {
            bool entering = dot(s.faceNormal, rayDir) < 0.0f;
            // eta = eta_i / eta_t for the side the ray starts on
            float eta = entering ? (1.0f / s.ior) : s.ior;
            float3 N = entering ? s.origShadingNormal : -s.origShadingNormal;
            float3 gN = entering ? s.faceNormal : -s.faceNormal;
            // The side (and hence eta) is already resolved from the geometric
            // face normal above. A shading normal tilted past the view horizon
            // would make fresnelDielectric re-flip the side-resolved eta
            // (spurious TIR) and hand refract() a wrong-side normal — fall
            // back to the geometric normal for this event when that happens.
            if (dot(N, V) <= 0.0f) {
                N = gN;
            }

            float F = fresnelDielectric(dot(N, V), eta);

            float3 newDir;
            bool reflected;
            float3 tint = make_float3(1.0f, 1.0f, 1.0f);
            if (randomFloat(seed) < F) {
                newDir = reflect(rayDir, N);
                reflected = true;
            } else if (!refract(rayDir, N, eta, newDir)) {
                newDir = reflect(rayDir, N);  // total internal reflection
                reflected = true;
            } else {
                reflected = false;
                // Surface tint for colored glass (KHR transmission)
                tint = s.baseColor;
                // Toggle medium state; volume absorption via KHR_materials_volume
                if (entering) {
                    inMedium = true;
                    if (s.attenuationDistance > 1e-6f) {
                        float invD = 1.0f / s.attenuationDistance;
                        mediumSigma = make_float3(
                            -logf(clamp(s.attenuationColor.x, 1e-4f, 1.0f)) * invD,
                            -logf(clamp(s.attenuationColor.y, 1e-4f, 1.0f)) * invD,
                            -logf(clamp(s.attenuationColor.z, 1e-4f, 1.0f)) * invD);
                    } else {
                        mediumSigma = make_float3(0.0f, 0.0f, 0.0f);
                    }
                } else {
                    inMedium = false;
                }
            }

            throughput = throughput * tint;

            // Branch probabilities equal the Fresnel weights, so they cancel —
            // no division needed (perfect importance sampling of the two deltas).
            if (trainCount < MAX_TRAIN_VERTICES) {
                TrainRecord& tr = train[trainCount++];
                tr.cellIdx = 0xFFFFFFFFu;          // delta vertex: chain only
                tr.exactPacked = 0xFFFFFFFFu;      // no guided lookup here
                tr.dirX = newDir.x; tr.dirY = newDir.y; tr.dirZ = newDir.z;
                tr.pdf = 0.0f;
                tr.local = emissionContrib;
                tr.beta = tint;
            } else {
                trainChainBroken = true;  // path continues unrecorded
            }

            rayOrigin = s.pos + (reflected ? gN : -gN) * RAY_EPS;
            rayDir = normalize(newDir);
            tmin = RAY_EPS;
            tmax = sceneFarDistance();
            prevDelta = true;
            prevPdf = 0.0f;
            continue;
        }

        //── Lobe weights for this vertex (shared by sampler, PDF, NEE MIS) ──
        float NdotV = fmaxf(dot(s.shadingNormal, V), BRDF_EPSILON);
        float pSpec = specularSelectProb(NdotV, s.baseColor, s.metallic, params.quality_mode);
        float pSheen = sheenSelectProb(s.sheenColor, s.baseColor, s.metallic, params.quality_mode);
        float pCC = clearcoatSelectProb(NdotV, s.clearcoat, params.quality_mode);

        //── Path-guide lookup (sampling + training context) ─────────────────
        // The guide learns INCIDENT RADIANCE, which is only a good sampling
        // distribution for wide (diffuse-ish) lobes. Scaling the guide
        // probability by the diffuse selection weight (1 - pSpec) makes
        // near-specular surfaces sample pure VNDF: a fixed alpha sent 30% of
        // samples at every metal vertex toward the light source, where the
        // mirror BRDF is ~zero — compounding per bounce, that blackened
        // metallic scenes and traced dead near-zero-throughput paths.
        // pSpec, the jittered cell, kappa, and the maturity confidence ramp are
        // all deterministic given the jitter, so sampler / combined PDF /
        // NEE MIS all derive the same effective alpha and lobe — the
        // estimator stays consistent.
        float guideAlpha = 0.0f;
        unsigned int trainCellIdx = PG_INVALID_CELL;
        // Exact-cell visit record for the SPLIT statistics: cell index (28
        // bits) plus the per-axis half-cell signs of the EXACT (unjittered)
        // vertex position, packed for the TrainRecord. Split statistics must
        // use exact positions — the box-filter jitter would smear the very
        // edge being detected — while lobe training stays on the jittered
        // cell (the guide distribution the sampler conditions on).
        unsigned int exactPacked = 0xFFFFFFFFu;
        float trainRelX = 0.0f, trainRelY = 0.0f, trainRelZ = 0.0f;
        GuideLobe guideLobe = {};
        if (guidingActive && !atCap) {
            incrementGuideStat(GUIDE_STAT_ATTEMPTS);
            unsigned int foundLevel = 0;
            unsigned int exactCellIdx = topDownCellLookup(
                grid, s.pos.x, s.pos.y, s.pos.z,
                params.path_guide_start_level, params.path_guide_max_level, &foundLevel);
            if (exactCellIdx == PG_INVALID_CELL) {
                // First touch of this region: allocate the base-level cell on
                // device (atomicCAS insert, deduplicated by construction —
                // replaces the staging buffer + readback + CPU merge round
                // trip, which took ~10 frames and flooded with duplicates).
                // Bounds-checked inside: out-of-grid vertices never allocate.
                // The fresh cell trains immediately; it cannot be sampled
                // until its first refit fits a lobe.
                exactCellIdx = pathGuideInsertBaseCell(
                    grid, s.pos.x, s.pos.y, s.pos.z, params.path_guide_start_level);
                foundLevel = params.path_guide_start_level;
            }
            if (exactCellIdx != PG_INVALID_CELL) {
                incrementGuideStat(GUIDE_STAT_CELL_FOUND);

                // Pack the exact-cell visit record: index + per-axis half-cell
                // signs of the exact position. Consumed by the backward pass
                // for the (unconditional) visit / half-cell radiance stats
                // that drive subdivision.
                {
                    float exCx, exCy, exCz;
                    pgCellCenter(grid, exactCellIdx, exCx, exCy, exCz);
                    exactPacked = (exactCellIdx & 0x0FFFFFFFu)
                                | ((s.pos.x > exCx) ? (1u << 29) : 0u)
                                | ((s.pos.y > exCy) ? (1u << 30) : 0u)
                                | ((s.pos.z > exCz) ? (1u << 31) : 0u);
                }

                // Stochastic box filter (Müller 2017): jitter the query by
                // ±0.5 cell at the found level and use THE cell at the
                // jittered position for training, sampling, and every PDF.
                // Marginalized over the jitter this is the same box-filtered
                // mixture the old 8-cell trilinear evaluated explicitly.
                float res = sparseResolutionAtLevel(grid, foundLevel);
                float invRes = 1.0f / res;
                float jx = s.pos.x + (randomFloat(seed) - 0.5f) * (grid.bounds_max[0] - grid.bounds_min[0]) * invRes;
                float jy = s.pos.y + (randomFloat(seed) - 0.5f) * (grid.bounds_max[1] - grid.bounds_min[1]) * invRes;
                float jz = s.pos.z + (randomFloat(seed) - 0.5f) * (grid.bounds_max[2] - grid.bounds_min[2]) * invRes;
                // Resolve the jittered sample at the home cell's OWN level (one
                // hash probe, no descent — co-leveled so the box-filter support
                // matches the jitter width). If the neighbor is COARSER (no
                // cell at foundLevel there — the fine side of a refinement
                // face), fall through to the parent level so the filter
                // crosses refinement boundaries instead of collapsing onto the
                // home cell: the old home-cell fallback pinned those deposits
                // at rel = ±1 on the boundary face and severed the Müller
                // splatting continuity exactly where the visible level seams
                // are. The home-cell fallback remains for jitters that leave
                // the grid, land in an unallocated base cell, or cross a
                // level gap >= 2 (no 2:1 balance is enforced) — those
                // deposits still clamp rel to the face.
                unsigned int ctxLevel = foundLevel;
                unsigned int ctxCellIdx = pathGuideCellAtLevel(grid, jx, jy, jz, foundLevel);
                // Walk UP through coarser levels until a cell exists (down
                // to start_level — nothing is allocated coarser). A single
                // parent probe only healed gap-1 faces; the grid enforces no
                // 2:1 balance, so a max-level region can directly border a
                // base-level one (observed: L6 against L2) and the filter
                // must cross that face too or the refined region keeps a
                // hard convergence seam at its border. Bounded: at most
                // (max_level - start_level) probes, each one hash lookup.
                while (ctxCellIdx == PG_INVALID_CELL &&
                       ctxLevel > params.path_guide_start_level) {
                    ctxLevel--;
                    ctxCellIdx = pathGuideCellAtLevel(grid, jx, jy, jz, ctxLevel);
                }
                if (ctxCellIdx == PG_INVALID_CELL) {
                    ctxCellIdx = exactCellIdx;   // jitter left coverage
                    ctxLevel = foundLevel;
                }
                // Train the JITTERED cell — the same box-filter cell used for
                // sampling/PDF — so training and sampling share one stochastic
                // support (Müller 2017).
                trainCellIdx = ctxCellIdx;

                // Position of the (jittered) deposit within its training cell,
                // normalized to [-1,1] from the cell center AT THE CTX CELL'S
                // OWN LEVEL (it may be the coarser parent across a refinement
                // face). Feeds the deposit centroid (parallax pivot) and
                // spread (kappa cap) — not subdivision, which uses the exact
                // stats above.
                float ctxRes = sparseResolutionAtLevel(grid, ctxLevel);
                float ctxCx, ctxCy, ctxCz;
                pgCellCenter(grid, ctxCellIdx, ctxCx, ctxCy, ctxCz);
                trainRelX = clamp((jx - ctxCx) * 2.0f * ctxRes / (grid.bounds_max[0] - grid.bounds_min[0]), -1.0f, 1.0f);
                trainRelY = clamp((jy - ctxCy) * 2.0f * ctxRes / (grid.bounds_max[1] - grid.bounds_min[1]), -1.0f, 1.0f);
                trainRelZ = clamp((jz - ctxCz) * 2.0f * ctxRes / (grid.bounds_max[2] - grid.bounds_min[2]), -1.0f, 1.0f);

                // (1-pSpec) sends near-specular surfaces to pure VNDF, but it
                // also zeroes guiding on ROUGH metals (pSpec pins to 1 for
                // metallic>0.95 regardless of roughness) — exactly where the
                // incident-radiance guide is the only variance reduction for
                // caustics / emissive-via-glossy chains. Restore a
                // roughness-proportional floor: smooth metals stay ~0 (guide
                // barely overlaps the mirror lobe), rough metals regain up to
                // PG_SPEC_GUIDE_FLOOR. Deterministic in (pSpec, roughness), so
                // sampler / combined PDF / NEE MIS all derive the same alpha.
                float guideFrac = fmaxf(1.0f - pSpec,
                    PG_SPEC_GUIDE_FLOOR * smoothstep01(0.08f, 0.30f, s.roughness));
                float wantGuide = clamp(params.path_guide_mis_weight, 0.0f, 0.95f)
                                * guideFrac;
                if (wantGuide > 0.02f) {
                    const float* cell = sparseCellDataPtr(grid, ctxCellIdx);
                    if (cell != nullptr) {
                        // Confidence ramp keeps barely-trained mixtures from
                        // grabbing full weight. Smoothstep (vs a linear clamp)
                        // gives zero-derivative ends, so guide weight fades in
                        // and saturates gently instead of with a kink — this
                        // softens the alpha discontinuity between a mature cell
                        // and a freshly-subdivided neighbor (a grid-boundary
                        // pop). conf only scales alpha, never the estimator, so
                        // any ramp shape stays unbiased.
                        //
                        // Confidence reads the SLOW-decayed maturity, not the
                        // fast EMA count: the fast count measures recent
                        // deposit RATE, and in dim regions lit by a small/hard
                        // light deposits only occur when a path FINDS light —
                        // rare under BSDF sampling — so rate-based confidence
                        // deadlocks at ~0 exactly where guiding is needed most
                        // (no guide -> no successes -> no deposits). Maturity
                        // accumulates evidence over ~50 refits, so consistent
                        // trickle deposits eventually switch guiding on and
                        // break the loop.
                        float conf = smoothstep01(0.0f, 32.0f, cell[PG_MATURITY]);
                        if (conf > 0.0f) {
                            // Eligible-lobe subset: only narrow lobes (kappa
                            // >= 2) with real mixture weight. Wide lobes are
                            // no better than the cosine BSDF leg and waste up
                            // to half their samples below the horizon;
                            // re-seeded exploratory lobes (kappa 1) stay
                            // training-only. The guide leg samples ONE
                            // eligible lobe picked by mixture weight — the
                            // same auxiliary-variable conditioning as the
                            // cell jitter, so using that lobe's pdf in every
                            // MIS denominator stays unbiased.
                            // Soft eligibility: ramp the kappa>=2 and
                            // weight>=0.05 gates with smoothstep instead of hard
                            // cliffs. A lobe whose kappa or weight straddles the
                            // threshold between adjacent cells no longer flips
                            // eligSum (and thus alpha) discontinuously across a
                            // boundary. The lower edges (kappa 1.5, weight 0.03)
                            // still exclude wide lobes and kappa=1 exploratory
                            // re-seeds (training-only), preserving the original
                            // intent. Eligibility scales alpha only, so it stays
                            // unbiased.
                            float eligW[PG_NUM_LOBES];
                            float eligSum = 0.0f;
                            const float3& sn = s.shadingNormal;
                            for (int k = 0; k < PG_NUM_LOBES; k++) {
                                const float* l = cell + k * PG_LOBE_STRIDE;
                                float ke = smoothstep01(1.5f, 3.0f, l[PG_L_KAPPA]);
                                float we = smoothstep01(0.03f, 0.07f, l[PG_L_WEIGHT]);
                                // Normal-alignment factor: a lobe whose mean
                                // points along/into the surface spends most of
                                // its (full-sphere) vMF mass below the horizon,
                                // where the guide draw is wasted (zero
                                // contribution, path terminates). Down-weighting
                                // it here reduces that waste UNBIASEDLY — it only
                                // reshapes the proposal (eligSum scales alpha,
                                // and the pick + combinedPdf condition on the
                                // realized eligibility). It is NOT a below-
                                // horizon resample: resampling would add an
                                // alpha*P(below)*pBsdf term the combinedPdf
                                // divisor does not account for, which biases.
                                float mn = l[PG_L_MU_X]*sn.x + l[PG_L_MU_Y]*sn.y + l[PG_L_MU_Z]*sn.z;
                                float ne = smoothstep01(-0.30f, 0.20f, mn);
                                eligW[k] = l[PG_L_WEIGHT] * ke * we * ne;
                                eligSum += eligW[k];
                            }
                            // Alpha scales by the eligible mixture mass: a
                            // cell that is half "narrow window" and half
                            // "broad sky" guides half its continuation budget
                            // and leaves the rest to the BSDF leg.
                            float alpha = wantGuide * conf * eligSum;
                            if (eligSum > 0.05f && alpha >= 0.02f) {
                                float target = randomFloat(seed) * eligSum;
                                float cum = 0.0f;
                                int pick = 0;
                                for (int k = 0; k < PG_NUM_LOBES; k++) {
                                    if (eligW[k] <= 0.0f) continue;
                                    pick = k;  // last eligible is the fallback
                                    cum += eligW[k];
                                    if (target <= cum) break;
                                }
                                const float* l = cell + pick * PG_LOBE_STRIDE;
                                incrementGuideStat(GUIDE_STAT_VALID_LOBE);
                                guideLobe.mux = l[PG_L_MU_X];
                                guideLobe.muy = l[PG_L_MU_Y];
                                guideLobe.muz = l[PG_L_MU_Z];
                                guideLobe.kappa = l[PG_L_KAPPA];
                                guideLobe.expNeg2K = l[PG_L_EXP_NEG2K];
                                // Parallax-aware reprojection: swing the lobe
                                // mean toward where its radiance actually comes
                                // from (pivot + mu*meanDist), reducing
                                // within-cell directional error and giving
                                // neighbouring cells a consistent aim near
                                // shared faces. The PIVOT is the cell's
                                // measured deposit centroid (CUM_SR/CUM_SMW,
                                // mapped from rel back to world), not the cell
                                // center: the lobe was fit where the deposits
                                // are, and pivoting there shrinks the residual
                                // positional error from half-a-cell to the
                                // deposit spread — the same quantity that now
                                // bounds kappa, so coarse cells aim as well as
                                // their evidence allows (Ruppert 2020 spirit).
                                // Reprojects mu in place; the cached vMF
                                // kappa/pdf are evaluated around the
                                // reprojected mean by sampler/PDF/NEE alike, so
                                // the estimator stays consistent and unbiased.
                                float pvX = ctxCx, pvY = ctxCy, pvZ = ctxCz;
                                float smw = cell[PG_CUM_SMW];
                                if (smw > 1e-4f) {
                                    float invSmw = 1.0f / smw;
                                    float halfInv = 0.5f / ctxRes;   // halfCell = extent/(2*res)
                                    pvX += cell[PG_CUM_SR_X] * invSmw * (grid.bounds_max[0] - grid.bounds_min[0]) * halfInv;
                                    pvY += cell[PG_CUM_SR_Y] * invSmw * (grid.bounds_max[1] - grid.bounds_min[1]) * halfInv;
                                    pvZ += cell[PG_CUM_SR_Z] * invSmw * (grid.bounds_max[2] - grid.bounds_min[2]) * halfInv;
                                }
                                pgParallaxReproject(pvX, pvY, pvZ, l[PG_L_MEAN_DIST],
                                    s.pos.x, s.pos.y, s.pos.z,
                                    guideLobe.mux, guideLobe.muy, guideLobe.muz);

                                // Product (BSDF x incident) guiding. The pure
                                // incident-radiance lobe ignores the
                                // view-dependent BSDF peak, so on a glossy
                                // receiver most guide samples land where the
                                // BSDF is ~0. Multiply it by a vMF approximation
                                // of the specular BSDF lobe — centered on the
                                // reflection direction R, concentration kb from
                                // roughness and scaled by pSpec — so the guide
                                // concentrates where BOTH incident radiance and
                                // the BSDF are high. Product of two vMFs is a
                                // vMF: r = kg*mu_g + kb*R, kappa_p = |r|, mu_p =
                                // r/|r|. Diffuse/rough surfaces get kb ~ 0 =>
                                // product ~ the incident lobe (unchanged). The
                                // product lobe replaces guideLobe BEFORE the
                                // sampler, the combined PDF, and NEE MIS read
                                // it, so all three condition on it — unbiased.
                                // (kb's roughness mapping is heuristic and a
                                // tuning knob; validate on a glossy-receiver
                                // scene.)
                                float alphaR = fmaxf(s.roughness * s.roughness, 1e-3f);
                                float kb = pSpec * fminf(2.0f / (alphaR * alphaR), 2000.0f);
                                if (kb > 1.0f) {
                                    float3 Rd = reflect(rayDir, s.shadingNormal);
                                    float rx = guideLobe.kappa * guideLobe.mux + kb * Rd.x;
                                    float ry = guideLobe.kappa * guideLobe.muy + kb * Rd.y;
                                    float rz = guideLobe.kappa * guideLobe.muz + kb * Rd.z;
                                    float kp = sqrtf(rx * rx + ry * ry + rz * rz);
                                    if (kp > 1e-4f) {
                                        float invkp = 1.0f / kp;
                                        guideLobe.mux = rx * invkp;
                                        guideLobe.muy = ry * invkp;
                                        guideLobe.muz = rz * invkp;
                                        guideLobe.kappa = fminf(kp, 2000.0f);
                                        guideLobe.expNeg2K = expf(-2.0f * guideLobe.kappa);
                                    }
                                }
                                guideAlpha = alpha;
                            }
                        }
                    }
                }
            }
        }

        //── Next event estimation (one light sample, one shadow ray) ────────
        float3 neeContrib = sampleDirectLight(s, V, pSpec, pSheen, pCC, guideAlpha,
            guideLobe, seed, atCap);
        // Off-cap, the stochastic transmission event above evaluates the
        // (1-t)*opaque + t*transmission blend by branching, so surviving to
        // NEE already carries the (1-t) factor. At the cap that draw never
        // happens, so the factor must be applied explicitly — otherwise a
        // deep glass chain renders its cap vertex as fully opaque (energy
        // ADDED, not just truncation bias).
        if (atCap) neeContrib = neeContrib * (1.0f - s.transmission);
        radiance = radiance + clampContribution(throughput * neeContrib);

        // RGB local emission for this vertex = surface emission + the NEE
        // contribution. Two deliberate, load-bearing choices live here:
        //   - UNCLAMPED on purpose: the image accumulates clampContribution()ed
        //     values (above), but the guide must learn the TRUE high-variance
        //     target the firefly clamp exists to tame, not the clamped proxy.
        //     The deposit is separately bounded by TRAIN_WEIGHT_CLAMP below.
        //   - delta-light NEE is intentionally included: it is folded into THIS
        //     vertex's reconstructed lo and deposited at the PREVIOUS vertex
        //     along the incoming direction ("this surface is bright from that
        //     direction"). Delta/transmission vertices set cellIdx=0xFFFFFFFF,
        //     so no lobe is ever aimed at a delta's own unreproducible
        //     direction. Do not "fix" either by clamping or excluding NEE.
        float3 localRGB = emissionContrib + neeContrib;

        //── Depth cap: no continuation past this vertex ─────────────────────
        if (atCap) {
            if (trainCount < MAX_TRAIN_VERTICES) {
                TrainRecord& tr = train[trainCount++];
                tr.cellIdx = 0xFFFFFFFFu;
                // Always the sentinel here: the guide block is gated !atCap,
                // so no lookup happened and cap vertices carry no visit.
                tr.exactPacked = exactPacked;
                tr.dirX = 0.0f; tr.dirY = 0.0f; tr.dirZ = 1.0f;
                tr.pdf = 0.0f;
                tr.local = localRGB;
                tr.beta = make_float3(0.0f, 0.0f, 0.0f);
            }
            break;
        }

        //── Russian roulette (throughput-based, unbiased) ────────────────────
        // Starts at depth 3, not 2: geometry seen THROUGH a reflection has its
        // first GI bounce at depth 2, and killing there made reflected GI
        // systematically noisier (and clamp-darker) than directly-viewed GI —
        // the "back wall is black in the cups" symptom. Survival caps at 1:
        // a path still carrying full throughput (mirror chains) loses real
        // energy if killed; the depth cap bounds the cost instead.
        float rrInv = 1.0f;
        if (depth >= 3) {
            float maxT = fmaxf(throughput.x, fmaxf(throughput.y, throughput.z));
            float survival = clamp(maxT, 0.05f, 1.0f);
            if (randomFloat(seed) >= survival) {
                if (trainCount < MAX_TRAIN_VERTICES) {
                    TrainRecord& tr = train[trainCount++];
                    tr.cellIdx = 0xFFFFFFFFu;
                    // The visit still counts — RR deaths correlate with
                    // darkness, and dropping them would reintroduce the
                    // brightness-correlated eligibility the visit stats
                    // exist to kill. (Every record site MUST set this
                    // field: train[] is uninitialized local memory.)
                    tr.exactPacked = exactPacked;
                    tr.dirX = 0.0f; tr.dirY = 0.0f; tr.dirZ = 1.0f;
                    tr.pdf = 0.0f;
                    tr.local = localRGB;
                    tr.beta = make_float3(0.0f, 0.0f, 0.0f);   // chain ends here
                }
                break;
            }
            rrInv = 1.0f / survival;
        }

        //── Sample continuation direction (one-sample MIS: guide vs BSDF) ───
        float3 L;
        bool haveDir = false;
        if (guideAlpha > 0.0f && randomFloat(seed) < guideAlpha) {
            // Guide leg: sample the vertex's conditioned vMF lobe
            float gx, gy, gz;
            vmfSampleCached(guideLobe.mux, guideLobe.muy, guideLobe.muz,
                guideLobe.kappa, guideLobe.expNeg2K,
                randomFloat(seed), randomFloat(seed), gx, gy, gz);
            L = make_float3(gx, gy, gz);
            haveDir = true;
        } else {
            incrementGuideStat(GUIDE_STAT_BSDF_SAMPLED);
            haveDir = sampleBSDFMixture(V, s.shadingNormal, s.roughness,
                s.clearcoatRoughness, pSpec, pSheen, pCC,
                randomFloat(seed), randomFloat(seed), randomFloat(seed), L);
        }

        float NdotL = haveDir ? dot(s.shadingNormal, L) : 0.0f;
        if (!haveDir || NdotL <= 0.0f) {
            // Below-horizon (vMF covers the sphere) or degenerate sample:
            // contribution is genuinely zero — terminate, record chain end.
            if (haveDir) incrementGuideStat(GUIDE_STAT_BELOW_HORIZ);
            if (trainCount < MAX_TRAIN_VERTICES) {
                TrainRecord& tr = train[trainCount++];
                tr.cellIdx = 0xFFFFFFFFu;
                tr.exactPacked = exactPacked;   // the visit still counts
                tr.dirX = 0.0f; tr.dirY = 0.0f; tr.dirZ = 1.0f;
                tr.pdf = 0.0f;
                tr.local = localRGB;
                tr.beta = make_float3(0.0f, 0.0f, 0.0f);
            }
            break;
        }

        // Combined PDF: the one-sample MIS estimator divides by the mixture
        // density — no separate weight needed.
        float pdfBsdf = pdfBSDFMixture(V, L, s.shadingNormal, s.roughness,
            s.clearcoatRoughness, pSpec, pSheen, pCC);
        float combinedPdf;
        if (guideAlpha > 0.0f) {
            float pdfGuide = guideLobePdf(guideLobe, L);
            combinedPdf = guideAlpha * pdfGuide + (1.0f - guideAlpha) * pdfBsdf;
        } else {
            combinedPdf = pdfBsdf;
        }
        if (combinedPdf <= 1e-8f) {
            if (trainCount < MAX_TRAIN_VERTICES) {
                TrainRecord& tr = train[trainCount++];
                tr.cellIdx = 0xFFFFFFFFu;
                tr.exactPacked = exactPacked;   // the visit still counts
                tr.dirX = 0.0f; tr.dirY = 0.0f; tr.dirZ = 1.0f;
                tr.pdf = 0.0f;
                tr.local = localRGB;
                tr.beta = make_float3(0.0f, 0.0f, 0.0f);
            }
            break;
        }

        float3 f = evalPbrBSDF(V, L, s.shadingNormal, s.baseColor, s.metallic, s.roughness,
            s.clearcoat, s.clearcoatRoughness, s.sheenColor, s.sheenRoughness,
            params.quality_mode, pSpec, pSheen, pCC, nullptr);

        float3 beta = f * (NdotL / combinedPdf) * rrInv;
        throughput = throughput * beta;
        incrementGuideStat(GUIDE_STAT_CONTRIBUTED);

        if (trainCount < MAX_TRAIN_VERTICES) {
            TrainRecord& tr = train[trainCount++];
            tr.cellIdx = trainCellIdx;
            tr.exactPacked = exactPacked;
            tr.dirX = L.x; tr.dirY = L.y; tr.dirZ = L.z;
            tr.pdf = combinedPdf;
            tr.local = localRGB;
            tr.beta = beta;
            tr.dist = sceneFarDistance();   // patched to the next segment's hit.t
            tr.relX = trainRelX; tr.relY = trainRelY; tr.relZ = trainRelZ;
            if (trainCellIdx != PG_INVALID_CELL) pendingDistRecord = trainCount - 1;
        } else {
            trainChainBroken = true;  // path continues unrecorded
        }

        // Continue the path
        float NdotD = dot(s.geomNormal, L);
        float3 offsetNormal = (NdotD > 0.0f) ? s.geomNormal : -s.geomNormal;
        rayOrigin = s.pos + offsetNormal * RAY_EPS;
        rayDir = L;
        tmin = RAY_EPS;
        tmax = sceneFarDistance();
        prevDelta = false;
        prevPdf = combinedPdf;
    }

    //── Training backward pass ───────────────────────────────────────────────
    // Reconstruct the incident radiance each vertex saw along its sampled
    // direction:  L_o(k) = local(k) + beta(k) * L_o(k+1), seeded with the
    // environment radiance if the path escaped. Deposit Li/pdf (clamped) into
    // the vertex's training cell. Training only shapes the guide — image
    // unbiasedness never depends on it. Subsampled per path (PG_TRAIN_PROB)
    // with compensating weight to cut contended atomic traffic.
    // Gated on path_guide_training: while the host state machine is paused no
    // refit folds the interval sums, so depositing would let them grow without
    // bound (float precision loss + a maturity spike that mass-subdivides on
    // resume).
    if (guidingActive && params.path_guide_training != 0 &&
        trainCount > 0 && randomFloat(seed) < PG_TRAIN_PROB) {
        // RGB recurrence: lo(k) = local(k) + beta(k) ⊙ lo(k+1). The deposit is
        // the LUMINANCE of this true RGB incident radiance — element-wise RGB
        // accumulation avoids the chromatic over/under-estimation a scalar
        // luminance recurrence produces across colored bounces.
        float3 lo = trainChainBroken ? make_float3(0.0f, 0.0f, 0.0f) : terminalEnv;
        for (int k = trainCount - 1; k >= 0; --k) {
            const TrainRecord& tr = train[k];
            float loLum = luminance3(lo);
            // Visit + half-cell radiance stats into the EXACT cell,
            // unconditionally (no radiance gate — see pathGuideVisitCell).
            // These drive subdivision; the lobe deposit below stays gated.
            if (tr.exactPacked != 0xFFFFFFFFu) {
                pathGuideVisitCell(grid, tr.exactPacked, loLum);
            }
            if (tr.cellIdx != 0xFFFFFFFFu && loLum > 1e-6f && tr.pdf > 1e-8f) {
                float* cell = sparseCellDataPtr(grid, tr.cellIdx);
                if (cell != nullptr) {
                    // Clamp BEFORE the subsampling compensation so the
                    // effective per-estimate clamp matches the unsubsampled
                    // semantics (clamp-after-scale would tighten it 4x and
                    // suppress exactly the bright signals worth learning).
                    float w = fminf(loLum / fmaxf(tr.pdf, 1e-4f), TRAIN_WEIGHT_CLAMP)
                            * PG_TRAIN_WEIGHT_SCALE;
                    pathGuideTrainCell(cell, tr.dirX, tr.dirY, tr.dirZ, w, tr.dist,
                                       tr.relX, tr.relY, tr.relZ, params.frame_index);
                }
            }
            lo = tr.local + tr.beta * lo;
        }
    }

    return radiance;
}

//------------------------------------------------------------------------------
// Raygen entry point
//------------------------------------------------------------------------------

extern "C" __global__ void __raygen__simple() {
    const uint3 idx = optixGetLaunchIndex();

    unsigned int pixelX = params.pick_mode ? params.pick_x : idx.x;
    unsigned int pixelY = params.pick_mode ? params.pick_y : idx.y;
    const unsigned int linear_idx = pixelY * params.width + pixelX;

    //── Pick mode: one center ray, write instance + hit position ────────────
    if (params.pick_mode) {
        float u = (pixelX + 0.5f) / (float)params.width;
        float v = (pixelY + 0.5f) / (float)params.height;
        float ndcX = 2.0f * u - 1.0f;
        float ndcY = 1.0f - 2.0f * v;
        float3 rayDir = normalize(params.camera.forward
            + params.camera.right * (ndcX * params.tan_half_fov_x)
            + params.camera.up * (ndcY * params.tan_half_fov_y));

        HitInfo hit = traceRadianceRay(params.camera.position, rayDir,
            params.camera.nearPlane, params.camera.farPlane);
        if (params.pick_result != nullptr) {
            if (hit.t >= 0.0f) {
                float3 p = params.camera.position + rayDir * hit.t;
                params.pick_result->instanceId = hit.instanceId;
                params.pick_result->hitX = p.x;
                params.pick_result->hitY = p.y;
                params.pick_result->hitZ = p.z;
            } else {
                params.pick_result->instanceId = 0xFFFFFFFFu;
                params.pick_result->hitX = 0.0f;
                params.pick_result->hitY = 0.0f;
                params.pick_result->hitZ = 0.0f;
            }
        }
        return;
    }

    const unsigned int spp = max(1u, params.samples_per_pixel);

    // Stratification grid covering the FULL pixel for any spp (the previous
    // ceil(sqrt(spp))^2 grid left rows of the pixel unsampled for non-square
    // spp — a systematic filter shift). When spp < gridX*gridY the chosen
    // strata are equidistributed and rotated per frame so accumulation covers
    // every stratum.
    unsigned int gridX = 1, gridY = 1, gridTotal = 1;
    if (spp > 1) {
        gridX = (unsigned int)ceilf(sqrtf((float)spp));
        gridY = (spp + gridX - 1) / gridX;
        gridTotal = gridX * gridY;
    }

    float3 accumulatedColor = make_float3(0.0f, 0.0f, 0.0f);
    float3 accumulatedAlbedo = make_float3(0.0f, 0.0f, 0.0f);
    float3 accumulatedNormal = make_float3(0.0f, 0.0f, 0.0f);
    float selCoverage = 0.0f;   // samples whose first hit is the selected instance
    float selRimSum = 0.0f;

    for (unsigned int sampleIdx = 0; sampleIdx < spp; ++sampleIdx) {
        unsigned int seed = mixSeed(pixelX, pixelY, params.frame_index, sampleIdx);

        float jitterX, jitterY;
        if (spp > 1) {
            unsigned int s = (unsigned int)(((unsigned long long)sampleIdx * gridTotal) / spp);
            s = (s + params.frame_index) % gridTotal;
            unsigned int sx = s % gridX;
            unsigned int sy = s / gridX;
            jitterX = (sx + randomFloat(seed)) / (float)gridX - 0.5f;
            jitterY = (sy + randomFloat(seed)) / (float)gridY - 0.5f;
        } else {
            // R2 low-discrepancy jitter for fast progressive AA convergence.
            // Cranley-Patterson rotation: R2 base + per-pixel random offset
            // so neighbouring pixels don't share the same sequence phase.
            unsigned int pixelHash = wangHash(pixelX * 0x9e3779b9u + pixelY);
            float offsetX = hashToFloat(pixelHash);
            float offsetY = hashToFloat(wangHash(pixelHash));
            float2 r2 = r2Sequence(params.frame_index);
            jitterX = fmodf(r2.x + offsetX, 1.0f) - 0.5f;
            jitterY = fmodf(r2.y + offsetY, 1.0f) - 0.5f;
        }

        const float u = ((float)pixelX + 0.5f + jitterX) / (float)params.width;
        const float v = ((float)pixelY + 0.5f + jitterY) / (float)params.height;
        const float ndcX = 2.0f * u - 1.0f;
        const float ndcY = 1.0f - 2.0f * v;

        float3 rayDir = normalize(params.camera.forward
            + params.camera.right * (ndcX * params.tan_half_fov_x)
            + params.camera.up * (ndcY * params.tan_half_fov_y));

        if (params.scene_handle == 0) {
            accumulatedColor = accumulatedColor + make_float3(u, v, 0.2f);
            continue;
        }

        unsigned int firstInstance = 0xFFFFFFFFu;
        float selectionRim = 0.0f;
        float3 sampleAlbedo = make_float3(0.0f, 0.0f, 0.0f);
        float3 sampleNormal = make_float3(0.0f, 0.0f, 0.0f);
        float3 sample = tracePath(params.camera.position, rayDir, seed,
            &firstInstance, &selectionRim, &sampleAlbedo, &sampleNormal);

        // Editor selection highlight: record coverage/rim only. The tint is
        // applied to the DISPLAYED value below, never to the accumulation
        // buffer — selection while guiding skips the accumulation reset, and
        // a baked-in tint would permanently stain the converged image.
        if (firstInstance != 0xFFFFFFFFu && firstInstance == params.selected_instance_id) {
            selCoverage += 1.0f;
            if (isfinite(selectionRim)) selRimSum += selectionRim;
        }

        // Degenerate-sample guard: a NaN would poison the accumulation buffer
        // permanently. Dropping the sample (vs propagating NaN) is the lesser
        // evil; NaNs indicate a bug and should not occur in normal operation.
        float sampleLum = sample.x + sample.y + sample.z;
        if (!isfinite(sampleLum)) {
            sample = make_float3(0.0f, 0.0f, 0.0f);
        }
        // Same guard for the denoiser AOVs: their progressive averages never
        // recover from a NaN either (degenerate geometry can make the
        // first-hit shading normal NaN even when the color sample is finite).
        if (!isfinite(sampleAlbedo.x + sampleAlbedo.y + sampleAlbedo.z)) {
            sampleAlbedo = make_float3(0.0f, 0.0f, 0.0f);
        }
        if (!isfinite(sampleNormal.x + sampleNormal.y + sampleNormal.z)) {
            sampleNormal = make_float3(0.0f, 0.0f, 0.0f);
        }

        accumulatedColor = accumulatedColor + sample;
        accumulatedAlbedo = accumulatedAlbedo + sampleAlbedo;
        accumulatedNormal = accumulatedNormal + sampleNormal;
    }

    float invSpp = 1.0f / (float)spp;
    float3 newColor = accumulatedColor * invSpp;
    float3 newAlbedo = accumulatedAlbedo * invSpp;
    float3 newNormal = accumulatedNormal * invSpp;

    float3 displayColor;
    if (params.accumulated_frames > 0 && params.accumulation_buffer != nullptr) {
        float4 accumulated = params.accumulation_buffer[linear_idx];
        float n = (float)(params.accumulated_frames + 1);

        float3 blended = make_float3(
            accumulated.x + (newColor.x - accumulated.x) / n,
            accumulated.y + (newColor.y - accumulated.y) / n,
            accumulated.z + (newColor.z - accumulated.z) / n);

        params.accumulation_buffer[linear_idx] = make_float4(blended.x, blended.y, blended.z, 1.0f);
        displayColor = blended;
    } else {
        if (params.accumulation_buffer != nullptr) {
            params.accumulation_buffer[linear_idx] = make_float4(newColor.x, newColor.y, newColor.z, 1.0f);
        }
        displayColor = newColor;
    }

    // Editor selection highlight (display-only tint, not physical): applied to
    // the blended value at display time — per-sample tinting would both bake
    // the highlight into the accumulation buffer and make its brightness vary
    // with spp/noise. Coverage-weighted so AA edge pixels tint partially.
    if (selCoverage > 0.0f) {
        float frac = selCoverage * invSpp;
        float3 tint = lerp(make_float3(1.0f, 1.0f, 1.0f),
                           make_float3(1.1f, 1.15f, 1.4f), frac);
        displayColor = displayColor * tint
                     + make_float3(0.2f, 0.4f, 1.0f) * (selRimSum * invSpp);
    }
    params.output_buffer[linear_idx] = make_float4(displayColor.x, displayColor.y, displayColor.z, 1.0f);

    // Denoiser AOV progressive accumulation (same frame counter as radiance)
    if (params.aov_albedo_buffer != nullptr) {
        if (params.accumulated_frames > 0) {
            float4 prev = params.aov_albedo_buffer[linear_idx];
            float n = (float)(params.accumulated_frames + 1);
            params.aov_albedo_buffer[linear_idx] = make_float4(
                prev.x + (newAlbedo.x - prev.x) / n,
                prev.y + (newAlbedo.y - prev.y) / n,
                prev.z + (newAlbedo.z - prev.z) / n, 1.0f);
        } else {
            params.aov_albedo_buffer[linear_idx] = make_float4(newAlbedo.x, newAlbedo.y, newAlbedo.z, 1.0f);
        }
    }
    if (params.aov_normal_buffer != nullptr) {
        if (params.accumulated_frames > 0) {
            float4 prev = params.aov_normal_buffer[linear_idx];
            float n = (float)(params.accumulated_frames + 1);
            params.aov_normal_buffer[linear_idx] = make_float4(
                prev.x + (newNormal.x - prev.x) / n,
                prev.y + (newNormal.y - prev.y) / n,
                prev.z + (newNormal.z - prev.z) / n, 1.0f);
        } else {
            params.aov_normal_buffer[linear_idx] = make_float4(newNormal.x, newNormal.y, newNormal.z, 1.0f);
        }
    }
}
