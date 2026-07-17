#include "light_manager.h"
#include "optix_engine.h"
#include <cmath>
#include <algorithm>
#include <iostream>

namespace spectra {

static float3 normalizeFloat3(float3 v) {
    float len = std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
    if (len > 0.0f) {
        return make_float3(v.x / len, v.y / len, v.z / len);
    }
    return v;
}

void LightManager::addDirectionalLight(const GpuDirectionalLight& light) {
    dirLights.push_back(light);
}

void LightManager::addAreaLight(const GpuAreaLight& light) {
    GpuAreaLight l = light;
    // Virtual rects are sampled on the device (raygen.cu) as
    // position + tangent*u*size.x + cross(normal, tangent)*v*size.y with
    // pdf_area = 1/area — that assumes an orthonormal frame, so Gram-Schmidt
    // the tangent against the normal here or the sampled parallelogram's
    // area (and plane) disagrees with the pdf and biases NEE. Mesh lights
    // (triCount > 0) sample their real triangles; leave their frame alone.
    if (l.triCount == 0) {
        l.normal = normalizeFloat3(l.normal);
        float d = l.tangent.x * l.normal.x + l.tangent.y * l.normal.y +
                  l.tangent.z * l.normal.z;
        float3 t = make_float3(l.tangent.x - l.normal.x * d,
                               l.tangent.y - l.normal.y * d,
                               l.tangent.z - l.normal.z * d);
        if (t.x * t.x + t.y * t.y + t.z * t.z < 1e-12f) {
            // Tangent (near-)parallel to normal: rebuild from a safe axis
            float3 up = (std::fabs(l.normal.y) < 0.99f)
                ? make_float3(0.0f, 1.0f, 0.0f)
                : make_float3(1.0f, 0.0f, 0.0f);
            t = make_float3(up.y * l.normal.z - up.z * l.normal.y,
                            up.z * l.normal.x - up.x * l.normal.z,
                            up.x * l.normal.y - up.y * l.normal.x);
        }
        l.tangent = normalizeFloat3(t);
    }
    areaLights.push_back(l);
}

void LightManager::addPointLight(const GpuPointLight& light) {
    pointLights.push_back(light);
}

void LightManager::updateDirectionalLight(uint32_t index, const ui::LightInfo& info) {
    if (index >= dirLights.size()) return;
    dirLights[index].direction = info.direction;
    dirLights[index].angularDiameter = info.angularDiameter;
    dirLights[index].irradiance = info.color;
}

void LightManager::updateAreaLight(uint32_t index, const ui::LightInfo& info) {
    if (index >= areaLights.size()) return;
    areaLights[index].position = info.position;
    areaLights[index].emission = info.color;
    areaLights[index].size = info.size;
    // Mesh lights (triCount > 0) keep the area of their actual triangles —
    // it drives the sampling PDF and MIS weights. Only virtual rectangles
    // derive area from the editable size.
    if (areaLights[index].triCount == 0) {
        areaLights[index].area = info.size.x * info.size.y;
    }
}

void LightManager::updatePointLight(uint32_t index, const ui::LightInfo& info) {
    if (index >= pointLights.size()) return;
    pointLights[index].position = info.position;
    pointLights[index].radius = info.radius;
    pointLights[index].intensity = info.color;
}

// Readback convention (must mirror PropertyPanel::updateLightFromSliders,
// which recomposes color_out = pickerColor * intensity): intensity is the
// MAX component and the returned color is normalized by it, so a display ->
// edit round trip reproduces the same emission. The old magnitude-based
// readback inflated white lights by sqrt(3) on every touch.
static void splitColorIntensity(const float3& c, float3& outColor, float& outIntensity) {
    float m = std::max({c.x, c.y, c.z});
    if (m > 1e-6f) {
        outIntensity = m;
        outColor = make_float3(c.x / m, c.y / m, c.z / m);
    } else {
        outIntensity = 0.0f;
        outColor = make_float3(1.0f, 1.0f, 1.0f);
    }
}

ui::LightInfo LightManager::getDirectionalLightInfo(uint32_t index) const {
    ui::LightInfo info = {};
    if (index >= dirLights.size()) return info;
    info.type = SceneNodeType::DirectionalLight;
    info.index = index;
    info.direction = dirLights[index].direction;
    info.angularDiameter = dirLights[index].angularDiameter;
    splitColorIntensity(dirLights[index].irradiance, info.color, info.intensity);
    return info;
}

ui::LightInfo LightManager::getAreaLightInfo(uint32_t index) const {
    ui::LightInfo info = {};
    if (index >= areaLights.size()) return info;
    info.type = SceneNodeType::AreaLight;
    info.index = index;
    info.position = areaLights[index].position;
    info.size = areaLights[index].size;
    splitColorIntensity(areaLights[index].emission, info.color, info.intensity);
    return info;
}

ui::LightInfo LightManager::getPointLightInfo(uint32_t index) const {
    ui::LightInfo info = {};
    if (index >= pointLights.size()) return info;
    info.type = SceneNodeType::PointLight;
    info.index = index;
    info.position = pointLights[index].position;
    info.radius = pointLights[index].radius;
    splitColorIntensity(pointLights[index].intensity, info.color, info.intensity);
    return info;
}

void LightManager::syncToGpu(OptixEngine* engine, cudaStream_t stream) {
    // Grow-only buffer sync: reallocates when the light count outgrows the
    // device buffer (the old code allocated once at first-sync size, so any
    // later addition wrote past the allocation), copies, and reports success.
    // On failure the previous buffer/pointer stay valid.
    auto syncBuffer = [stream](CUdeviceptr& dptr, size_t& capacity,
                               const void* src, size_t count, size_t elemSize,
                               const char* what) -> bool {
        size_t bytes = count * elemSize;
        if (count > capacity) {
            CUdeviceptr newPtr = 0;
            cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&newPtr), bytes);
            if (err != cudaSuccess) {
                std::cerr << "[LightManager] cudaMalloc(" << what << ") failed: "
                          << cudaGetErrorString(err) << "\n";
                return false;
            }
            if (dptr) cudaFree(reinterpret_cast<void*>(dptr));
            dptr = newPtr;
            capacity = count;
        }
        cudaError_t err = stream
            ? cudaMemcpyAsync(reinterpret_cast<void*>(dptr), src, bytes,
                              cudaMemcpyHostToDevice, stream)
            : cudaMemcpy(reinterpret_cast<void*>(dptr), src, bytes,
                         cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            std::cerr << "[LightManager] light upload (" << what << ") failed: "
                      << cudaGetErrorString(err) << "\n";
            return false;
        }
        return true;
    };

    // Directional lights
    if (!dirLights.empty() &&
        syncBuffer(d_dirLights, dirCapacity, dirLights.data(),
                   dirLights.size(), sizeof(GpuDirectionalLight), "dir lights")) {
        engine->setDirectionalLights(reinterpret_cast<GpuDirectionalLight*>(d_dirLights),
                                      static_cast<uint32_t>(dirLights.size()));
    }

    // Area lights
    if (!areaLights.empty() &&
        syncBuffer(d_areaLights, areaCapacity, areaLights.data(),
                   areaLights.size(), sizeof(GpuAreaLight), "area lights")) {
        engine->setAreaLights(reinterpret_cast<GpuAreaLight*>(d_areaLights),
                               static_cast<uint32_t>(areaLights.size()));
    }

    // Point lights
    if (!pointLights.empty() &&
        syncBuffer(d_pointLights, pointCapacity, pointLights.data(),
                   pointLights.size(), sizeof(GpuPointLight), "point lights")) {
        engine->setPointLights(reinterpret_cast<GpuPointLight*>(d_pointLights),
                                static_cast<uint32_t>(pointLights.size()));
    }

    // Compute the light-selection total for NEE importance sampling.
    // These weight formulas MUST match raygen.cu::selectLight exactly — the
    // device walks the same lists with the same per-light weights and divides
    // by this total; any mismatch biases the estimator.
    //   point:       lum(intensity)
    //   directional: lum(irradiance) * 10   (selection-importance heuristic:
    //                directional lights illuminate everything, so oversample)
    //   area:        lum(emission) * area   (~flux; big bright panels matter more)
    float totalLightLum = 0.0f;
    std::vector<float> weights;
    std::vector<uint32_t> entries;
    weights.reserve(pointLights.size() + dirLights.size() + areaLights.size());
    entries.reserve(weights.capacity());
    for (size_t i = 0; i < pointLights.size(); ++i) {
        const auto& light = pointLights[i];
        float w = 0.2126f * light.intensity.x +
                  0.7152f * light.intensity.y +
                  0.0722f * light.intensity.z;
        totalLightLum += w;
        weights.push_back(w);
        entries.push_back((LIGHT_KIND_POINT << 24) | static_cast<uint32_t>(i));
    }
    for (size_t i = 0; i < dirLights.size(); ++i) {
        const auto& light = dirLights[i];
        float w = (0.2126f * light.irradiance.x +
                   0.7152f * light.irradiance.y +
                   0.0722f * light.irradiance.z) * 10.0f;
        totalLightLum += w;
        weights.push_back(w);
        entries.push_back((LIGHT_KIND_DIR << 24) | static_cast<uint32_t>(i));
    }
    for (size_t i = 0; i < areaLights.size(); ++i) {
        const auto& light = areaLights[i];
        float w = (0.2126f * light.emission.x +
                   0.7152f * light.emission.y +
                   0.0722f * light.emission.z) * light.area;
        totalLightLum += w;
        weights.push_back(w);
        entries.push_back((LIGHT_KIND_AREA << 24) | static_cast<uint32_t>(i));
    }
    engine->setTotalLightLuminance(totalLightLum);

    // Walker/Vose alias table over the scene lights: the device picks a
    // uniform bucket and accepts it with prob[] or takes its alias — O(1) and
    // exactly proportional to weight/total, replacing the per-vertex linear
    // CDF walk in raygen.cu::selectLight.
    const size_t n = weights.size();
    if (n == 0 || totalLightLum <= 0.0f) {
        engine->setLightAliasTable(nullptr, nullptr, nullptr, 0);
    } else {
        std::vector<float> prob(n);
        std::vector<uint32_t> alias(n);
        std::vector<float> scaled(n);
        std::vector<uint32_t> small, large;
        small.reserve(n); large.reserve(n);
        for (size_t i = 0; i < n; ++i) {
            scaled[i] = weights[i] * static_cast<float>(n) / totalLightLum;
            (scaled[i] < 1.0f ? small : large).push_back(static_cast<uint32_t>(i));
        }
        while (!small.empty() && !large.empty()) {
            uint32_t s = small.back(); small.pop_back();
            uint32_t l = large.back(); large.pop_back();
            prob[s] = scaled[s];
            alias[s] = l;
            scaled[l] = (scaled[l] + scaled[s]) - 1.0f;
            (scaled[l] < 1.0f ? small : large).push_back(l);
        }
        // Leftovers are exactly 1 up to rounding
        while (!large.empty()) { prob[large.back()] = 1.0f; alias[large.back()] = large.back(); large.pop_back(); }
        while (!small.empty()) { prob[small.back()] = 1.0f; alias[small.back()] = small.back(); small.pop_back(); }

        bool ok = true;
        if (n > aliasCapacity) {
            auto grow = [&](CUdeviceptr& dptr, size_t bytes, const char* what) {
                CUdeviceptr newPtr = 0;
                cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&newPtr), bytes);
                if (err != cudaSuccess) {
                    std::cerr << "[LightManager] cudaMalloc(" << what << ") failed: "
                              << cudaGetErrorString(err) << "\n";
                    return false;
                }
                if (dptr) cudaFree(reinterpret_cast<void*>(dptr));
                dptr = newPtr;
                return true;
            };
            ok = grow(d_aliasProb, n * sizeof(float), "light alias prob") &&
                 grow(d_aliasIdx, n * sizeof(uint32_t), "light alias idx") &&
                 grow(d_aliasEntries, n * sizeof(uint32_t), "light alias entries");
            if (ok) aliasCapacity = n;
        }
        if (ok) {
            // Synchronous copies on purpose: prob/alias/entries are locals,
            // so an async copy could outlive them. The table is a few dozen
            // elements — the sync cost is noise.
            auto upload = [](CUdeviceptr dst, const void* src, size_t bytes) {
                return cudaMemcpy(reinterpret_cast<void*>(dst), src, bytes,
                                  cudaMemcpyHostToDevice) == cudaSuccess;
            };
            ok = upload(d_aliasProb, prob.data(), n * sizeof(float)) &&
                 upload(d_aliasIdx, alias.data(), n * sizeof(uint32_t)) &&
                 upload(d_aliasEntries, entries.data(), n * sizeof(uint32_t));
            if (!ok) {
                std::cerr << "[LightManager] light alias table upload failed\n";
            }
        }
        if (ok) {
            engine->setLightAliasTable(
                reinterpret_cast<float*>(d_aliasProb),
                reinterpret_cast<uint32_t*>(d_aliasIdx),
                reinterpret_cast<uint32_t*>(d_aliasEntries),
                static_cast<uint32_t>(n));
        } else {
            engine->setLightAliasTable(nullptr, nullptr, nullptr, 0);
        }
    }
}

void LightManager::cleanup() {
    if (d_dirLights) { cudaFree(reinterpret_cast<void*>(d_dirLights)); d_dirLights = 0; }
    if (d_areaLights) { cudaFree(reinterpret_cast<void*>(d_areaLights)); d_areaLights = 0; }
    if (d_pointLights) { cudaFree(reinterpret_cast<void*>(d_pointLights)); d_pointLights = 0; }
    if (d_aliasProb) { cudaFree(reinterpret_cast<void*>(d_aliasProb)); d_aliasProb = 0; }
    if (d_aliasIdx) { cudaFree(reinterpret_cast<void*>(d_aliasIdx)); d_aliasIdx = 0; }
    if (d_aliasEntries) { cudaFree(reinterpret_cast<void*>(d_aliasEntries)); d_aliasEntries = 0; }
    dirCapacity = 0;
    areaCapacity = 0;
    pointCapacity = 0;
    aliasCapacity = 0;
}

void LightManager::createDefaultLights() {
    // Sun light
    GpuDirectionalLight sunLight;
    sunLight.direction = make_float3(0.5f, -0.8f, 0.3f);
    sunLight.angularDiameter = 0.2f;
    sunLight.irradiance = make_float3(3.0f, 2.9f, 2.7f);
    addDirectionalLight(sunLight);

    // Key light (virtual rectangle: no geometry, NEE-only)
    GpuAreaLight keyLight = {};
    keyLight.position = make_float3(3.0f, 4.0f, 2.0f);
    keyLight.normal = normalizeFloat3(make_float3(-0.3f, -0.8f, -0.2f));
    keyLight.tangent = make_float3(1.0f, 0.0f, 0.0f);
    keyLight.emission = make_float3(200.0f, 150.0f, 160.0f);
    keyLight.size = make_float2(2.0f, 2.0f);
    keyLight.area = 4.0f;
    keyLight.triOffset = 0;
    keyLight.triCount = 0;
    keyLight.instanceId = UINT32_MAX;
    addAreaLight(keyLight);

    // Fill light (virtual rectangle)
    GpuAreaLight fillLight = {};
    fillLight.position = make_float3(-2.5f, 2.0f, 3.0f);
    fillLight.normal = normalizeFloat3(make_float3(0.4f, -0.5f, -0.6f));
    fillLight.tangent = make_float3(0.0f, 0.0f, 1.0f);
    fillLight.emission = make_float3(100.0f, 110.0f, 120.0f);
    fillLight.size = make_float2(1.5f, 1.5f);
    fillLight.area = 2.25f;
    fillLight.triOffset = 0;
    fillLight.triCount = 0;
    fillLight.instanceId = UINT32_MAX;
    addAreaLight(fillLight);
}

} // namespace spectra
