#include "light_manager.h"
#include "optix_engine.h"
#include <cmath>

namespace spectra {

void LightManager::addDirectionalLight(const GpuDirectionalLight& light) {
    dirLights.push_back(light);
}

void LightManager::addAreaLight(const GpuAreaLight& light) {
    areaLights.push_back(light);
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

ui::LightInfo LightManager::getDirectionalLightInfo(uint32_t index) const {
    ui::LightInfo info = {};
    if (index >= dirLights.size()) return info;
    info.type = SceneNodeType::DirectionalLight;
    info.index = index;
    info.direction = dirLights[index].direction;
    info.color = dirLights[index].irradiance;
    info.angularDiameter = dirLights[index].angularDiameter;
    info.intensity = std::sqrt(info.color.x * info.color.x +
                                info.color.y * info.color.y +
                                info.color.z * info.color.z);
    return info;
}

ui::LightInfo LightManager::getAreaLightInfo(uint32_t index) const {
    ui::LightInfo info = {};
    if (index >= areaLights.size()) return info;
    info.type = SceneNodeType::AreaLight;
    info.index = index;
    info.position = areaLights[index].position;
    info.color = areaLights[index].emission;
    info.size = areaLights[index].size;
    info.intensity = std::sqrt(info.color.x * info.color.x +
                                info.color.y * info.color.y +
                                info.color.z * info.color.z);
    return info;
}

ui::LightInfo LightManager::getPointLightInfo(uint32_t index) const {
    ui::LightInfo info = {};
    if (index >= pointLights.size()) return info;
    info.type = SceneNodeType::PointLight;
    info.index = index;
    info.position = pointLights[index].position;
    info.radius = pointLights[index].radius;
    info.color = pointLights[index].intensity;
    info.intensity = std::sqrt(info.color.x * info.color.x +
                                info.color.y * info.color.y +
                                info.color.z * info.color.z);
    return info;
}

void LightManager::syncToGpu(OptixEngine* engine, cudaStream_t stream) {
    auto copy = [stream](void* dst, const void* src, size_t size) {
        if (stream) {
            cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, stream);
        } else {
            cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
        }
    };

    // Directional lights
    if (!dirLights.empty()) {
        size_t size = dirLights.size() * sizeof(GpuDirectionalLight);
        if (!d_dirLights) {
            cudaMalloc(reinterpret_cast<void**>(&d_dirLights), size);
        }
        copy(reinterpret_cast<void*>(d_dirLights), dirLights.data(), size);
        engine->setDirectionalLights(reinterpret_cast<GpuDirectionalLight*>(d_dirLights),
                                      static_cast<uint32_t>(dirLights.size()));
    }

    // Area lights
    if (!areaLights.empty()) {
        size_t size = areaLights.size() * sizeof(GpuAreaLight);
        if (!d_areaLights) {
            cudaMalloc(reinterpret_cast<void**>(&d_areaLights), size);
        }
        copy(reinterpret_cast<void*>(d_areaLights), areaLights.data(), size);
        engine->setAreaLights(reinterpret_cast<GpuAreaLight*>(d_areaLights),
                               static_cast<uint32_t>(areaLights.size()));
    }

    // Point lights
    if (!pointLights.empty()) {
        size_t size = pointLights.size() * sizeof(GpuPointLight);
        if (!d_pointLights) {
            cudaMalloc(reinterpret_cast<void**>(&d_pointLights), size);
        }
        copy(reinterpret_cast<void*>(d_pointLights), pointLights.data(), size);
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
    for (const auto& light : pointLights) {
        totalLightLum += 0.2126f * light.intensity.x +
                         0.7152f * light.intensity.y +
                         0.0722f * light.intensity.z;
    }
    for (const auto& light : dirLights) {
        totalLightLum += (0.2126f * light.irradiance.x +
                          0.7152f * light.irradiance.y +
                          0.0722f * light.irradiance.z) * 10.0f;
    }
    for (const auto& light : areaLights) {
        totalLightLum += (0.2126f * light.emission.x +
                          0.7152f * light.emission.y +
                          0.0722f * light.emission.z) * light.area;
    }
    engine->setTotalLightLuminance(totalLightLum);
}

void LightManager::cleanup() {
    if (d_dirLights) { cudaFree(reinterpret_cast<void*>(d_dirLights)); d_dirLights = 0; }
    if (d_areaLights) { cudaFree(reinterpret_cast<void*>(d_areaLights)); d_areaLights = 0; }
    if (d_pointLights) { cudaFree(reinterpret_cast<void*>(d_pointLights)); d_pointLights = 0; }
}

static float3 normalizeFloat3(float3 v) {
    float len = std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
    if (len > 0.0f) {
        return make_float3(v.x / len, v.y / len, v.z / len);
    }
    return v;
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
