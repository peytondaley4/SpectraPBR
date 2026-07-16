#pragma once

#include "shared_types.h"
#include "ui/property_panel.h"
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

namespace spectra {

class OptixEngine;

class LightManager {
public:
    LightManager() = default;
    ~LightManager() { cleanup(); }

    // Add lights
    void addDirectionalLight(const GpuDirectionalLight& light);
    void addAreaLight(const GpuAreaLight& light);
    void addPointLight(const GpuPointLight& light);

    // Update lights from UI
    void updateDirectionalLight(uint32_t index, const ui::LightInfo& info);
    void updateAreaLight(uint32_t index, const ui::LightInfo& info);
    void updatePointLight(uint32_t index, const ui::LightInfo& info);

    // Get light info for UI
    ui::LightInfo getDirectionalLightInfo(uint32_t index) const;
    ui::LightInfo getAreaLightInfo(uint32_t index) const;
    ui::LightInfo getPointLightInfo(uint32_t index) const;

    // Sync to GPU (optional stream: when non-null uses async copy and does not block)
    void syncToGpu(OptixEngine* engine, cudaStream_t stream = nullptr);

    // Cleanup
    void cleanup();

    // Accessors
    size_t getDirectionalLightCount() const { return dirLights.size(); }
    size_t getAreaLightCount() const { return areaLights.size(); }
    size_t getPointLightCount() const { return pointLights.size(); }

    // Mesh-light lookup by owning instance (UINT32_MAX when the instance has
    // no light). Mesh-light emission is MIRRORED from the instance's
    // material emissive — NEE (this light) and path-hit emission (the SBT
    // material) must always agree or MIS between them biases.
    uint32_t findAreaLightByInstance(uint32_t instanceId) const {
        for (size_t i = 0; i < areaLights.size(); i++) {
            if (areaLights[i].instanceId == instanceId)
                return static_cast<uint32_t>(i);
        }
        return UINT32_MAX;
    }
    void setAreaLightEmission(uint32_t index, const float3& emission) {
        if (index < areaLights.size()) areaLights[index].emission = emission;
    }
    // Re-baked world-space geometry after a transform edit (mesh lights)
    void setAreaLightGeometry(uint32_t index, const float3& position,
                              const float3& normal, float area, const float2& size) {
        if (index >= areaLights.size()) return;
        areaLights[index].position = position;
        areaLights[index].normal = normal;
        areaLights[index].area = area;
        areaLights[index].size = size;
    }
    const GpuAreaLight* getAreaLight(uint32_t index) const {
        return index < areaLights.size() ? &areaLights[index] : nullptr;
    }

    // Create default lighting setup
    void createDefaultLights();

private:
    std::vector<GpuDirectionalLight> dirLights;
    std::vector<GpuAreaLight> areaLights;
    std::vector<GpuPointLight> pointLights;

    CUdeviceptr d_dirLights = 0;
    CUdeviceptr d_areaLights = 0;
    CUdeviceptr d_pointLights = 0;
    // Device buffer capacities in ELEMENTS — the buffers grow when the light
    // count outgrows them (adding a light at runtime must not overflow the
    // first-sync allocation).
    size_t dirCapacity = 0;
    size_t areaCapacity = 0;
    size_t pointCapacity = 0;

    // Walker/Vose alias table over all scene lights for O(1) device-side NEE
    // selection (same probabilities as the luminance CDF it replaces).
    // Rebuilt on every sync — light counts are tiny compared to the per-frame
    // cost the table removes.
    CUdeviceptr d_aliasProb = 0;     // float  [entry count]
    CUdeviceptr d_aliasIdx = 0;      // uint32 [entry count]
    CUdeviceptr d_aliasEntries = 0;  // uint32 [entry count], kind << 24 | index
    size_t aliasCapacity = 0;
};

} // namespace spectra
