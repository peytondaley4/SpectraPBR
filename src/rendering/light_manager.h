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

    // Create default lighting setup
    void createDefaultLights();

private:
    std::vector<GpuDirectionalLight> dirLights;
    std::vector<GpuAreaLight> areaLights;
    std::vector<GpuPointLight> pointLights;

    CUdeviceptr d_dirLights = 0;
    CUdeviceptr d_areaLights = 0;
    CUdeviceptr d_pointLights = 0;
};

} // namespace spectra
