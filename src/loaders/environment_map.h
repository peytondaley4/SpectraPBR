#pragma once

#include <cuda_runtime.h>
#include <string>
#include <cstdint>

namespace spectra {

/**
 * EnvironmentMap - Loads HDR environment maps and computes importance sampling CDFs
 * 
 * Supports equirectangular HDR images (.hdr format via stb_image).
 * Computes 2D CDF for efficient importance sampling:
 * - Conditional CDF: P(u|v) - probability of selecting column u given row v
 * - Marginal CDF: P(v) - probability of selecting row v
 */
class EnvironmentMap {
public:
    EnvironmentMap() = default;
    ~EnvironmentMap();

    // Non-copyable
    EnvironmentMap(const EnvironmentMap&) = delete;
    EnvironmentMap& operator=(const EnvironmentMap&) = delete;

    /**
     * Load HDR environment map from file
     * Supports .hdr (Radiance) format
     * Returns true on success
     */
    bool loadFromFile(const std::string& path);

    /**
     * Free all GPU resources
     */
    void clear();

    /**
     * Check if environment map is loaded and ready
     */
    bool isLoaded() const { return m_texture != 0; }

    // GPU texture accessors
    cudaTextureObject_t getTexture() const { return m_texture; }
    cudaTextureObject_t getConditionalCDF() const { return m_conditionalCDF; }
    cudaTextureObject_t getMarginalCDF() const { return m_marginalCDF; }

    // Dimensions
    uint32_t getWidth() const { return m_width; }
    uint32_t getHeight() const { return m_height; }

    // Total luminance (for PDF normalization)
    float getTotalLuminance() const { return m_totalLuminance; }

    // Get loaded file path
    const std::string& getPath() const { return m_path; }

private:
    // Build importance sampling CDFs from luminance data
    bool buildCDFs(const float* rgbData);

    // Create CUDA texture from float data
    bool createTexture(const float* rgbData);

    // GPU resources
    cudaTextureObject_t m_texture = 0;
    cudaArray_t m_textureArray = nullptr;

    cudaTextureObject_t m_conditionalCDF = 0;  // 2D: width x height
    cudaArray_t m_conditionalCDFArray = nullptr;

    cudaTextureObject_t m_marginalCDF = 0;     // 1D: height elements
    cudaArray_t m_marginalCDFArray = nullptr;

    // Dimensions
    uint32_t m_width = 0;
    uint32_t m_height = 0;

    // Total weighted luminance for PDF normalization
    float m_totalLuminance = 0.0f;

    // Source path
    std::string m_path;
};

} // namespace spectra
