#pragma once

#include <cuda_runtime.h>
#include <string>
#include <cstdint>

namespace spectra {

/**
 * EnvironmentMap - Loads HDR environment maps and builds an importance sampler.
 *
 * Supports equirectangular HDR images (.hdr format via stb_image).
 * Builds a Walker/Vose ALIAS TABLE over the W*H texels for O(1) importance
 * sampling proportional to sin(theta)-weighted luminance:
 * - env_alias_prob / env_alias_idx: the alias buckets (one pick + one compare)
 * - env_pmf: per-texel selection probability, used by environmentPdf() for MIS
 * (Replaced the old 2D conditional/marginal CDF binary search; same
 *  distribution, O(1) instead of O(log W + log H) dependent texture fetches.)
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

    // GPU accessors
    cudaTextureObject_t getTexture() const { return m_texture; }
    // Importance-sampling alias table + per-texel pmf (device linear buffers).
    const float* getAliasProb() const { return m_d_aliasProb; }
    const unsigned int* getAliasIdx() const { return m_d_aliasIdx; }
    const float* getPmf() const { return m_d_pmf; }

    // Dimensions
    uint32_t getWidth() const { return m_width; }
    uint32_t getHeight() const { return m_height; }

    // Total luminance (for PDF normalization)
    float getTotalLuminance() const { return m_totalLuminance; }

    // Get loaded file path
    const std::string& getPath() const { return m_path; }

private:
    // Build the Vose alias table + per-texel pmf from luminance data
    bool buildAliasTable(const float* rgbData);

    // Create CUDA texture from float data
    bool createTexture(const float* rgbData);

    // GPU resources
    cudaTextureObject_t m_texture = 0;
    cudaArray_t m_textureArray = nullptr;

    // Importance-sampling: Walker/Vose alias table over W*H texels + per-texel
    // selection probability. Device linear buffers (random index access, no
    // filtering — not textures).
    float*        m_d_aliasProb = nullptr;   // [W*H] bucket accept-probabilities
    unsigned int* m_d_aliasIdx  = nullptr;   // [W*H] bucket fallback texels
    float*        m_d_pmf       = nullptr;   // [W*H] per-texel selection probability

    // Dimensions
    uint32_t m_width = 0;
    uint32_t m_height = 0;

    // Total weighted luminance for PDF normalization
    float m_totalLuminance = 0.0f;

    // Source path
    std::string m_path;
};

} // namespace spectra
