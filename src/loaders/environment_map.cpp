#include "environment_map.h"
#include "cuda/cuda_texture_utils.h"
#include <stb_image.h>
#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>

namespace spectra {

EnvironmentMap::~EnvironmentMap() {
    clear();
}

void EnvironmentMap::clear() {
    // Destroy the env radiance texture
    if (m_texture) {
        cudaDestroyTextureObject(m_texture);
        m_texture = 0;
    }
    if (m_textureArray) {
        cudaFreeArray(m_textureArray);
        m_textureArray = nullptr;
    }

    // Free importance-sampling device buffers
    if (m_d_aliasProb) { cudaFree(m_d_aliasProb); m_d_aliasProb = nullptr; }
    if (m_d_aliasIdx)  { cudaFree(m_d_aliasIdx);  m_d_aliasIdx  = nullptr; }
    if (m_d_pmf)       { cudaFree(m_d_pmf);       m_d_pmf       = nullptr; }

    m_width = 0;
    m_height = 0;
    m_totalLuminance = 0.0f;
    m_path.clear();
}

bool EnvironmentMap::loadFromFile(const std::string& path) {
    // Clear any existing data
    clear();

    // Load HDR image with stb_image
    int width, height, channels;
    float* data = stbi_loadf(path.c_str(), &width, &height, &channels, 3);  // Force RGB

    if (!data) {
        std::cerr << "[EnvironmentMap] Failed to load: " << path
                  << " - " << stbi_failure_reason() << "\n";
        return false;
    }

    m_width = static_cast<uint32_t>(width);
    m_height = static_cast<uint32_t>(height);
    m_path = path;

    std::cout << "[EnvironmentMap] Loaded: " << path 
              << " (" << m_width << "x" << m_height << ")\n";

    // Create GPU texture
    if (!createTexture(data)) {
        stbi_image_free(data);
        clear();
        return false;
    }

    // Build importance-sampling alias table
    if (!buildAliasTable(data)) {
        stbi_image_free(data);
        clear();
        return false;
    }

    stbi_image_free(data);

    std::cout << "[EnvironmentMap] Total luminance: " << m_totalLuminance << "\n";
    return true;
}

bool EnvironmentMap::createTexture(const float* rgbData) {
    // Convert RGB to RGBA (CUDA textures prefer 4-channel)
    std::vector<float4> rgbaData(m_width * m_height);
    for (uint32_t i = 0; i < m_width * m_height; i++) {
        rgbaData[i] = make_float4(
            rgbData[i * 3 + 0],
            rgbData[i * 3 + 1],
            rgbData[i * 3 + 2],
            1.0f
        );
    }

    // Create CUDA array for the texture
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
    cudaError_t err = cudaMallocArray(&m_textureArray, &channelDesc, m_width, m_height);
    if (err != cudaSuccess) {
        std::cerr << "[EnvironmentMap] Failed to allocate CUDA array: " 
                  << cudaGetErrorString(err) << "\n";
        return false;
    }

    // Copy data to CUDA array
    err = cudaMemcpy2DToArray(
        m_textureArray, 0, 0,
        rgbaData.data(),
        m_width * sizeof(float4),
        m_width * sizeof(float4),
        m_height,
        cudaMemcpyHostToDevice
    );
    if (err != cudaSuccess) {
        std::cerr << "[EnvironmentMap] Failed to copy texture data: " 
                  << cudaGetErrorString(err) << "\n";
        return false;
    }

    // Create texture object (wrap horizontal, clamp vertical, linear filter, HDR floats)
    if (!createCudaTexture(m_texture, m_textureArray,
            cudaAddressModeWrap, cudaAddressModeClamp,
            cudaFilterModeLinear, cudaReadModeElementType)) {
        return false;
    }

    return true;
}

bool EnvironmentMap::buildAliasTable(const float* rgbData) {
    // Per-texel sin(theta)-weighted luminance = the unnormalized selection mass
    // (identical weighting to the old CDF path: each equirectangular row spans a
    // different latitude, weighted by its solid angle).
    const uint32_t n = m_width * m_height;
    const float PI = 3.14159265358979323846f;

    // Accumulate in double: a float running sum over millions of texels rounds
    // small (dark-texel) weights to zero, drifting the pmf the device divides
    // by away from the alias table's realized selection probabilities.
    std::vector<float> weighted(n);
    double totalLuminance = 0.0;
    for (uint32_t y = 0; y < m_height; y++) {
        float v = (y + 0.5f) / m_height;          // 0 (top) .. 1 (bottom)
        float sinTheta = std::sin(v * PI);
        for (uint32_t x = 0; x < m_width; x++) {
            uint32_t idx = y * m_width + x;
            float r = rgbData[idx * 3 + 0];
            float g = rgbData[idx * 3 + 1];
            float b = rgbData[idx * 3 + 2];
            float w = (0.2126f * r + 0.7152f * g + 0.0722f * b) * sinTheta;
            weighted[idx] = w;
            totalLuminance += w;
        }
    }
    if (totalLuminance <= 0.0) {
        std::cerr << "[EnvironmentMap] Warning: environment map has zero luminance\n";
        totalLuminance = 1.0;  // avoid division by zero
    }
    m_totalLuminance = static_cast<float>(totalLuminance);

    // Per-texel selection probability (normalized). This is exactly the
    // marginalPdf*conditionalPdf product the old CDF environmentPdf computed, so
    // the device PDF formula pmf[texel]*(W*H)/(2*pi^2*sin theta) is unchanged.
    std::vector<float> pmf(n);
    const double invTotal = 1.0 / totalLuminance;
    for (uint32_t i = 0; i < n; i++) pmf[i] = static_cast<float>(weighted[i] * invTotal);

    // Walker/Vose alias construction. scaled[i] = pmf[i]*n; pair light buckets
    // (<1) with heavy ones (>=1) until every bucket holds <=2 outcomes.
    std::vector<float> prob(n);
    std::vector<uint32_t> alias(n);
    std::vector<float> scaled(n);
    std::vector<uint32_t> small, large;
    small.reserve(n);
    large.reserve(n);
    for (uint32_t i = 0; i < n; i++) {
        scaled[i] = pmf[i] * static_cast<float>(n);
        (scaled[i] < 1.0f ? small : large).push_back(i);
    }
    while (!small.empty() && !large.empty()) {
        uint32_t l = small.back(); small.pop_back();
        uint32_t g = large.back(); large.pop_back();
        prob[l] = scaled[l];
        alias[l] = g;
        scaled[g] = (scaled[g] + scaled[l]) - 1.0f;   // = scaled[g] - (1 - scaled[l])
        (scaled[g] < 1.0f ? small : large).push_back(g);
    }
    // Leftovers from FP drift: accept with probability 1 (self-alias is never
    // taken because prob==1, but set it so a stray read is harmless).
    while (!large.empty()) { uint32_t g = large.back(); large.pop_back(); prob[g] = 1.0f; alias[g] = g; }
    while (!small.empty()) { uint32_t s = small.back(); small.pop_back(); prob[s] = 1.0f; alias[s] = s; }

    // Upload to device linear buffers.
    auto upload = [](void** dptr, const void* src, size_t bytes, const char* what) -> bool {
        cudaError_t err = cudaMalloc(dptr, bytes);
        if (err != cudaSuccess) {
            std::cerr << "[EnvironmentMap] alias " << what << " malloc failed: "
                      << cudaGetErrorString(err) << "\n";
            return false;
        }
        err = cudaMemcpy(*dptr, src, bytes, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            std::cerr << "[EnvironmentMap] alias " << what << " copy failed: "
                      << cudaGetErrorString(err) << "\n";
            return false;
        }
        return true;
    };
    if (!upload(reinterpret_cast<void**>(&m_d_aliasProb), prob.data(),  (size_t)n * sizeof(float),    "prob")) return false;
    if (!upload(reinterpret_cast<void**>(&m_d_aliasIdx),  alias.data(), (size_t)n * sizeof(uint32_t), "idx"))  return false;
    if (!upload(reinterpret_cast<void**>(&m_d_pmf),       pmf.data(),   (size_t)n * sizeof(float),    "pmf"))  return false;

    std::cout << "[EnvironmentMap] Built alias table (" << n << " texels)\n";
    return true;
}

} // namespace spectra
