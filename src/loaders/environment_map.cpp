#include "environment_map.h"
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
    // Destroy texture objects
    if (m_texture) {
        cudaDestroyTextureObject(m_texture);
        m_texture = 0;
    }
    if (m_conditionalCDF) {
        cudaDestroyTextureObject(m_conditionalCDF);
        m_conditionalCDF = 0;
    }
    if (m_marginalCDF) {
        cudaDestroyTextureObject(m_marginalCDF);
        m_marginalCDF = 0;
    }

    // Free CUDA arrays
    if (m_textureArray) {
        cudaFreeArray(m_textureArray);
        m_textureArray = nullptr;
    }
    if (m_conditionalCDFArray) {
        cudaFreeArray(m_conditionalCDFArray);
        m_conditionalCDFArray = nullptr;
    }
    if (m_marginalCDFArray) {
        cudaFreeArray(m_marginalCDFArray);
        m_marginalCDFArray = nullptr;
    }

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

    // Build importance sampling CDFs
    if (!buildCDFs(data)) {
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

    // Create texture object
    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = m_textureArray;

    cudaTextureDesc texDesc = {};
    texDesc.addressMode[0] = cudaAddressModeWrap;   // Wrap horizontally
    texDesc.addressMode[1] = cudaAddressModeClamp;  // Clamp vertically
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeElementType;     // Return float directly
    texDesc.normalizedCoords = 1;                   // Use [0,1] coordinates

    err = cudaCreateTextureObject(&m_texture, &resDesc, &texDesc, nullptr);
    if (err != cudaSuccess) {
        std::cerr << "[EnvironmentMap] Failed to create texture object: " 
                  << cudaGetErrorString(err) << "\n";
        return false;
    }

    return true;
}

bool EnvironmentMap::buildCDFs(const float* rgbData) {
    // Compute luminance for each pixel, weighted by solid angle (sin theta)
    // For equirectangular maps, each row represents a different latitude
    
    std::vector<float> luminance(m_width * m_height);
    std::vector<float> rowSums(m_height, 0.0f);

    const float PI = 3.14159265358979323846f;

    for (uint32_t y = 0; y < m_height; y++) {
        // Compute sin(theta) weight for this row
        // v goes from 0 (top) to 1 (bottom)
        // theta goes from 0 (north pole) to PI (south pole)
        float v = (y + 0.5f) / m_height;
        float theta = v * PI;
        float sinTheta = std::sin(theta);

        for (uint32_t x = 0; x < m_width; x++) {
            uint32_t idx = y * m_width + x;
            float r = rgbData[idx * 3 + 0];
            float g = rgbData[idx * 3 + 1];
            float b = rgbData[idx * 3 + 2];

            // Compute luminance using standard coefficients
            float lum = 0.2126f * r + 0.7152f * g + 0.0722f * b;

            // Weight by solid angle
            float weightedLum = lum * sinTheta;
            luminance[idx] = weightedLum;
            rowSums[y] += weightedLum;
        }
    }

    // Compute total luminance
    m_totalLuminance = 0.0f;
    for (uint32_t y = 0; y < m_height; y++) {
        m_totalLuminance += rowSums[y];
    }

    if (m_totalLuminance <= 0.0f) {
        std::cerr << "[EnvironmentMap] Warning: Environment map has zero luminance\n";
        m_totalLuminance = 1.0f;  // Avoid division by zero
    }

    // Build conditional CDF (per row) - P(u|v)
    // Each row stores the CDF of selecting column u given we're in row v
    std::vector<float> conditionalCDF(m_width * m_height);

    for (uint32_t y = 0; y < m_height; y++) {
        float rowSum = rowSums[y];
        if (rowSum <= 0.0f) rowSum = 1.0f;  // Avoid division by zero

        float cumulative = 0.0f;
        for (uint32_t x = 0; x < m_width; x++) {
            uint32_t idx = y * m_width + x;
            cumulative += luminance[idx] / rowSum;
            conditionalCDF[idx] = cumulative;
        }
        // Ensure last element is exactly 1.0
        conditionalCDF[y * m_width + m_width - 1] = 1.0f;
    }

    // Build marginal CDF - P(v)
    // Probability of selecting row v based on sum of luminance in that row
    std::vector<float> marginalCDF(m_height);
    float cumulative = 0.0f;
    for (uint32_t y = 0; y < m_height; y++) {
        cumulative += rowSums[y] / m_totalLuminance;
        marginalCDF[y] = cumulative;
    }
    // Ensure last element is exactly 1.0
    marginalCDF[m_height - 1] = 1.0f;

    // Create CUDA array for conditional CDF (2D texture)
    {
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
        cudaError_t err = cudaMallocArray(&m_conditionalCDFArray, &channelDesc, m_width, m_height);
        if (err != cudaSuccess) {
            std::cerr << "[EnvironmentMap] Failed to allocate conditional CDF array: " 
                      << cudaGetErrorString(err) << "\n";
            return false;
        }

        err = cudaMemcpy2DToArray(
            m_conditionalCDFArray, 0, 0,
            conditionalCDF.data(),
            m_width * sizeof(float),
            m_width * sizeof(float),
            m_height,
            cudaMemcpyHostToDevice
        );
        if (err != cudaSuccess) {
            std::cerr << "[EnvironmentMap] Failed to copy conditional CDF: " 
                      << cudaGetErrorString(err) << "\n";
            return false;
        }

        cudaResourceDesc resDesc = {};
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = m_conditionalCDFArray;

        cudaTextureDesc texDesc = {};
        texDesc.addressMode[0] = cudaAddressModeClamp;
        texDesc.addressMode[1] = cudaAddressModeClamp;
        texDesc.filterMode = cudaFilterModePoint;  // No interpolation for CDF lookup
        texDesc.readMode = cudaReadModeElementType;
        texDesc.normalizedCoords = 0;  // Use pixel coordinates for binary search

        err = cudaCreateTextureObject(&m_conditionalCDF, &resDesc, &texDesc, nullptr);
        if (err != cudaSuccess) {
            std::cerr << "[EnvironmentMap] Failed to create conditional CDF texture: " 
                      << cudaGetErrorString(err) << "\n";
            return false;
        }
    }

    // Create CUDA array for marginal CDF (1D texture stored as 2D with height=1)
    {
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
        cudaError_t err = cudaMallocArray(&m_marginalCDFArray, &channelDesc, m_height, 1);
        if (err != cudaSuccess) {
            std::cerr << "[EnvironmentMap] Failed to allocate marginal CDF array: " 
                      << cudaGetErrorString(err) << "\n";
            return false;
        }

        err = cudaMemcpy2DToArray(
            m_marginalCDFArray, 0, 0,
            marginalCDF.data(),
            m_height * sizeof(float),
            m_height * sizeof(float),
            1,
            cudaMemcpyHostToDevice
        );
        if (err != cudaSuccess) {
            std::cerr << "[EnvironmentMap] Failed to copy marginal CDF: " 
                      << cudaGetErrorString(err) << "\n";
            return false;
        }

        cudaResourceDesc resDesc = {};
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = m_marginalCDFArray;

        cudaTextureDesc texDesc = {};
        texDesc.addressMode[0] = cudaAddressModeClamp;
        texDesc.filterMode = cudaFilterModePoint;  // No interpolation for CDF lookup
        texDesc.readMode = cudaReadModeElementType;
        texDesc.normalizedCoords = 0;  // Use pixel coordinates

        err = cudaCreateTextureObject(&m_marginalCDF, &resDesc, &texDesc, nullptr);
        if (err != cudaSuccess) {
            std::cerr << "[EnvironmentMap] Failed to create marginal CDF texture: " 
                      << cudaGetErrorString(err) << "\n";
            return false;
        }
    }

    std::cout << "[EnvironmentMap] Built importance sampling CDFs\n";
    return true;
}

} // namespace spectra
