#pragma once

#include <cuda_runtime.h>
#include <iostream>

namespace spectra {

// Create a CUDA texture object from a cudaArray with common default settings.
// Covers the most common case: 2D array, linear filtering, normalized coords.
// Returns true on success, false on failure (logs error to stderr).
inline bool createCudaTexture(
    cudaTextureObject_t& outTex,
    cudaArray_t array,
    cudaAddressMode addressMode0 = cudaAddressModeClamp,
    cudaAddressMode addressMode1 = cudaAddressModeClamp,
    cudaFilterMode filterMode = cudaFilterModeLinear,
    cudaTextureReadMode readMode = cudaReadModeNormalizedFloat,
    bool normalizedCoords = true)
{
    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = array;

    cudaTextureDesc texDesc = {};
    texDesc.addressMode[0] = addressMode0;
    texDesc.addressMode[1] = addressMode1;
    texDesc.filterMode = filterMode;
    texDesc.readMode = readMode;
    texDesc.normalizedCoords = normalizedCoords ? 1 : 0;

    cudaError_t err = cudaCreateTextureObject(&outTex, &resDesc, &texDesc, nullptr);
    if (err != cudaSuccess) {
        std::cerr << "[CUDA] Failed to create texture object: "
                  << cudaGetErrorString(err) << "\n";
        return false;
    }
    return true;
}

} // namespace spectra
