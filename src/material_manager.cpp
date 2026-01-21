#include "material_manager.h"
#include <iostream>

namespace spectra {

MaterialManager::~MaterialManager() {
    clear();
}

MaterialHandle MaterialManager::addMaterial(const MaterialData& matData) {
    GpuMaterial gpuMat = {};

    // Copy basic properties
    gpuMat.baseColor = matData.baseColor;
    gpuMat.metallic = matData.metallic;
    gpuMat.roughness = matData.roughness;
    gpuMat.emissive = matData.emissive;
    gpuMat.alphaMode = matData.alphaMode;
    gpuMat.alphaCutoff = matData.alphaCutoff;

    // Initialize textures to 0 (no texture)
    gpuMat.baseColorTex = 0;
    gpuMat.normalTex = 0;
    gpuMat.metallicRoughnessTex = 0;
    gpuMat.emissiveTex = 0;

    std::vector<TextureHandle> texHandles;

    // Load textures if texture manager available
    if (m_textureManager) {
        // Base color texture (sRGB)
        if (!matData.baseColorTexPath.empty()) {
            TextureHandle h = m_textureManager->loadFromFile(matData.baseColorTexPath, true);
            if (h != INVALID_TEXTURE_HANDLE) {
                gpuMat.baseColorTex = m_textureManager->getTextureObject(h);
                texHandles.push_back(h);
            }
        }

        // Normal map (linear)
        if (!matData.normalTexPath.empty()) {
            TextureHandle h = m_textureManager->loadFromFile(matData.normalTexPath, false);
            if (h != INVALID_TEXTURE_HANDLE) {
                gpuMat.normalTex = m_textureManager->getTextureObject(h);
                texHandles.push_back(h);
            }
        }

        // Metallic-roughness texture (linear)
        if (!matData.metallicRoughnessTexPath.empty()) {
            TextureHandle h = m_textureManager->loadFromFile(matData.metallicRoughnessTexPath, false);
            if (h != INVALID_TEXTURE_HANDLE) {
                gpuMat.metallicRoughnessTex = m_textureManager->getTextureObject(h);
                texHandles.push_back(h);
            }
        }

        // Emissive texture (sRGB)
        if (!matData.emissiveTexPath.empty()) {
            TextureHandle h = m_textureManager->loadFromFile(matData.emissiveTexPath, true);
            if (h != INVALID_TEXTURE_HANDLE) {
                gpuMat.emissiveTex = m_textureManager->getTextureObject(h);
                texHandles.push_back(h);
            }
        }
    }

    MaterialHandle handle = static_cast<MaterialHandle>(m_materials.size());
    m_materials.push_back(gpuMat);
    m_materialTextures.push_back(texHandles);

    std::cout << "[MaterialManager] Added material " << handle
              << " (baseColor: [" << gpuMat.baseColor.x << ", " << gpuMat.baseColor.y
              << ", " << gpuMat.baseColor.z << "], metal: " << gpuMat.metallic
              << ", rough: " << gpuMat.roughness << ")\n";

    return handle;
}

const GpuMaterial* MaterialManager::get(MaterialHandle handle) const {
    if (handle >= m_materials.size()) {
        return nullptr;
    }
    return &m_materials[handle];
}

void MaterialManager::clear() {
    // Release texture references
    if (m_textureManager) {
        for (const auto& texHandles : m_materialTextures) {
            for (TextureHandle h : texHandles) {
                m_textureManager->release(h);
            }
        }
    }

    m_materials.clear();
    m_materialTextures.clear();
    std::cout << "[MaterialManager] Cleared all materials\n";
}

MaterialHandle MaterialManager::createDefaultMaterial() {
    MaterialData defaultMat;
    defaultMat.baseColor = make_float4(0.8f, 0.8f, 0.8f, 1.0f);
    defaultMat.metallic = 0.0f;
    defaultMat.roughness = 0.5f;
    defaultMat.emissive = make_float3(0.0f, 0.0f, 0.0f);
    defaultMat.alphaMode = ALPHA_MODE_OPAQUE;
    defaultMat.alphaCutoff = 0.5f;

    return addMaterial(defaultMat);
}

} // namespace spectra
