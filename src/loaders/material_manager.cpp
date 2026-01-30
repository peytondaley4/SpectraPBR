#include "material_manager.h"
#include <iostream>

namespace spectra {

MaterialManager::~MaterialManager() {
    clear();
}

MaterialHandle MaterialManager::addMaterial(const MaterialData& matData) {
    GpuMaterial gpuMat = {};

    //--------------------------------------------------------------------------
    // Core PBR properties
    //--------------------------------------------------------------------------
    gpuMat.baseColor = matData.baseColor;
    gpuMat.metallic = matData.metallic;
    gpuMat.roughness = matData.roughness;
    gpuMat.emissive = matData.emissive;
    gpuMat.alphaMode = matData.alphaMode;
    gpuMat.alphaCutoff = matData.alphaCutoff;
    gpuMat.doubleSided = matData.doubleSided ? 1 : 0;

    //--------------------------------------------------------------------------
    // KHR_materials_transmission
    //--------------------------------------------------------------------------
    gpuMat.transmission = matData.transmission;
    gpuMat.ior = matData.ior;

    //--------------------------------------------------------------------------
    // KHR_materials_volume
    //--------------------------------------------------------------------------
    gpuMat.attenuationColor = matData.attenuationColor;
    gpuMat.attenuationDistance = matData.attenuationDistance;
    gpuMat.thickness = matData.thickness;

    //--------------------------------------------------------------------------
    // KHR_materials_clearcoat
    //--------------------------------------------------------------------------
    gpuMat.clearcoat = matData.clearcoat;
    gpuMat.clearcoatRoughness = matData.clearcoatRoughness;

    //--------------------------------------------------------------------------
    // KHR_materials_sheen
    //--------------------------------------------------------------------------
    gpuMat.sheenColor = matData.sheenColor;
    gpuMat.sheenRoughness = matData.sheenRoughness;

    //--------------------------------------------------------------------------
    // KHR_materials_specular
    //--------------------------------------------------------------------------
    gpuMat.specularFactor = matData.specularFactor;
    gpuMat.specularColorFactor = matData.specularColorFactor;

    //--------------------------------------------------------------------------
    // Occlusion
    //--------------------------------------------------------------------------
    gpuMat.occlusionStrength = matData.occlusionStrength;

    //--------------------------------------------------------------------------
    // Initialize all textures to 0 (no texture)
    //--------------------------------------------------------------------------
    gpuMat.baseColorTex = 0;
    gpuMat.normalTex = 0;
    gpuMat.metallicRoughnessTex = 0;
    gpuMat.emissiveTex = 0;
    gpuMat.transmissionTex = 0;
    gpuMat.clearcoatTex = 0;
    gpuMat.clearcoatRoughnessTex = 0;
    gpuMat.clearcoatNormalTex = 0;
    gpuMat.sheenColorTex = 0;
    gpuMat.sheenRoughnessTex = 0;
    gpuMat.specularTex = 0;
    gpuMat.specularColorTex = 0;
    gpuMat.occlusionTex = 0;

    std::vector<TextureHandle> texHandles;

    // Helper lambda for loading textures
    auto loadTexture = [&](const std::string& path, bool isSRGB, cudaTextureObject_t& texObj) {
        if (!path.empty() && m_textureManager) {
            TextureHandle h = m_textureManager->loadFromFile(path, isSRGB);
            if (h != INVALID_TEXTURE_HANDLE) {
                texObj = m_textureManager->getTextureObject(h);
                texHandles.push_back(h);
            }
        }
    };

    //--------------------------------------------------------------------------
    // Load all textures
    //--------------------------------------------------------------------------
    
    // Core textures
    loadTexture(matData.baseColorTexPath, true, gpuMat.baseColorTex);           // sRGB
    loadTexture(matData.normalTexPath, false, gpuMat.normalTex);                 // Linear
    loadTexture(matData.metallicRoughnessTexPath, false, gpuMat.metallicRoughnessTex); // Linear
    loadTexture(matData.emissiveTexPath, true, gpuMat.emissiveTex);              // sRGB

    // Transmission texture (linear - it's a factor)
    loadTexture(matData.transmissionTexPath, false, gpuMat.transmissionTex);

    // Clearcoat textures (all linear)
    loadTexture(matData.clearcoatTexPath, false, gpuMat.clearcoatTex);
    loadTexture(matData.clearcoatRoughnessTexPath, false, gpuMat.clearcoatRoughnessTex);
    loadTexture(matData.clearcoatNormalTexPath, false, gpuMat.clearcoatNormalTex);

    // Sheen textures
    loadTexture(matData.sheenColorTexPath, true, gpuMat.sheenColorTex);          // sRGB (color)
    loadTexture(matData.sheenRoughnessTexPath, false, gpuMat.sheenRoughnessTex); // Linear

    // Specular textures
    loadTexture(matData.specularTexPath, false, gpuMat.specularTex);             // Linear (factor)
    loadTexture(matData.specularColorTexPath, true, gpuMat.specularColorTex);    // sRGB (color)

    // Occlusion texture (linear - R channel)
    loadTexture(matData.occlusionTexPath, false, gpuMat.occlusionTex);

    //--------------------------------------------------------------------------
    // Store material
    //--------------------------------------------------------------------------
    MaterialHandle handle = static_cast<MaterialHandle>(m_materials.size());
    m_materials.push_back(gpuMat);
    m_materialTextures.push_back(texHandles);

    // Log material info
    std::cout << "[MaterialManager] Added material " << handle
              << " (baseColor: [" << gpuMat.baseColor.x << ", " << gpuMat.baseColor.y
              << ", " << gpuMat.baseColor.z << "], metal: " << gpuMat.metallic
              << ", rough: " << gpuMat.roughness;
    
    if (gpuMat.transmission > 0.0f) {
        std::cout << ", trans: " << gpuMat.transmission << ", ior: " << gpuMat.ior;
    }
    if (gpuMat.clearcoat > 0.0f) {
        std::cout << ", clearcoat: " << gpuMat.clearcoat;
    }
    if (gpuMat.sheenColor.x > 0.0f || gpuMat.sheenColor.y > 0.0f || gpuMat.sheenColor.z > 0.0f) {
        std::cout << ", sheen";
    }
    std::cout << ")\n";

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
