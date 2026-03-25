#pragma once

#include "shared_types.h"
#include "texture_manager.h"
#include <vector>
#include <cstdint>

namespace spectra {

// Handle to a GPU material
using MaterialHandle = uint32_t;
constexpr MaterialHandle INVALID_MATERIAL_HANDLE = UINT32_MAX;

class MaterialManager {
public:
    MaterialManager() = default;
    ~MaterialManager();

    // Non-copyable
    MaterialManager(const MaterialManager&) = delete;
    MaterialManager& operator=(const MaterialManager&) = delete;

    // Set texture manager for loading material textures
    void setTextureManager(TextureManager* texMgr) { m_textureManager = texMgr; }

    // Add material from CPU data
    // Returns material handle (index into material array)
    MaterialHandle addMaterial(const MaterialData& matData);

    // Get GpuMaterial for a handle
    const GpuMaterial* get(MaterialHandle handle) const;

    // Update material properties at runtime (preserves texture handles)
    // Only updates scalar properties (baseColor, metallic, roughness, emissive, etc.)
    // Caller must rebuild SBT after this call for changes to reach the GPU.
    bool updateMaterial(MaterialHandle handle, const GpuMaterial& updated);

    // Get all GPU materials (for SBT)
    const std::vector<GpuMaterial>& getAllMaterials() const { return m_materials; }

    // Get material count
    size_t getMaterialCount() const { return m_materials.size(); }

    // Clear all materials
    void clear();

    // Create default material (white diffuse)
    MaterialHandle createDefaultMaterial();

private:
    TextureManager* m_textureManager = nullptr;
    std::vector<GpuMaterial> m_materials;
    std::vector<std::vector<TextureHandle>> m_materialTextures;  // Track textures per material for cleanup
};

} // namespace spectra
