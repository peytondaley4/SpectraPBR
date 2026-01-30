#pragma once

#include "shared_types.h"
#include <filesystem>
#include <memory>
#include <optional>
#include <string>

namespace spectra {

class ModelLoader {
public:
    ModelLoader() = default;
    ~ModelLoader() = default;

    // Non-copyable
    ModelLoader(const ModelLoader&) = delete;
    ModelLoader& operator=(const ModelLoader&) = delete;

    // Load a glTF/glb file
    // Returns loaded model on success, or nullopt on failure
    std::optional<LoadedModel> load(const std::filesystem::path& path);

    // Get last error message
    const std::string& getLastError() const { return m_lastError; }

private:
    std::string m_lastError;

    // Helper to generate flat normals if missing
    void generateFlatNormals(MeshData& mesh);

    // Helper to generate tangents using MikkTSpace
    void generateTangents(MeshData& mesh);
};

} // namespace spectra
