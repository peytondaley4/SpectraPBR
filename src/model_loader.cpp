#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <tiny_gltf.h>

#include "model_loader.h"
#include <iostream>
#include <unordered_map>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

namespace spectra {

// Helper to get typed accessor data
template<typename T>
static const T* getAccessorData(const tinygltf::Model& model, int accessorIndex) {
    if (accessorIndex < 0) return nullptr;
    const auto& accessor = model.accessors[accessorIndex];
    const auto& bufferView = model.bufferViews[accessor.bufferView];
    const auto& buffer = model.buffers[bufferView.buffer];
    return reinterpret_cast<const T*>(&buffer.data[bufferView.byteOffset + accessor.byteOffset]);
}

// Get stride for an accessor (0 means tightly packed)
static size_t getAccessorStride(const tinygltf::Model& model, int accessorIndex) {
    if (accessorIndex < 0) return 0;
    const auto& accessor = model.accessors[accessorIndex];
    const auto& bufferView = model.bufferViews[accessor.bufferView];
    if (bufferView.byteStride > 0) {
        return bufferView.byteStride;
    }
    // Tightly packed
    return tinygltf::GetComponentSizeInBytes(accessor.componentType) *
           tinygltf::GetNumComponentsInType(accessor.type);
}

// Get number of elements in accessor
static size_t getAccessorCount(const tinygltf::Model& model, int accessorIndex) {
    if (accessorIndex < 0) return 0;
    return model.accessors[accessorIndex].count;
}

std::optional<LoadedModel> ModelLoader::load(const std::filesystem::path& path) {
    tinygltf::Model model;
    tinygltf::TinyGLTF loader;
    std::string err, warn;

    bool success = false;
    std::string ext = path.extension().string();

    if (ext == ".glb") {
        success = loader.LoadBinaryFromFile(&model, &err, &warn, path.string());
    } else {
        success = loader.LoadASCIIFromFile(&model, &err, &warn, path.string());
    }

    if (!warn.empty()) {
        std::cout << "[ModelLoader] Warning: " << warn << "\n";
    }

    if (!err.empty()) {
        m_lastError = err;
        std::cerr << "[ModelLoader] Error: " << err << "\n";
    }

    if (!success) {
        m_lastError = "Failed to load glTF file: " + path.string();
        return std::nullopt;
    }

    LoadedModel result;
    result.name = path.stem().string();

    std::filesystem::path basePath = path.parent_path();

    // Load materials
    for (const auto& material : model.materials) {
        MaterialData matData;

        // Textures helper
        auto getTexturePath = [&](int texIndex) -> std::string {
            if (texIndex < 0) return "";
            if (texIndex >= static_cast<int>(model.textures.size())) return "";
            const auto& texture = model.textures[texIndex];
            if (texture.source < 0) return "";
            if (texture.source >= static_cast<int>(model.images.size())) return "";
            const auto& image = model.images[texture.source];
            if (image.uri.empty()) return "";  // Embedded texture
            return (basePath / image.uri).string();
        };

        // Check for KHR_materials_pbrSpecularGlossiness extension
        bool hasSpecGloss = false;
        auto specGlossIt = material.extensions.find("KHR_materials_pbrSpecularGlossiness");
        if (specGlossIt != material.extensions.end() && specGlossIt->second.IsObject()) {
            hasSpecGloss = true;
            const auto& sg = specGlossIt->second;

            // Diffuse factor (equivalent to baseColor)
            if (sg.Has("diffuseFactor") && sg.Get("diffuseFactor").IsArray()) {
                const auto& df = sg.Get("diffuseFactor");
                if (df.ArrayLen() >= 4) {
                    matData.baseColor = make_float4(
                        static_cast<float>(df.Get(0).IsNumber() ? df.Get(0).GetNumberAsDouble() : 1.0),
                        static_cast<float>(df.Get(1).IsNumber() ? df.Get(1).GetNumberAsDouble() : 1.0),
                        static_cast<float>(df.Get(2).IsNumber() ? df.Get(2).GetNumberAsDouble() : 1.0),
                        static_cast<float>(df.Get(3).IsNumber() ? df.Get(3).GetNumberAsDouble() : 1.0)
                    );
                }
            }

            // Diffuse texture (equivalent to baseColorTexture)
            if (sg.Has("diffuseTexture") && sg.Get("diffuseTexture").IsObject()) {
                const auto& dt = sg.Get("diffuseTexture");
                if (dt.Has("index") && dt.Get("index").IsInt()) {
                    matData.baseColorTexPath = getTexturePath(dt.Get("index").GetNumberAsInt());
                }
            }

            // Glossiness to roughness conversion (roughness = 1 - glossiness)
            float glossiness = 1.0f;
            if (sg.Has("glossinessFactor") && sg.Get("glossinessFactor").IsNumber()) {
                glossiness = static_cast<float>(sg.Get("glossinessFactor").GetNumberAsDouble());
            }
            matData.roughness = 1.0f - glossiness;

            // Specular-glossiness workflow typically doesn't use metallic
            matData.metallic = 0.0f;
        }

        // Fall back to standard pbrMetallicRoughness if no specular-glossiness
        if (!hasSpecGloss) {
            // Base color
            if (material.pbrMetallicRoughness.baseColorFactor.size() >= 4) {
                matData.baseColor = make_float4(
                    static_cast<float>(material.pbrMetallicRoughness.baseColorFactor[0]),
                    static_cast<float>(material.pbrMetallicRoughness.baseColorFactor[1]),
                    static_cast<float>(material.pbrMetallicRoughness.baseColorFactor[2]),
                    static_cast<float>(material.pbrMetallicRoughness.baseColorFactor[3])
                );
            }

            // Metallic and roughness
            matData.metallic = static_cast<float>(material.pbrMetallicRoughness.metallicFactor);
            matData.roughness = static_cast<float>(material.pbrMetallicRoughness.roughnessFactor);

            // Base color texture
            matData.baseColorTexPath = getTexturePath(
                material.pbrMetallicRoughness.baseColorTexture.index);
            matData.metallicRoughnessTexPath = getTexturePath(
                material.pbrMetallicRoughness.metallicRoughnessTexture.index);
        }

        // Emissive (common to both workflows)
        if (material.emissiveFactor.size() >= 3) {
            matData.emissive = make_float3(
                static_cast<float>(material.emissiveFactor[0]),
                static_cast<float>(material.emissiveFactor[1]),
                static_cast<float>(material.emissiveFactor[2])
            );
        }

        // Alpha mode (common to both workflows)
        if (material.alphaMode == "MASK") {
            matData.alphaMode = ALPHA_MODE_MASK;
            matData.alphaCutoff = static_cast<float>(material.alphaCutoff);
        } else if (material.alphaMode == "BLEND") {
            matData.alphaMode = ALPHA_MODE_BLEND;
        } else {
            matData.alphaMode = ALPHA_MODE_OPAQUE;
        }

        // Normal texture (common to both workflows)
        matData.normalTexPath = getTexturePath(material.normalTexture.index);
        matData.emissiveTexPath = getTexturePath(material.emissiveTexture.index);

        // Occlusion texture
        matData.occlusionTexPath = getTexturePath(material.occlusionTexture.index);
        if (material.occlusionTexture.strength >= 0.0) {
            matData.occlusionStrength = static_cast<float>(material.occlusionTexture.strength);
        }

        // Double-sided flag
        matData.doubleSided = material.doubleSided;

        //----------------------------------------------------------------------
        // KHR_materials_transmission (Glass/Water)
        //----------------------------------------------------------------------
        auto transIt = material.extensions.find("KHR_materials_transmission");
        if (transIt != material.extensions.end() && transIt->second.IsObject()) {
            const auto& trans = transIt->second;

            if (trans.Has("transmissionFactor") && trans.Get("transmissionFactor").IsNumber()) {
                matData.transmission = static_cast<float>(trans.Get("transmissionFactor").GetNumberAsDouble());
            }

            if (trans.Has("transmissionTexture") && trans.Get("transmissionTexture").IsObject()) {
                const auto& tt = trans.Get("transmissionTexture");
                if (tt.Has("index") && tt.Get("index").IsInt()) {
                    matData.transmissionTexPath = getTexturePath(tt.Get("index").GetNumberAsInt());
                }
            }
        }

        //----------------------------------------------------------------------
        // KHR_materials_ior (Index of Refraction)
        //----------------------------------------------------------------------
        auto iorIt = material.extensions.find("KHR_materials_ior");
        if (iorIt != material.extensions.end() && iorIt->second.IsObject()) {
            const auto& iorExt = iorIt->second;
            if (iorExt.Has("ior") && iorExt.Get("ior").IsNumber()) {
                matData.ior = static_cast<float>(iorExt.Get("ior").GetNumberAsDouble());
            }
        }

        //----------------------------------------------------------------------
        // KHR_materials_volume (Absorption)
        //----------------------------------------------------------------------
        auto volumeIt = material.extensions.find("KHR_materials_volume");
        if (volumeIt != material.extensions.end() && volumeIt->second.IsObject()) {
            const auto& vol = volumeIt->second;

            if (vol.Has("thicknessFactor") && vol.Get("thicknessFactor").IsNumber()) {
                matData.thickness = static_cast<float>(vol.Get("thicknessFactor").GetNumberAsDouble());
            }

            if (vol.Has("attenuationDistance") && vol.Get("attenuationDistance").IsNumber()) {
                matData.attenuationDistance = static_cast<float>(vol.Get("attenuationDistance").GetNumberAsDouble());
            }

            if (vol.Has("attenuationColor") && vol.Get("attenuationColor").IsArray()) {
                const auto& ac = vol.Get("attenuationColor");
                if (ac.ArrayLen() >= 3) {
                    matData.attenuationColor = make_float3(
                        static_cast<float>(ac.Get(0).GetNumberAsDouble()),
                        static_cast<float>(ac.Get(1).GetNumberAsDouble()),
                        static_cast<float>(ac.Get(2).GetNumberAsDouble())
                    );
                }
            }
        }

        //----------------------------------------------------------------------
        // KHR_materials_clearcoat (Car Paint)
        //----------------------------------------------------------------------
        auto clearcoatIt = material.extensions.find("KHR_materials_clearcoat");
        if (clearcoatIt != material.extensions.end() && clearcoatIt->second.IsObject()) {
            const auto& cc = clearcoatIt->second;

            if (cc.Has("clearcoatFactor") && cc.Get("clearcoatFactor").IsNumber()) {
                matData.clearcoat = static_cast<float>(cc.Get("clearcoatFactor").GetNumberAsDouble());
            }

            if (cc.Has("clearcoatRoughnessFactor") && cc.Get("clearcoatRoughnessFactor").IsNumber()) {
                matData.clearcoatRoughness = static_cast<float>(cc.Get("clearcoatRoughnessFactor").GetNumberAsDouble());
            }

            if (cc.Has("clearcoatTexture") && cc.Get("clearcoatTexture").IsObject()) {
                const auto& cct = cc.Get("clearcoatTexture");
                if (cct.Has("index") && cct.Get("index").IsInt()) {
                    matData.clearcoatTexPath = getTexturePath(cct.Get("index").GetNumberAsInt());
                }
            }

            if (cc.Has("clearcoatRoughnessTexture") && cc.Get("clearcoatRoughnessTexture").IsObject()) {
                const auto& ccrt = cc.Get("clearcoatRoughnessTexture");
                if (ccrt.Has("index") && ccrt.Get("index").IsInt()) {
                    matData.clearcoatRoughnessTexPath = getTexturePath(ccrt.Get("index").GetNumberAsInt());
                }
            }

            if (cc.Has("clearcoatNormalTexture") && cc.Get("clearcoatNormalTexture").IsObject()) {
                const auto& ccnt = cc.Get("clearcoatNormalTexture");
                if (ccnt.Has("index") && ccnt.Get("index").IsInt()) {
                    matData.clearcoatNormalTexPath = getTexturePath(ccnt.Get("index").GetNumberAsInt());
                }
            }
        }

        //----------------------------------------------------------------------
        // KHR_materials_sheen (Cloth/Velvet)
        //----------------------------------------------------------------------
        auto sheenIt = material.extensions.find("KHR_materials_sheen");
        if (sheenIt != material.extensions.end() && sheenIt->second.IsObject()) {
            const auto& sheen = sheenIt->second;

            if (sheen.Has("sheenColorFactor") && sheen.Get("sheenColorFactor").IsArray()) {
                const auto& sc = sheen.Get("sheenColorFactor");
                if (sc.ArrayLen() >= 3) {
                    matData.sheenColor = make_float3(
                        static_cast<float>(sc.Get(0).GetNumberAsDouble()),
                        static_cast<float>(sc.Get(1).GetNumberAsDouble()),
                        static_cast<float>(sc.Get(2).GetNumberAsDouble())
                    );
                }
            }

            if (sheen.Has("sheenRoughnessFactor") && sheen.Get("sheenRoughnessFactor").IsNumber()) {
                matData.sheenRoughness = static_cast<float>(sheen.Get("sheenRoughnessFactor").GetNumberAsDouble());
            }

            if (sheen.Has("sheenColorTexture") && sheen.Get("sheenColorTexture").IsObject()) {
                const auto& sct = sheen.Get("sheenColorTexture");
                if (sct.Has("index") && sct.Get("index").IsInt()) {
                    matData.sheenColorTexPath = getTexturePath(sct.Get("index").GetNumberAsInt());
                }
            }

            if (sheen.Has("sheenRoughnessTexture") && sheen.Get("sheenRoughnessTexture").IsObject()) {
                const auto& srt = sheen.Get("sheenRoughnessTexture");
                if (srt.Has("index") && srt.Get("index").IsInt()) {
                    matData.sheenRoughnessTexPath = getTexturePath(srt.Get("index").GetNumberAsInt());
                }
            }
        }

        //----------------------------------------------------------------------
        // KHR_materials_specular (Fine-tune reflectance)
        //----------------------------------------------------------------------
        auto specularIt = material.extensions.find("KHR_materials_specular");
        if (specularIt != material.extensions.end() && specularIt->second.IsObject()) {
            const auto& spec = specularIt->second;

            if (spec.Has("specularFactor") && spec.Get("specularFactor").IsNumber()) {
                matData.specularFactor = static_cast<float>(spec.Get("specularFactor").GetNumberAsDouble());
            }

            if (spec.Has("specularColorFactor") && spec.Get("specularColorFactor").IsArray()) {
                const auto& scf = spec.Get("specularColorFactor");
                if (scf.ArrayLen() >= 3) {
                    matData.specularColorFactor = make_float3(
                        static_cast<float>(scf.Get(0).GetNumberAsDouble()),
                        static_cast<float>(scf.Get(1).GetNumberAsDouble()),
                        static_cast<float>(scf.Get(2).GetNumberAsDouble())
                    );
                }
            }

            if (spec.Has("specularTexture") && spec.Get("specularTexture").IsObject()) {
                const auto& st = spec.Get("specularTexture");
                if (st.Has("index") && st.Get("index").IsInt()) {
                    matData.specularTexPath = getTexturePath(st.Get("index").GetNumberAsInt());
                }
            }

            if (spec.Has("specularColorTexture") && spec.Get("specularColorTexture").IsObject()) {
                const auto& sct = spec.Get("specularColorTexture");
                if (sct.Has("index") && sct.Get("index").IsInt()) {
                    matData.specularColorTexPath = getTexturePath(sct.Get("index").GetNumberAsInt());
                }
            }
        }

        result.materials.push_back(matData);
    }

    // Add default material if none exist
    if (result.materials.empty()) {
        result.materials.push_back(MaterialData{});
    }

    // Load meshes
    for (size_t meshIdx = 0; meshIdx < model.meshes.size(); ++meshIdx) {
        const auto& mesh = model.meshes[meshIdx];

        for (const auto& primitive : mesh.primitives) {
            if (primitive.mode != TINYGLTF_MODE_TRIANGLES) {
                std::cout << "[ModelLoader] Skipping non-triangle primitive\n";
                continue;
            }

            MeshData meshData;
            meshData.materialIndex = (primitive.material >= 0)
                ? static_cast<uint32_t>(primitive.material) : 0;

            // Get attribute accessors
            int posAccessor = -1, normAccessor = -1, uvAccessor = -1, tanAccessor = -1;

            auto posIt = primitive.attributes.find("POSITION");
            if (posIt != primitive.attributes.end()) posAccessor = posIt->second;

            auto normIt = primitive.attributes.find("NORMAL");
            if (normIt != primitive.attributes.end()) normAccessor = normIt->second;

            auto uvIt = primitive.attributes.find("TEXCOORD_0");
            if (uvIt != primitive.attributes.end()) uvAccessor = uvIt->second;

            auto tanIt = primitive.attributes.find("TANGENT");
            if (tanIt != primitive.attributes.end()) tanAccessor = tanIt->second;

            if (posAccessor < 0) {
                std::cerr << "[ModelLoader] Mesh primitive has no POSITION attribute\n";
                continue;
            }

            size_t vertexCount = getAccessorCount(model, posAccessor);
            meshData.vertices.resize(vertexCount);

            // Get data pointers
            const float* positions = getAccessorData<float>(model, posAccessor);
            const float* normals = (normAccessor >= 0) ? getAccessorData<float>(model, normAccessor) : nullptr;
            const float* uvs = (uvAccessor >= 0) ? getAccessorData<float>(model, uvAccessor) : nullptr;
            const float* tangents = (tanAccessor >= 0) ? getAccessorData<float>(model, tanAccessor) : nullptr;

            // Get strides
            size_t posStride = getAccessorStride(model, posAccessor) / sizeof(float);
            size_t normStride = normals ? getAccessorStride(model, normAccessor) / sizeof(float) : 0;
            size_t uvStride = uvs ? getAccessorStride(model, uvAccessor) / sizeof(float) : 0;
            size_t tanStride = tangents ? getAccessorStride(model, tanAccessor) / sizeof(float) : 0;

            // Fill vertices
            for (size_t i = 0; i < vertexCount; ++i) {
                GpuVertex& v = meshData.vertices[i];

                // Position
                v.position = make_float3(
                    positions[i * posStride + 0],
                    positions[i * posStride + 1],
                    positions[i * posStride + 2]
                );

                // Normal (default up if missing)
                if (normals) {
                    v.normal = make_float3(
                        normals[i * normStride + 0],
                        normals[i * normStride + 1],
                        normals[i * normStride + 2]
                    );
                } else {
                    v.normal = make_float3(0.0f, 1.0f, 0.0f);
                }

                // UV (default 0 if missing)
                if (uvs) {
                    v.u = uvs[i * uvStride + 0];
                    v.v = uvs[i * uvStride + 1];
                } else {
                    v.u = 0.0f;
                    v.v = 0.0f;
                }

                // Tangent (will generate if missing)
                if (tangents) {
                    v.tangent = make_float4(
                        tangents[i * tanStride + 0],
                        tangents[i * tanStride + 1],
                        tangents[i * tanStride + 2],
                        tangents[i * tanStride + 3]
                    );
                } else {
                    v.tangent = make_float4(1.0f, 0.0f, 0.0f, 1.0f);
                }
            }

            // Load indices
            if (primitive.indices >= 0) {
                const auto& indicesAccessor = model.accessors[primitive.indices];
                const auto& indicesBufferView = model.bufferViews[indicesAccessor.bufferView];
                const auto& indicesBuffer = model.buffers[indicesBufferView.buffer];

                meshData.indices.resize(indicesAccessor.count);

                const uint8_t* indexData = &indicesBuffer.data[
                    indicesBufferView.byteOffset + indicesAccessor.byteOffset];

                for (size_t i = 0; i < indicesAccessor.count; ++i) {
                    if (indicesAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT) {
                        meshData.indices[i] = reinterpret_cast<const uint16_t*>(indexData)[i];
                    } else if (indicesAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT) {
                        meshData.indices[i] = reinterpret_cast<const uint32_t*>(indexData)[i];
                    } else if (indicesAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE) {
                        meshData.indices[i] = indexData[i];
                    }
                }
            } else {
                // Non-indexed geometry - generate indices
                meshData.indices.resize(vertexCount);
                for (size_t i = 0; i < vertexCount; ++i) {
                    meshData.indices[i] = static_cast<uint32_t>(i);
                }
            }

            // Generate normals if missing
            if (!normals) {
                generateFlatNormals(meshData);
            }

            // Generate tangents if missing
            if (!tangents) {
                generateTangents(meshData);
            }

            result.meshes.push_back(std::move(meshData));
        }
    }

    // Process scene graph to get instances
    std::function<void(int, glm::mat4)> processNode = [&](int nodeIdx, glm::mat4 parentTransform) {
        const auto& node = model.nodes[nodeIdx];

        // Compute local transform
        glm::mat4 localTransform(1.0f);

        if (node.matrix.size() == 16) {
            // Use matrix directly
            for (int i = 0; i < 16; ++i) {
                localTransform[i / 4][i % 4] = static_cast<float>(node.matrix[i]);
            }
        } else {
            // Build from TRS
            glm::vec3 translation(0.0f);
            glm::quat rotation(1.0f, 0.0f, 0.0f, 0.0f);
            glm::vec3 scale(1.0f);

            if (node.translation.size() == 3) {
                translation = glm::vec3(
                    static_cast<float>(node.translation[0]),
                    static_cast<float>(node.translation[1]),
                    static_cast<float>(node.translation[2])
                );
            }
            if (node.rotation.size() == 4) {
                rotation = glm::quat(
                    static_cast<float>(node.rotation[3]),  // w
                    static_cast<float>(node.rotation[0]),  // x
                    static_cast<float>(node.rotation[1]),  // y
                    static_cast<float>(node.rotation[2])   // z
                );
            }
            if (node.scale.size() == 3) {
                scale = glm::vec3(
                    static_cast<float>(node.scale[0]),
                    static_cast<float>(node.scale[1]),
                    static_cast<float>(node.scale[2])
                );
            }

            glm::mat4 T = glm::translate(glm::mat4(1.0f), translation);
            glm::mat4 R = glm::mat4_cast(rotation);
            glm::mat4 S = glm::scale(glm::mat4(1.0f), scale);
            localTransform = T * R * S;
        }

        glm::mat4 worldTransform = parentTransform * localTransform;

        // If this node has a mesh, create instances for it
        if (node.mesh >= 0) {
            // Count primitives up to this mesh to get correct mesh indices
            size_t baseMeshIdx = 0;
            for (int m = 0; m < node.mesh; ++m) {
                baseMeshIdx += model.meshes[m].primitives.size();
            }

            for (size_t p = 0; p < model.meshes[node.mesh].primitives.size(); ++p) {
                ModelInstance instance;
                instance.meshIndex = static_cast<uint32_t>(baseMeshIdx + p);

                // Store as 3x4 row-major (OptiX format)
                const float* m = glm::value_ptr(glm::transpose(worldTransform));
                for (int i = 0; i < 12; ++i) {
                    instance.transform[i] = m[i];
                }

                result.instances.push_back(instance);
            }
        }

        // Process children
        for (int childIdx : node.children) {
            processNode(childIdx, worldTransform);
        }
    };

    // Process all scenes (typically just one)
    for (const auto& scene : model.scenes) {
        for (int nodeIdx : scene.nodes) {
            processNode(nodeIdx, glm::mat4(1.0f));
        }
    }

    // If no instances were created (no scene), create one per mesh
    if (result.instances.empty()) {
        for (size_t i = 0; i < result.meshes.size(); ++i) {
            ModelInstance instance;
            instance.meshIndex = static_cast<uint32_t>(i);
            // Identity transform
            instance.transform[0] = 1.0f; instance.transform[1] = 0.0f; instance.transform[2] = 0.0f; instance.transform[3] = 0.0f;
            instance.transform[4] = 0.0f; instance.transform[5] = 1.0f; instance.transform[6] = 0.0f; instance.transform[7] = 0.0f;
            instance.transform[8] = 0.0f; instance.transform[9] = 0.0f; instance.transform[10] = 1.0f; instance.transform[11] = 0.0f;
            result.instances.push_back(instance);
        }
    }

    std::cout << "[ModelLoader] Loaded: " << result.name
              << " (" << result.meshes.size() << " meshes, "
              << result.materials.size() << " materials, "
              << result.instances.size() << " instances)\n";

    return result;
}

void ModelLoader::generateFlatNormals(MeshData& mesh) {
    // Generate flat normals per face
    for (size_t i = 0; i + 2 < mesh.indices.size(); i += 3) {
        uint32_t i0 = mesh.indices[i + 0];
        uint32_t i1 = mesh.indices[i + 1];
        uint32_t i2 = mesh.indices[i + 2];

        glm::vec3 p0(mesh.vertices[i0].position.x,
                     mesh.vertices[i0].position.y,
                     mesh.vertices[i0].position.z);
        glm::vec3 p1(mesh.vertices[i1].position.x,
                     mesh.vertices[i1].position.y,
                     mesh.vertices[i1].position.z);
        glm::vec3 p2(mesh.vertices[i2].position.x,
                     mesh.vertices[i2].position.y,
                     mesh.vertices[i2].position.z);

        glm::vec3 normal = glm::normalize(glm::cross(p1 - p0, p2 - p0));

        float3 n = make_float3(normal.x, normal.y, normal.z);
        mesh.vertices[i0].normal = n;
        mesh.vertices[i1].normal = n;
        mesh.vertices[i2].normal = n;
    }
}

void ModelLoader::generateTangents(MeshData& mesh) {
    // Simple tangent generation based on UV gradients
    // For production, use MikkTSpace

    // Initialize tangents to zero
    std::vector<glm::vec3> tangents(mesh.vertices.size(), glm::vec3(0.0f));
    std::vector<glm::vec3> bitangents(mesh.vertices.size(), glm::vec3(0.0f));

    for (size_t i = 0; i + 2 < mesh.indices.size(); i += 3) {
        uint32_t i0 = mesh.indices[i + 0];
        uint32_t i1 = mesh.indices[i + 1];
        uint32_t i2 = mesh.indices[i + 2];

        const GpuVertex& v0 = mesh.vertices[i0];
        const GpuVertex& v1 = mesh.vertices[i1];
        const GpuVertex& v2 = mesh.vertices[i2];

        glm::vec3 p0(v0.position.x, v0.position.y, v0.position.z);
        glm::vec3 p1(v1.position.x, v1.position.y, v1.position.z);
        glm::vec3 p2(v2.position.x, v2.position.y, v2.position.z);

        glm::vec2 uv0(v0.u, v0.v);
        glm::vec2 uv1(v1.u, v1.v);
        glm::vec2 uv2(v2.u, v2.v);

        glm::vec3 edge1 = p1 - p0;
        glm::vec3 edge2 = p2 - p0;
        glm::vec2 deltaUV1 = uv1 - uv0;
        glm::vec2 deltaUV2 = uv2 - uv0;

        float denom = deltaUV1.x * deltaUV2.y - deltaUV2.x * deltaUV1.y;
        if (std::abs(denom) < 1e-6f) {
            continue;  // Degenerate UV
        }

        float f = 1.0f / denom;

        glm::vec3 tangent;
        tangent.x = f * (deltaUV2.y * edge1.x - deltaUV1.y * edge2.x);
        tangent.y = f * (deltaUV2.y * edge1.y - deltaUV1.y * edge2.y);
        tangent.z = f * (deltaUV2.y * edge1.z - deltaUV1.y * edge2.z);

        glm::vec3 bitangent;
        bitangent.x = f * (-deltaUV2.x * edge1.x + deltaUV1.x * edge2.x);
        bitangent.y = f * (-deltaUV2.x * edge1.y + deltaUV1.x * edge2.y);
        bitangent.z = f * (-deltaUV2.x * edge1.z + deltaUV1.x * edge2.z);

        tangents[i0] += tangent;
        tangents[i1] += tangent;
        tangents[i2] += tangent;

        bitangents[i0] += bitangent;
        bitangents[i1] += bitangent;
        bitangents[i2] += bitangent;
    }

    // Orthonormalize and compute handedness
    for (size_t i = 0; i < mesh.vertices.size(); ++i) {
        GpuVertex& v = mesh.vertices[i];
        glm::vec3 n(v.normal.x, v.normal.y, v.normal.z);
        glm::vec3 t = tangents[i];
        glm::vec3 b = bitangents[i];

        // Gram-Schmidt orthonormalize
        t = glm::normalize(t - n * glm::dot(n, t));

        // Calculate handedness
        float w = (glm::dot(glm::cross(n, t), b) < 0.0f) ? -1.0f : 1.0f;

        v.tangent = make_float4(t.x, t.y, t.z, w);
    }
}

} // namespace spectra
