#pragma once

#include <string>
#include <vector>
#include <cstdint>

namespace spectra {

//------------------------------------------------------------------------------
// Scene Node Types for hierarchical tree view
//------------------------------------------------------------------------------
enum class SceneNodeType : uint32_t {
    Root,               // Scene root
    Model,              // Model container
    Mesh,               // Mesh within model
    Instance,           // Instance of a mesh
    LightsGroup,        // Lights container
    PointLight,         // Point light
    DirectionalLight,   // Directional light
    AreaLight           // Area light
};

//------------------------------------------------------------------------------
// Hierarchy Node - Single node in the scene tree
//------------------------------------------------------------------------------
struct HierarchyNode {
    std::string name;
    SceneNodeType type;
    uint32_t dataIndex;           // Instance ID, light index, or mesh index depending on type
    uint32_t parentIndex;         // Parent node index (UINT32_MAX for root)
    std::vector<uint32_t> childIndices;
    bool expanded = true;
};

//------------------------------------------------------------------------------
// Scene Hierarchy - Manages the hierarchical scene tree structure
//------------------------------------------------------------------------------
class SceneHierarchy {
public:
    SceneHierarchy();
    ~SceneHierarchy() = default;

    // Non-copyable
    SceneHierarchy(const SceneHierarchy&) = delete;
    SceneHierarchy& operator=(const SceneHierarchy&) = delete;

    // Clear all nodes
    void clear();

    // Build initial structure with root and lights group
    void initialize();

    // Add a model node (returns model node index)
    uint32_t addModel(const std::string& name);

    // Add an instance to a model (meshIndex for display, instanceId for selection)
    void addInstance(uint32_t modelIndex, uint32_t meshIndex,
                     uint32_t instanceId, const std::string& name);

    // Add lights
    void addDirectionalLight(uint32_t index, const std::string& name);
    void addAreaLight(uint32_t index, const std::string& name);
    void addPointLight(uint32_t index, const std::string& name);

    // Access nodes
    const std::vector<HierarchyNode>& getNodes() const { return m_nodes; }
    HierarchyNode* getNode(uint32_t index);
    const HierarchyNode* getNode(uint32_t index) const;

    // Get root node index
    uint32_t getRootIndex() const { return m_rootIndex; }

    // Get lights group index
    uint32_t getLightsGroupIndex() const { return m_lightsGroupIndex; }

    // Toggle node expansion
    void setExpanded(uint32_t nodeIndex, bool expanded);
    bool isExpanded(uint32_t nodeIndex) const;

    // Get node count
    size_t getNodeCount() const { return m_nodes.size(); }

    // Find node by instance ID (for syncing selection)
    uint32_t findInstanceNode(uint32_t instanceId) const;

    // Get instance count
    uint32_t getInstanceCount() const { return m_instanceCount; }

    // Get light counts
    uint32_t getDirectionalLightCount() const { return m_dirLightCount; }
    uint32_t getAreaLightCount() const { return m_areaLightCount; }
    uint32_t getPointLightCount() const { return m_pointLightCount; }

private:
    uint32_t addNode(const std::string& name, SceneNodeType type,
                     uint32_t dataIndex, uint32_t parentIndex);

    std::vector<HierarchyNode> m_nodes;
    uint32_t m_rootIndex = UINT32_MAX;
    uint32_t m_lightsGroupIndex = UINT32_MAX;

    uint32_t m_instanceCount = 0;
    uint32_t m_dirLightCount = 0;
    uint32_t m_areaLightCount = 0;
    uint32_t m_pointLightCount = 0;
};

} // namespace spectra
