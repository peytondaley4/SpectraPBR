#include "scene_hierarchy.h"

namespace spectra {

SceneHierarchy::SceneHierarchy() {
    initialize();
}

void SceneHierarchy::clear() {
    m_nodes.clear();
    m_rootIndex = UINT32_MAX;
    m_lightsGroupIndex = UINT32_MAX;
    m_instanceCount = 0;
    m_dirLightCount = 0;
    m_areaLightCount = 0;
    m_pointLightCount = 0;
}

void SceneHierarchy::initialize() {
    clear();

    // Create root node
    m_rootIndex = addNode("Scene", SceneNodeType::Root, 0, UINT32_MAX);

    // Create lights group under root
    m_lightsGroupIndex = addNode("Lights", SceneNodeType::LightsGroup, 0, m_rootIndex);
}

uint32_t SceneHierarchy::addNode(const std::string& name, SceneNodeType type,
                                  uint32_t dataIndex, uint32_t parentIndex) {
    uint32_t index = static_cast<uint32_t>(m_nodes.size());

    HierarchyNode node;
    node.name = name;
    node.type = type;
    node.dataIndex = dataIndex;
    node.parentIndex = parentIndex;
    node.expanded = true;

    m_nodes.push_back(std::move(node));

    // Add to parent's children list
    if (parentIndex != UINT32_MAX && parentIndex < m_nodes.size()) {
        m_nodes[parentIndex].childIndices.push_back(index);
    }

    return index;
}

uint32_t SceneHierarchy::addModel(const std::string& name) {
    return addNode(name, SceneNodeType::Model, 0, m_rootIndex);
}

void SceneHierarchy::addInstance(uint32_t modelIndex, uint32_t meshIndex,
                                  uint32_t instanceId, const std::string& name) {
    if (modelIndex >= m_nodes.size()) return;

    addNode(name, SceneNodeType::Instance, instanceId, modelIndex);
    m_instanceCount++;
}

void SceneHierarchy::addDirectionalLight(uint32_t index, const std::string& name) {
    addNode(name, SceneNodeType::DirectionalLight, index, m_lightsGroupIndex);
    m_dirLightCount++;
}

void SceneHierarchy::addAreaLight(uint32_t index, const std::string& name) {
    addNode(name, SceneNodeType::AreaLight, index, m_lightsGroupIndex);
    m_areaLightCount++;
}

void SceneHierarchy::addPointLight(uint32_t index, const std::string& name) {
    addNode(name, SceneNodeType::PointLight, index, m_lightsGroupIndex);
    m_pointLightCount++;
}

HierarchyNode* SceneHierarchy::getNode(uint32_t index) {
    if (index >= m_nodes.size()) return nullptr;
    return &m_nodes[index];
}

const HierarchyNode* SceneHierarchy::getNode(uint32_t index) const {
    if (index >= m_nodes.size()) return nullptr;
    return &m_nodes[index];
}

void SceneHierarchy::setExpanded(uint32_t nodeIndex, bool expanded) {
    if (nodeIndex < m_nodes.size()) {
        m_nodes[nodeIndex].expanded = expanded;
    }
}

bool SceneHierarchy::isExpanded(uint32_t nodeIndex) const {
    if (nodeIndex >= m_nodes.size()) return false;
    return m_nodes[nodeIndex].expanded;
}

uint32_t SceneHierarchy::findInstanceNode(uint32_t instanceId) const {
    for (size_t i = 0; i < m_nodes.size(); i++) {
        const auto& node = m_nodes[i];
        if (node.type == SceneNodeType::Instance && node.dataIndex == instanceId) {
            return static_cast<uint32_t>(i);
        }
    }
    return UINT32_MAX;
}

} // namespace spectra
