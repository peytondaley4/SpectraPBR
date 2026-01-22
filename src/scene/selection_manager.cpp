#include "selection_manager.h"

namespace spectra {

void SelectionManager::setSelectedInstanceId(uint32_t instanceId) {
    if (m_selectedInstanceId != instanceId) {
        uint32_t oldId = m_selectedInstanceId;
        m_selectedInstanceId = instanceId;

        if (m_onSelectionChanged) {
            m_onSelectionChanged(oldId, instanceId);
        }
    }
}

} // namespace spectra
