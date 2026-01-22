#pragma once

#include <cstdint>
#include <functional>

namespace spectra {

//------------------------------------------------------------------------------
// Selection Manager - Tracks currently selected scene objects
//------------------------------------------------------------------------------
class SelectionManager {
public:
    SelectionManager() = default;
    ~SelectionManager() = default;

    // Set the selected instance ID (UINT32_MAX = no selection)
    void setSelectedInstanceId(uint32_t instanceId);

    // Get the selected instance ID
    uint32_t getSelectedInstanceId() const { return m_selectedInstanceId; }

    // Check if anything is selected
    bool hasSelection() const { return m_selectedInstanceId != UINT32_MAX; }

    // Clear selection
    void clearSelection() { setSelectedInstanceId(UINT32_MAX); }

    // Callback for selection changes
    using SelectionCallback = std::function<void(uint32_t oldId, uint32_t newId)>;
    void setOnSelectionChanged(SelectionCallback callback) { m_onSelectionChanged = callback; }

private:
    uint32_t m_selectedInstanceId = UINT32_MAX;  // No selection
    SelectionCallback m_onSelectionChanged;
};

} // namespace spectra
