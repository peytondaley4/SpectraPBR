#pragma once

#include "widget.h"
#include <functional>
#include <string>

namespace spectra {
namespace ui {

//------------------------------------------------------------------------------
// Slider - Draggable slider for numeric value editing
//------------------------------------------------------------------------------
class Slider : public Widget {
public:
    Slider();
    ~Slider() override = default;

    // Value range
    void setRange(float min, float max);
    float getMin() const { return m_min; }
    float getMax() const { return m_max; }

    // Current value
    void setValue(float value);
    float getValue() const { return m_value; }

    // Get normalized value (0-1)
    float getNormalizedValue() const;

    // Set value from normalized (0-1)
    void setNormalizedValue(float normalized);

    // Label (displayed to the left)
    void setLabel(const std::string& label) { m_label = label; markDirty(); }
    const std::string& getLabel() const { return m_label; }

    // Show value text
    void setShowValue(bool show) { m_showValue = show; markDirty(); }
    bool getShowValue() const { return m_showValue; }

    // Value format (printf style, e.g., "%.2f")
    void setValueFormat(const std::string& format) { m_valueFormat = format; markDirty(); }
    const std::string& getValueFormat() const { return m_valueFormat; }

    // Label width (space for label)
    void setLabelWidth(float width) { m_labelWidth = width; markDirty(); }
    float getLabelWidth() const { return m_labelWidth; }

    // Value change callback
    using ValueChangedCallback = std::function<void(Slider*, float)>;
    void setOnValueChanged(ValueChangedCallback callback) { m_onValueChanged = callback; }

    // Layout constants
    static constexpr float TRACK_HEIGHT = 4.0f;
    static constexpr float THUMB_WIDTH = 12.0f;
    static constexpr float THUMB_HEIGHT = 18.0f;

    // Input handling (public for composite widgets like ColorPicker)
    bool onMouseDown(float2 pos, int button) override;
    bool onMouseUp(float2 pos, int button) override;
    bool onMouseMove(float2 pos) override;

protected:
    void generateGeometry(std::vector<UIQuad>& outQuads, text::TextLayout* textLayout) override;

private:
    Rect getTrackBounds() const;
    Rect getThumbBounds() const;
    float valueFromMouseX(float mouseX) const;

    float m_min = 0.0f;
    float m_max = 1.0f;
    float m_value = 0.5f;

    std::string m_label;
    std::string m_valueFormat = "%.2f";
    float m_labelWidth = 60.0f;
    bool m_showValue = true;

    bool m_dragging = false;
    int m_dragButton = -1;   // Button that started the drag
    bool m_thumbHovered = false;

    // Cached value string to avoid per-frame allocations
    mutable char m_cachedValueStr[32] = {};
    mutable float m_cachedValue = -999999.0f;  // Sentinel value

    ValueChangedCallback m_onValueChanged;
};

} // namespace ui
} // namespace spectra
