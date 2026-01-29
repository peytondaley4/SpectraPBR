#pragma once

#include "widget.h"
#include "slider.h"
#include <functional>
#include <memory>

namespace spectra {
namespace ui {

//------------------------------------------------------------------------------
// ColorPicker - RGB color picker with 3 sliders
//------------------------------------------------------------------------------
class ColorPicker : public Widget {
public:
    ColorPicker();
    ~ColorPicker() override = default;

    // Color value (RGB, 0-1 range)
    void setColor(float3 color);
    float3 getColor() const { return m_color; }

    // Individual channels
    void setRed(float r);
    void setGreen(float g);
    void setBlue(float b);
    float getRed() const { return m_color.x; }
    float getGreen() const { return m_color.y; }
    float getBlue() const { return m_color.z; }

    // Show color preview swatch
    void setShowPreview(bool show) { m_showPreview = show; markDirty(); }
    bool getShowPreview() const { return m_showPreview; }

    // Intensity multiplier for HDR colors (for emission/irradiance)
    void setIntensityRange(float max);
    float getIntensityMax() const { return m_intensityMax; }

    // Color change callback
    using ColorChangedCallback = std::function<void(ColorPicker*, float3)>;
    void setOnColorChanged(ColorChangedCallback callback) { m_onColorChanged = callback; }

    // Layout constants
    static constexpr float SLIDER_HEIGHT = 24.0f;
    static constexpr float SLIDER_SPACING = 2.0f;
    static constexpr float PREVIEW_SIZE = 40.0f;

    // Override for collecting geometry from sliders
    void collectGeometry(std::vector<UIQuad>& outQuads, text::TextLayout* textLayout) override;

    // Input handling (public for composite widgets like PropertyPanel)
    bool onMouseDown(float2 pos, int button) override;
    bool onMouseUp(float2 pos, int button) override;
    bool onMouseMove(float2 pos) override;

protected:
    void generateGeometry(std::vector<UIQuad>& outQuads, text::TextLayout* textLayout) override;

private:
    void createSliders();
    void updateSlidersFromColor();
    void onSliderChanged(int channel, float value);

    float3 m_color = make_float3(1.0f, 1.0f, 1.0f);
    float m_intensityMax = 1.0f;
    bool m_showPreview = true;

    std::unique_ptr<Slider> m_redSlider;
    std::unique_ptr<Slider> m_greenSlider;
    std::unique_ptr<Slider> m_blueSlider;

    ColorChangedCallback m_onColorChanged;
};

} // namespace ui
} // namespace spectra
