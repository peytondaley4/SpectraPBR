#include "color_picker.h"
#include "text/text_layout.h"

namespace spectra {
namespace ui {

ColorPicker::ColorPicker() {
    setSize(200.0f, SLIDER_HEIGHT * 3 + SLIDER_SPACING * 2 + 8.0f);
    createSliders();
}

void ColorPicker::createSliders() {
    // Red slider
    m_redSlider = std::make_unique<Slider>();
    m_redSlider->setLabel("R");
    m_redSlider->setLabelWidth(20.0f);
    m_redSlider->setRange(0.0f, m_intensityMax);
    m_redSlider->setValue(m_color.x);
    m_redSlider->setOnValueChanged([this](Slider*, float v) { onSliderChanged(0, v); });

    // Green slider
    m_greenSlider = std::make_unique<Slider>();
    m_greenSlider->setLabel("G");
    m_greenSlider->setLabelWidth(20.0f);
    m_greenSlider->setRange(0.0f, m_intensityMax);
    m_greenSlider->setValue(m_color.y);
    m_greenSlider->setOnValueChanged([this](Slider*, float v) { onSliderChanged(1, v); });

    // Blue slider
    m_blueSlider = std::make_unique<Slider>();
    m_blueSlider->setLabel("B");
    m_blueSlider->setLabelWidth(20.0f);
    m_blueSlider->setRange(0.0f, m_intensityMax);
    m_blueSlider->setValue(m_color.z);
    m_blueSlider->setOnValueChanged([this](Slider*, float v) { onSliderChanged(2, v); });
}

void ColorPicker::setColor(float3 color) {
    if (m_color.x == color.x && m_color.y == color.y && m_color.z == color.z) return;
    m_color = color;
    updateSlidersFromColor();
    markDirty();
}

void ColorPicker::setRed(float r) {
    m_color.x = r;
    if (m_redSlider) m_redSlider->setValue(r);
    markDirty();
}

void ColorPicker::setGreen(float g) {
    m_color.y = g;
    if (m_greenSlider) m_greenSlider->setValue(g);
    markDirty();
}

void ColorPicker::setBlue(float b) {
    m_color.z = b;
    if (m_blueSlider) m_blueSlider->setValue(b);
    markDirty();
}

void ColorPicker::setIntensityRange(float max) {
    m_intensityMax = max;
    if (m_redSlider) m_redSlider->setRange(0.0f, max);
    if (m_greenSlider) m_greenSlider->setRange(0.0f, max);
    if (m_blueSlider) m_blueSlider->setRange(0.0f, max);
    markDirty();
}

void ColorPicker::updateSlidersFromColor() {
    if (m_redSlider) m_redSlider->setValue(m_color.x);
    if (m_greenSlider) m_greenSlider->setValue(m_color.y);
    if (m_blueSlider) m_blueSlider->setValue(m_color.z);
}

void ColorPicker::onSliderChanged(int channel, float value) {
    switch (channel) {
        case 0: m_color.x = value; break;
        case 1: m_color.y = value; break;
        case 2: m_color.z = value; break;
    }
    markDirty();

    if (m_onColorChanged) {
        m_onColorChanged(this, m_color);
    }
}

void ColorPicker::collectGeometry(std::vector<UIQuad>& outQuads, text::TextLayout* textLayout) {
    // First generate our own geometry
    generateGeometry(outQuads, textLayout);

    // Check if layout needs updating (bounds changed)
    Rect bounds = getAbsoluteBounds();
    bool boundsChanged = (bounds.x != m_lastBounds.x || bounds.y != m_lastBounds.y ||
                          bounds.width != m_lastBounds.width || bounds.height != m_lastBounds.height);

    // Only update slider positions/sizes when bounds change
    if (boundsChanged || !m_layoutCached) {
        float previewOffset = m_showPreview ? PREVIEW_SIZE + 8.0f : 0.0f;
        float sliderWidth = bounds.width - previewOffset - 8.0f;

        if (m_redSlider) {
            m_redSlider->setPosition(bounds.x + previewOffset + 4.0f, bounds.y + 4.0f);
            m_redSlider->setSize(sliderWidth, SLIDER_HEIGHT);
        }

        if (m_greenSlider) {
            m_greenSlider->setPosition(bounds.x + previewOffset + 4.0f,
                                       bounds.y + 4.0f + SLIDER_HEIGHT + SLIDER_SPACING);
            m_greenSlider->setSize(sliderWidth, SLIDER_HEIGHT);
        }

        if (m_blueSlider) {
            m_blueSlider->setPosition(bounds.x + previewOffset + 4.0f,
                                      bounds.y + 4.0f + (SLIDER_HEIGHT + SLIDER_SPACING) * 2);
            m_blueSlider->setSize(sliderWidth, SLIDER_HEIGHT);
        }

        m_lastBounds = bounds;
        m_layoutCached = true;
    }

    // Set theme only if it changed (Widget::setTheme now has early exit)
    const Theme* theme = getTheme();
    if (m_redSlider) {
        m_redSlider->setTheme(theme);
        m_redSlider->collectGeometry(outQuads, textLayout);
    }
    if (m_greenSlider) {
        m_greenSlider->setTheme(theme);
        m_greenSlider->collectGeometry(outQuads, textLayout);
    }
    if (m_blueSlider) {
        m_blueSlider->setTheme(theme);
        m_blueSlider->collectGeometry(outQuads, textLayout);
    }
}

void ColorPicker::generateGeometry(std::vector<UIQuad>& outQuads, text::TextLayout* textLayout) {
    if (!m_showPreview) return;

    const Theme* theme = getTheme();
    Rect bounds = getAbsoluteBounds();
    float depth = getEffectiveDepth();

    // Draw color preview swatch
    float previewX = bounds.x + 4.0f;
    float previewY = bounds.y + (bounds.height - PREVIEW_SIZE) * 0.5f;

    // Normalize color for display (cap at 1.0 for preview)
    float displayR = m_color.x > 1.0f ? 1.0f : m_color.x;
    float displayG = m_color.y > 1.0f ? 1.0f : m_color.y;
    float displayB = m_color.z > 1.0f ? 1.0f : m_color.z;

    Rect previewRect = { previewX, previewY, PREVIEW_SIZE, PREVIEW_SIZE };
    float4 previewColor = make_float4(displayR, displayG, displayB, 1.0f);

    // Border around preview
    Rect borderRect = { previewX - 1.0f, previewY - 1.0f, PREVIEW_SIZE + 2.0f, PREVIEW_SIZE + 2.0f };
    outQuads.push_back(makeSolidQuad(borderRect, theme->panelBorder, depth));
    outQuads.push_back(makeSolidQuad(previewRect, previewColor, depth + 0.001f));
}

bool ColorPicker::onMouseDown(float2 pos, int button) {
    if (!m_visible || !m_enabled) return false;

    // Forward to sliders
    if (m_redSlider && m_redSlider->onMouseDown(pos, button)) return true;
    if (m_greenSlider && m_greenSlider->onMouseDown(pos, button)) return true;
    if (m_blueSlider && m_blueSlider->onMouseDown(pos, button)) return true;

    return containsPoint(pos);
}

bool ColorPicker::onMouseUp(float2 pos, int button) {
    if (!m_visible || !m_enabled) return false;

    bool consumed = false;
    if (m_redSlider) consumed = m_redSlider->onMouseUp(pos, button) || consumed;
    if (m_greenSlider) consumed = m_greenSlider->onMouseUp(pos, button) || consumed;
    if (m_blueSlider) consumed = m_blueSlider->onMouseUp(pos, button) || consumed;

    return consumed || containsPoint(pos);
}

bool ColorPicker::onMouseMove(float2 pos) {
    if (!m_visible || !m_enabled) return false;

    bool consumed = false;
    if (m_redSlider) consumed = m_redSlider->onMouseMove(pos) || consumed;
    if (m_greenSlider) consumed = m_greenSlider->onMouseMove(pos) || consumed;
    if (m_blueSlider) consumed = m_blueSlider->onMouseMove(pos) || consumed;

    return consumed;
}

} // namespace ui
} // namespace spectra
