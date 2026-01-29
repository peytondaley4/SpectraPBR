#include "slider.h"
#include "text/text_layout.h"
#include <algorithm>
#include <cstdio>

namespace spectra {
namespace ui {

Slider::Slider() {
    setSize(200.0f, 24.0f);
}

void Slider::setRange(float min, float max) {
    m_min = min;
    m_max = max;
    // Clamp current value to new range
    m_value = std::clamp(m_value, m_min, m_max);
    markDirty();
}

void Slider::setValue(float value) {
    float newValue = std::clamp(value, m_min, m_max);
    if (newValue != m_value) {
        m_value = newValue;
        markDirty();
        if (m_onValueChanged) {
            m_onValueChanged(this, m_value);
        }
    }
}

float Slider::getNormalizedValue() const {
    if (m_max <= m_min) return 0.0f;
    return (m_value - m_min) / (m_max - m_min);
}

void Slider::setNormalizedValue(float normalized) {
    setValue(m_min + normalized * (m_max - m_min));
}

Rect Slider::getTrackBounds() const {
    Rect bounds = getAbsoluteBounds();
    float trackX = bounds.x + m_labelWidth;
    float trackWidth = bounds.width - m_labelWidth - 50.0f;  // Leave space for value text
    float trackY = bounds.y + (bounds.height - TRACK_HEIGHT) * 0.5f;

    return { trackX, trackY, trackWidth, TRACK_HEIGHT };
}

Rect Slider::getThumbBounds() const {
    Rect track = getTrackBounds();
    float normalized = getNormalizedValue();
    float thumbX = track.x + normalized * (track.width - THUMB_WIDTH);
    float thumbY = track.y + (TRACK_HEIGHT - THUMB_HEIGHT) * 0.5f;

    return { thumbX, thumbY, THUMB_WIDTH, THUMB_HEIGHT };
}

float Slider::valueFromMouseX(float mouseX) const {
    Rect track = getTrackBounds();
    float normalized = (mouseX - track.x - THUMB_WIDTH * 0.5f) / (track.width - THUMB_WIDTH);
    normalized = std::clamp(normalized, 0.0f, 1.0f);
    return m_min + normalized * (m_max - m_min);
}

void Slider::generateGeometry(std::vector<UIQuad>& outQuads, text::TextLayout* textLayout) {
    const Theme* theme = getTheme();
    Rect bounds = getAbsoluteBounds();
    float depth = getEffectiveDepth();

    // Draw label
    if (!m_label.empty() && textLayout) {
        float4 labelColor = theme->propertyLabel;
        float labelY = bounds.y + bounds.height * 0.5f - textLayout->getLineHeight(0.5f) * 0.5f;
        textLayout->layout(m_label, make_float2(bounds.x + 4.0f, labelY), 0.5f, labelColor,
                          TextAlign::Left, depth + 0.001f, outQuads);
    }

    // Draw track background
    Rect track = getTrackBounds();
    outQuads.push_back(makeSolidQuad(track, theme->sliderTrack, depth));

    // Draw filled portion of track
    float normalized = getNormalizedValue();
    if (normalized > 0.0f) {
        Rect filled = { track.x, track.y, track.width * normalized, track.height };
        outQuads.push_back(makeSolidQuad(filled, theme->sliderTrackFilled, depth + 0.001f));
    }

    // Draw thumb
    Rect thumb = getThumbBounds();
    float4 thumbColor;
    if (m_dragging) {
        thumbColor = theme->sliderThumbActive;
    } else if (m_thumbHovered) {
        thumbColor = theme->sliderThumbHover;
    } else {
        thumbColor = theme->sliderThumb;
    }
    outQuads.push_back(makeSolidQuad(thumb, thumbColor, depth + 0.002f));

    // Draw value text
    if (m_showValue && textLayout) {
        char valueStr[32];
        snprintf(valueStr, sizeof(valueStr), m_valueFormat.c_str(), m_value);

        float4 valueColor = theme->propertyValue;
        float valueX = track.x + track.width + 8.0f;
        float valueY = bounds.y + bounds.height * 0.5f - textLayout->getLineHeight(0.5f) * 0.5f;
        textLayout->layout(valueStr, make_float2(valueX, valueY), 0.5f, valueColor,
                          TextAlign::Left, depth + 0.001f, outQuads);
    }
}

bool Slider::onMouseDown(float2 pos, int button) {
    if (!m_visible || !m_enabled) return false;
    if (button != 0) return false;

    Rect thumb = getThumbBounds();
    if (thumb.contains(pos)) {
        m_dragging = true;
        m_dragButton = button;
        m_active = true;
        setValue(valueFromMouseX(pos.x));
        return true;
    }

    return false;
}

bool Slider::onMouseUp(float2 pos, int button) {
    if (!m_visible || !m_enabled) return false;

    if (m_dragging && button == m_dragButton) {
        m_dragging = false;
        m_dragButton = -1;
        m_active = false;
        markDirty();
        return true;
    }

    return false;
}

bool Slider::onMouseMove(float2 pos) {
    if (!m_visible || !m_enabled) return false;

    // Update hover state for thumb
    Rect thumb = getThumbBounds();
    bool wasHovered = m_thumbHovered;
    m_thumbHovered = thumb.contains(pos);
    if (wasHovered != m_thumbHovered) {
        markDirty();
    }

    // Handle dragging
    if (m_dragging) {
        setValue(valueFromMouseX(pos.x));
        return true;
    }

    return m_thumbHovered;
}

} // namespace ui
} // namespace spectra
