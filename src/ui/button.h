#pragma once

#include "widget.h"
#include <functional>

namespace spectra {
namespace ui {

//------------------------------------------------------------------------------
// Button - Clickable button with text label
//------------------------------------------------------------------------------
class Button : public Widget {
public:
    Button();
    explicit Button(const std::string& label);
    ~Button() override = default;

    // Label
    void setLabel(const std::string& label) { m_label = label; markDirty(); }
    const std::string& getLabel() const { return m_label; }

    // Text scale
    void setTextScale(float scale) { m_textScale = scale; markDirty(); }
    float getTextScale() const { return m_textScale; }

    // Click callback
    using ClickCallback = std::function<void()>;
    void setOnClick(ClickCallback callback) { m_onClick = callback; }

    // Toggle mode (button stays active after click)
    void setToggleMode(bool toggle) { m_toggleMode = toggle; }
    bool isToggleMode() const { return m_toggleMode; }

    void setToggled(bool toggled) { m_toggled = toggled; markDirty(); }
    bool isToggled() const { return m_toggled; }

protected:
    void generateGeometry(std::vector<UIQuad>& outQuads, text::TextLayout* textLayout) override;
    bool onMouseDown(float2 pos, int button) override;
    bool onMouseUp(float2 pos, int button) override;

private:
    std::string m_label;
    float m_textScale = 0.6f;
    ClickCallback m_onClick;
    bool m_toggleMode = false;
    bool m_toggled = false;
};

} // namespace ui
} // namespace spectra
