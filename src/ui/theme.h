#pragma once

#include <cuda_runtime.h>

namespace spectra {
namespace ui {

//------------------------------------------------------------------------------
// Theme - Color palette for UI elements
//------------------------------------------------------------------------------
struct Theme {
    // Panel colors
    float4 panelBackground;
    float4 panelBackgroundAlt;   // Alternate background (for headers, etc.)
    float4 panelBorder;

    // Button colors
    float4 buttonNormal;
    float4 buttonHover;
    float4 buttonActive;
    float4 buttonDisabled;

    // Text colors
    float4 textPrimary;
    float4 textSecondary;
    float4 textDisabled;

    // Tree/List colors
    float4 treeSelected;
    float4 treeHover;
    float4 treeExpander;

    // Highlight/accent colors
    float4 highlight;
    float4 highlightText;

    // Scrollbar colors
    float4 scrollbarTrack;
    float4 scrollbarThumb;
    float4 scrollbarThumbHover;
    float4 scrollbarThumbActive;

    // Separator
    float4 separator;

    // Slider colors
    float4 sliderTrack;
    float4 sliderTrackFilled;
    float4 sliderThumb;
    float4 sliderThumbHover;
    float4 sliderThumbActive;

    // Property panel colors
    float4 propertyHeader;
    float4 propertyValue;
    float4 propertySeparator;
    float4 propertyLabel;
};

//------------------------------------------------------------------------------
// Predefined Themes
//------------------------------------------------------------------------------

// Color palette — only the values that differ between themes
struct ThemePalette {
    float panelBg;          // Base panel background brightness
    float panelBgAlt;       // Alternate background brightness
    float panelBorder;      // Border brightness
    float buttonNormal;     // Button normal brightness
    float buttonHover;      // Button hover brightness
    float buttonActive;     // Button active brightness
    float buttonDisabled;   // Button disabled brightness
    float textPrimary;      // Primary text brightness
    float textSecondary;    // Secondary text brightness
    float textDisabled;     // Disabled text brightness
    float treeHover;        // Tree hover brightness
    float treeExpander;     // Tree expander brightness
    float scrollTrack;      // Scrollbar track brightness
    float scrollThumb;      // Scrollbar thumb brightness
    float scrollThumbHover; // Scrollbar thumb hover brightness
    float scrollThumbActive;// Scrollbar thumb active brightness
    float separator;        // Separator brightness
    float sliderTrack;      // Slider track brightness
    float sliderThumb;      // Slider thumb brightness
    float sliderThumbHover; // Slider thumb hover brightness
    float propHeader;       // Property header brightness
    float propValue;        // Property value brightness
    float propSeparator;    // Property separator brightness
    float propLabel;        // Property label brightness
    float treeSelectedA;    // Tree selected alpha
};

inline float4 gray(float v, float a = 1.0f) { return make_float4(v, v, v, a); }

inline Theme createThemeFromPalette(const ThemePalette& p) {
    Theme t;
    t.panelBackground    = gray(p.panelBg);
    t.panelBackgroundAlt = gray(p.panelBgAlt);
    t.panelBorder        = gray(p.panelBorder);

    t.buttonNormal   = gray(p.buttonNormal);
    t.buttonHover    = gray(p.buttonHover);
    t.buttonActive   = gray(p.buttonActive);
    t.buttonDisabled = gray(p.buttonDisabled, 0.5f);

    t.textPrimary   = gray(p.textPrimary);
    t.textSecondary = gray(p.textSecondary);
    t.textDisabled  = gray(p.textDisabled);

    t.treeSelected = make_float4(0.20f, 0.40f, 0.70f, p.treeSelectedA);
    t.treeHover    = gray(p.treeHover);
    t.treeExpander = gray(p.treeExpander);

    // Shared accent color
    t.highlight     = make_float4(0.26f, 0.59f, 0.98f, 1.0f);
    t.highlightText = make_float4(1.0f, 1.0f, 1.0f, 1.0f);

    t.scrollbarTrack       = gray(p.scrollTrack, 0.5f);
    t.scrollbarThumb       = gray(p.scrollThumb, 0.8f);
    t.scrollbarThumbHover  = gray(p.scrollThumbHover, 0.9f);
    t.scrollbarThumbActive = gray(p.scrollThumbActive, 1.0f);

    t.separator = gray(p.separator);

    t.sliderTrack       = gray(p.sliderTrack);
    t.sliderTrackFilled = make_float4(0.26f, 0.59f, 0.98f, 1.0f);
    t.sliderThumb       = gray(p.sliderThumb);
    t.sliderThumbHover  = gray(p.sliderThumbHover);
    t.sliderThumbActive = make_float4(0.26f, 0.59f, 0.98f, 1.0f);

    t.propertyHeader    = gray(p.propHeader);
    t.propertyValue     = gray(p.propValue);
    t.propertySeparator = gray(p.propSeparator);
    t.propertyLabel     = gray(p.propLabel);

    return t;
}

inline Theme createDarkTheme() {
    ThemePalette p;
    p.panelBg = 0.15f;  p.panelBgAlt = 0.12f;  p.panelBorder = 0.25f;
    p.buttonNormal = 0.25f;  p.buttonHover = 0.35f;  p.buttonActive = 0.20f;  p.buttonDisabled = 0.18f;
    p.textPrimary = 0.95f;  p.textSecondary = 0.70f;  p.textDisabled = 0.45f;
    p.treeHover = 0.30f;  p.treeExpander = 0.60f;  p.treeSelectedA = 1.0f;
    p.scrollTrack = 0.10f;  p.scrollThumb = 0.40f;  p.scrollThumbHover = 0.50f;  p.scrollThumbActive = 0.60f;
    p.separator = 0.30f;
    p.sliderTrack = 0.20f;  p.sliderThumb = 0.50f;  p.sliderThumbHover = 0.60f;
    p.propHeader = 0.18f;  p.propValue = 0.85f;  p.propSeparator = 0.25f;  p.propLabel = 0.60f;
    return createThemeFromPalette(p);
}

inline Theme createLightTheme() {
    ThemePalette p;
    p.panelBg = 0.94f;  p.panelBgAlt = 0.88f;  p.panelBorder = 0.70f;
    p.buttonNormal = 0.85f;  p.buttonHover = 0.75f;  p.buttonActive = 0.90f;  p.buttonDisabled = 0.80f;
    p.textPrimary = 0.10f;  p.textSecondary = 0.35f;  p.textDisabled = 0.55f;
    p.treeHover = 0.80f;  p.treeExpander = 0.40f;  p.treeSelectedA = 0.8f;
    p.scrollTrack = 0.85f;  p.scrollThumb = 0.60f;  p.scrollThumbHover = 0.50f;  p.scrollThumbActive = 0.40f;
    p.separator = 0.70f;
    p.sliderTrack = 0.75f;  p.sliderThumb = 0.50f;  p.sliderThumbHover = 0.40f;
    p.propHeader = 0.85f;  p.propValue = 0.15f;  p.propSeparator = 0.75f;  p.propLabel = 0.40f;
    return createThemeFromPalette(p);
}

// Global theme instances
inline const Theme THEME_DARK = createDarkTheme();
inline const Theme THEME_LIGHT = createLightTheme();

} // namespace ui
} // namespace spectra
