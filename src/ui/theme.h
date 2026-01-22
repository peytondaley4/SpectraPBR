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
};

//------------------------------------------------------------------------------
// Predefined Themes
//------------------------------------------------------------------------------

// Dark theme (default)
inline Theme createDarkTheme() {
    Theme theme;

    // Panel colors - dark gray backgrounds
    theme.panelBackground    = make_float4(0.15f, 0.15f, 0.15f, 0.95f);
    theme.panelBackgroundAlt = make_float4(0.12f, 0.12f, 0.12f, 0.95f);
    theme.panelBorder        = make_float4(0.25f, 0.25f, 0.25f, 1.0f);

    // Button colors
    theme.buttonNormal   = make_float4(0.25f, 0.25f, 0.25f, 1.0f);
    theme.buttonHover    = make_float4(0.35f, 0.35f, 0.35f, 1.0f);
    theme.buttonActive   = make_float4(0.20f, 0.20f, 0.20f, 1.0f);
    theme.buttonDisabled = make_float4(0.18f, 0.18f, 0.18f, 0.5f);

    // Text colors
    theme.textPrimary   = make_float4(0.95f, 0.95f, 0.95f, 1.0f);
    theme.textSecondary = make_float4(0.70f, 0.70f, 0.70f, 1.0f);
    theme.textDisabled  = make_float4(0.45f, 0.45f, 0.45f, 1.0f);

    // Tree/List colors
    theme.treeSelected = make_float4(0.20f, 0.40f, 0.70f, 1.0f);
    theme.treeHover    = make_float4(0.30f, 0.30f, 0.30f, 1.0f);
    theme.treeExpander = make_float4(0.60f, 0.60f, 0.60f, 1.0f);

    // Highlight/accent - blue accent
    theme.highlight     = make_float4(0.26f, 0.59f, 0.98f, 1.0f);
    theme.highlightText = make_float4(1.0f, 1.0f, 1.0f, 1.0f);

    // Scrollbar
    theme.scrollbarTrack       = make_float4(0.10f, 0.10f, 0.10f, 0.5f);
    theme.scrollbarThumb       = make_float4(0.40f, 0.40f, 0.40f, 0.8f);
    theme.scrollbarThumbHover  = make_float4(0.50f, 0.50f, 0.50f, 0.9f);
    theme.scrollbarThumbActive = make_float4(0.60f, 0.60f, 0.60f, 1.0f);

    // Separator
    theme.separator = make_float4(0.30f, 0.30f, 0.30f, 1.0f);

    return theme;
}

// Light theme
inline Theme createLightTheme() {
    Theme theme;

    // Panel colors - light gray backgrounds
    theme.panelBackground    = make_float4(0.94f, 0.94f, 0.94f, 0.95f);
    theme.panelBackgroundAlt = make_float4(0.88f, 0.88f, 0.88f, 0.95f);
    theme.panelBorder        = make_float4(0.70f, 0.70f, 0.70f, 1.0f);

    // Button colors
    theme.buttonNormal   = make_float4(0.85f, 0.85f, 0.85f, 1.0f);
    theme.buttonHover    = make_float4(0.75f, 0.75f, 0.75f, 1.0f);
    theme.buttonActive   = make_float4(0.90f, 0.90f, 0.90f, 1.0f);
    theme.buttonDisabled = make_float4(0.80f, 0.80f, 0.80f, 0.5f);

    // Text colors
    theme.textPrimary   = make_float4(0.10f, 0.10f, 0.10f, 1.0f);
    theme.textSecondary = make_float4(0.35f, 0.35f, 0.35f, 1.0f);
    theme.textDisabled  = make_float4(0.55f, 0.55f, 0.55f, 1.0f);

    // Tree/List colors
    theme.treeSelected = make_float4(0.26f, 0.59f, 0.98f, 0.8f);
    theme.treeHover    = make_float4(0.80f, 0.80f, 0.80f, 1.0f);
    theme.treeExpander = make_float4(0.40f, 0.40f, 0.40f, 1.0f);

    // Highlight/accent - blue accent
    theme.highlight     = make_float4(0.26f, 0.59f, 0.98f, 1.0f);
    theme.highlightText = make_float4(1.0f, 1.0f, 1.0f, 1.0f);

    // Scrollbar
    theme.scrollbarTrack       = make_float4(0.85f, 0.85f, 0.85f, 0.5f);
    theme.scrollbarThumb       = make_float4(0.60f, 0.60f, 0.60f, 0.8f);
    theme.scrollbarThumbHover  = make_float4(0.50f, 0.50f, 0.50f, 0.9f);
    theme.scrollbarThumbActive = make_float4(0.40f, 0.40f, 0.40f, 1.0f);

    // Separator
    theme.separator = make_float4(0.70f, 0.70f, 0.70f, 1.0f);

    return theme;
}

// Global theme instances
inline const Theme THEME_DARK = createDarkTheme();
inline const Theme THEME_LIGHT = createLightTheme();

} // namespace ui
} // namespace spectra
