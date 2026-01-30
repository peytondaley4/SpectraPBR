#include "property_panel.h"
#include "text/text_layout.h"
#include <cstdio>
#include <algorithm>
#include <cmath>

namespace spectra {
namespace ui {

PropertyPanel::PropertyPanel() {
    setSize(300.0f, 500.0f);
    createWidgets();
}

void PropertyPanel::createWidgets() {
    // Create the main panel (handles background/border/dragging)
    m_panel = std::make_unique<Panel>();
    m_panel->setShowHeader(true);
    m_panel->setTitle("Properties");
    m_panel->setShowBorder(true);
    m_panel->setCloseable(false);
    m_panel->setDraggable(true);

    // Position sliders
    m_posXSlider = std::make_unique<Slider>();
    m_posXSlider->setLabel("X");
    m_posXSlider->setLabelWidth(24.0f);
    m_posXSlider->setRange(-50.0f, 50.0f);
    m_posXSlider->setValueFormat("%.1f");
    m_posXSlider->setParent(this);
    m_posXSlider->setOnValueChanged([this](Slider*, float) { updateLightFromSliders(); });

    m_posYSlider = std::make_unique<Slider>();
    m_posYSlider->setLabel("Y");
    m_posYSlider->setLabelWidth(24.0f);
    m_posYSlider->setRange(-50.0f, 50.0f);
    m_posYSlider->setValueFormat("%.1f");
    m_posYSlider->setParent(this);
    m_posYSlider->setOnValueChanged([this](Slider*, float) { updateLightFromSliders(); });

    m_posZSlider = std::make_unique<Slider>();
    m_posZSlider->setLabel("Z");
    m_posZSlider->setLabelWidth(24.0f);
    m_posZSlider->setRange(-50.0f, 50.0f);
    m_posZSlider->setValueFormat("%.1f");
    m_posZSlider->setParent(this);
    m_posZSlider->setOnValueChanged([this](Slider*, float) { updateLightFromSliders(); });

    // Color picker
    m_colorPicker = std::make_unique<ColorPicker>();
    m_colorPicker->setParent(this);
    m_colorPicker->setOnColorChanged([this](ColorPicker*, float3) { updateLightFromSliders(); });

    // Intensity slider
    m_intensitySlider = std::make_unique<Slider>();
    m_intensitySlider->setLabel("Intensity");
    m_intensitySlider->setLabelWidth(70.0f);
    m_intensitySlider->setRange(0.0f, 500.0f);
    m_intensitySlider->setValueFormat("%.0f");
    m_intensitySlider->setParent(this);
    m_intensitySlider->setOnValueChanged([this](Slider*, float) { updateLightFromSliders(); });

    // Radius slider (point lights)
    m_radiusSlider = std::make_unique<Slider>();
    m_radiusSlider->setLabel("Radius");
    m_radiusSlider->setLabelWidth(70.0f);
    m_radiusSlider->setRange(0.0f, 5.0f);
    m_radiusSlider->setValueFormat("%.2f");
    m_radiusSlider->setParent(this);
    m_radiusSlider->setOnValueChanged([this](Slider*, float) { updateLightFromSliders(); });

    // Angular diameter slider (directional lights)
    m_angularDiameterSlider = std::make_unique<Slider>();
    m_angularDiameterSlider->setLabel("Angular");
    m_angularDiameterSlider->setLabelWidth(70.0f);
    m_angularDiameterSlider->setRange(0.0f, 1.0f);
    m_angularDiameterSlider->setValueFormat("%.2f");
    m_angularDiameterSlider->setParent(this);
    m_angularDiameterSlider->setOnValueChanged([this](Slider*, float) { updateLightFromSliders(); });

    // Size sliders (area lights)
    m_sizeXSlider = std::make_unique<Slider>();
    m_sizeXSlider->setLabel("Width");
    m_sizeXSlider->setLabelWidth(60.0f);
    m_sizeXSlider->setRange(0.1f, 10.0f);
    m_sizeXSlider->setValueFormat("%.1f");
    m_sizeXSlider->setParent(this);
    m_sizeXSlider->setOnValueChanged([this](Slider*, float) { updateLightFromSliders(); });

    m_sizeYSlider = std::make_unique<Slider>();
    m_sizeYSlider->setLabel("Height");
    m_sizeYSlider->setLabelWidth(60.0f);
    m_sizeYSlider->setRange(0.1f, 10.0f);
    m_sizeYSlider->setValueFormat("%.1f");
    m_sizeYSlider->setParent(this);
    m_sizeYSlider->setOnValueChanged([this](Slider*, float) { updateLightFromSliders(); });

    // Material editing widgets
    m_baseColorPicker = std::make_unique<ColorPicker>();
    m_baseColorPicker->setShowPreview(true);
    m_baseColorPicker->setParent(this);
    m_baseColorPicker->setOnColorChanged([this](ColorPicker*, float3) { updateMaterialFromSliders(); });

    m_metallicSlider = std::make_unique<Slider>();
    m_metallicSlider->setLabel("Metallic");
    m_metallicSlider->setLabelWidth(60.0f);
    m_metallicSlider->setRange(0.0f, 1.0f);
    m_metallicSlider->setValueFormat("%.2f");
    m_metallicSlider->setParent(this);
    m_metallicSlider->setOnValueChanged([this](Slider*, float) { updateMaterialFromSliders(); });

    m_roughnessSlider = std::make_unique<Slider>();
    m_roughnessSlider->setLabel("Roughness");
    m_roughnessSlider->setLabelWidth(70.0f);
    m_roughnessSlider->setRange(0.0f, 1.0f);
    m_roughnessSlider->setValueFormat("%.2f");
    m_roughnessSlider->setParent(this);
    m_roughnessSlider->setOnValueChanged([this](Slider*, float) { updateMaterialFromSliders(); });

    m_emissivePicker = std::make_unique<ColorPicker>();
    m_emissivePicker->setShowPreview(true);
    m_emissivePicker->setIntensityRange(10.0f);
    m_emissivePicker->setParent(this);
    m_emissivePicker->setOnColorChanged([this](ColorPicker*, float3) { updateMaterialFromSliders(); });

    m_emissiveIntensitySlider = std::make_unique<Slider>();
    m_emissiveIntensitySlider->setLabel("Em. Int.");
    m_emissiveIntensitySlider->setLabelWidth(60.0f);
    m_emissiveIntensitySlider->setRange(0.0f, 100.0f);
    m_emissiveIntensitySlider->setValueFormat("%.1f");
    m_emissiveIntensitySlider->setParent(this);
    m_emissiveIntensitySlider->setOnValueChanged([this](Slider*, float) { updateMaterialFromSliders(); });
}

void PropertyPanel::setDraggable(bool draggable) {
    if (m_panel) m_panel->setDraggable(draggable);
}

bool PropertyPanel::isDraggable() const {
    return m_panel ? m_panel->isDraggable() : false;
}

void PropertyPanel::showInstanceProperties(const InstanceInfo& info) {
    m_instanceInfo = info;
    m_displayMode = DisplayMode::Instance;

    // Update material widgets
    if (m_baseColorPicker) {
        m_baseColorPicker->setColor(make_float3(info.baseColor.x, info.baseColor.y, info.baseColor.z));
        m_baseColorPicker->setVisible(true);
    }
    if (m_metallicSlider) {
        m_metallicSlider->setValue(info.metallic);
        m_metallicSlider->setVisible(true);
    }
    if (m_roughnessSlider) {
        m_roughnessSlider->setValue(info.roughness);
        m_roughnessSlider->setVisible(true);
    }
    if (m_emissivePicker) {
        float intensity = std::max({info.emissive.x, info.emissive.y, info.emissive.z, 1.0f});
        m_emissivePicker->setColor(make_float3(info.emissive.x / intensity, info.emissive.y / intensity, info.emissive.z / intensity));
        m_emissivePicker->setVisible(true);
    }
    if (m_emissiveIntensitySlider) {
        m_emissiveIntensitySlider->setValue(std::max({info.emissive.x, info.emissive.y, info.emissive.z, 0.0f}));
        m_emissiveIntensitySlider->setVisible(true);
    }

    // Hide light widgets
    if (m_posXSlider) m_posXSlider->setVisible(false);
    if (m_posYSlider) m_posYSlider->setVisible(false);
    if (m_posZSlider) m_posZSlider->setVisible(false);
    if (m_colorPicker) m_colorPicker->setVisible(false);
    if (m_intensitySlider) m_intensitySlider->setVisible(false);
    if (m_radiusSlider) m_radiusSlider->setVisible(false);
    if (m_angularDiameterSlider) m_angularDiameterSlider->setVisible(false);
    if (m_sizeXSlider) m_sizeXSlider->setVisible(false);
    if (m_sizeYSlider) m_sizeYSlider->setVisible(false);

    markDirty();
}

void PropertyPanel::showLightProperties(const LightInfo& info) {
    m_lightInfo = info;
    m_displayMode = DisplayMode::Light;

    // Update light widgets
    if (m_posXSlider) { m_posXSlider->setValue(info.position.x); m_posXSlider->setVisible(true); }
    if (m_posYSlider) { m_posYSlider->setValue(info.position.y); m_posYSlider->setVisible(true); }
    if (m_posZSlider) { m_posZSlider->setValue(info.position.z); m_posZSlider->setVisible(true); }

    float maxComponent = std::max({info.color.x, info.color.y, info.color.z, 1.0f});
    if (m_colorPicker) {
        m_colorPicker->setColor(make_float3(info.color.x / maxComponent, info.color.y / maxComponent, info.color.z / maxComponent));
        m_colorPicker->setVisible(true);
    }
    if (m_intensitySlider) { m_intensitySlider->setValue(info.intensity); m_intensitySlider->setVisible(true); }

    // Type-specific
    if (m_radiusSlider) { m_radiusSlider->setValue(info.radius); m_radiusSlider->setVisible(info.type == SceneNodeType::PointLight); }
    if (m_angularDiameterSlider) { m_angularDiameterSlider->setValue(info.angularDiameter); m_angularDiameterSlider->setVisible(info.type == SceneNodeType::DirectionalLight); }
    if (m_sizeXSlider) { m_sizeXSlider->setValue(info.size.x); m_sizeXSlider->setVisible(info.type == SceneNodeType::AreaLight); }
    if (m_sizeYSlider) { m_sizeYSlider->setValue(info.size.y); m_sizeYSlider->setVisible(info.type == SceneNodeType::AreaLight); }

    // Hide material widgets
    if (m_baseColorPicker) m_baseColorPicker->setVisible(false);
    if (m_metallicSlider) m_metallicSlider->setVisible(false);
    if (m_roughnessSlider) m_roughnessSlider->setVisible(false);
    if (m_emissivePicker) m_emissivePicker->setVisible(false);
    if (m_emissiveIntensitySlider) m_emissiveIntensitySlider->setVisible(false);

    markDirty();
}

void PropertyPanel::clearProperties() {
    m_displayMode = DisplayMode::Empty;
    markDirty();
}

void PropertyPanel::updateLightFromSliders() {
    if (m_displayMode != DisplayMode::Light) return;

    m_lightInfo.position = make_float3(m_posXSlider->getValue(), m_posYSlider->getValue(), m_posZSlider->getValue());
    float3 normalizedColor = m_colorPicker->getColor();
    m_lightInfo.intensity = m_intensitySlider->getValue();
    m_lightInfo.color = make_float3(normalizedColor.x * m_lightInfo.intensity, normalizedColor.y * m_lightInfo.intensity, normalizedColor.z * m_lightInfo.intensity);

    if (m_lightInfo.type == SceneNodeType::PointLight) m_lightInfo.radius = m_radiusSlider->getValue();
    else if (m_lightInfo.type == SceneNodeType::DirectionalLight) m_lightInfo.angularDiameter = m_angularDiameterSlider->getValue();
    else if (m_lightInfo.type == SceneNodeType::AreaLight) m_lightInfo.size = make_float2(m_sizeXSlider->getValue(), m_sizeYSlider->getValue());

    if (m_onLightEdit) m_onLightEdit(m_lightInfo.type, m_lightInfo.index, m_lightInfo);
    markDirty();
}

void PropertyPanel::updateMaterialFromSliders() {
    if (m_displayMode != DisplayMode::Instance) return;

    if (m_baseColorPicker) {
        float3 c = m_baseColorPicker->getColor();
        m_instanceInfo.baseColor = make_float4(c.x, c.y, c.z, m_instanceInfo.baseColor.w);
    }
    if (m_metallicSlider) m_instanceInfo.metallic = m_metallicSlider->getValue();
    if (m_roughnessSlider) m_instanceInfo.roughness = m_roughnessSlider->getValue();
    if (m_emissivePicker && m_emissiveIntensitySlider) {
        float3 c = m_emissivePicker->getColor();
        float i = m_emissiveIntensitySlider->getValue();
        m_instanceInfo.emissive = make_float3(c.x * i, c.y * i, c.z * i);
    }

    if (m_onMaterialEdit) {
        GpuMaterial mat = {};
        mat.baseColor = m_instanceInfo.baseColor;
        mat.metallic = m_instanceInfo.metallic;
        mat.roughness = m_instanceInfo.roughness;
        mat.emissive = m_instanceInfo.emissive;
        mat.ior = 1.5f;
        mat.alphaCutoff = 0.5f;
        m_onMaterialEdit(m_instanceInfo.instanceId, mat);
    }
    markDirty();
}

void PropertyPanel::updatePanelAndTheme() {
    Rect bounds = getAbsoluteBounds();
    const Theme* theme = getTheme();

    bool boundsChanged = (bounds.x != m_lastBounds.x || bounds.y != m_lastBounds.y ||
                          bounds.width != m_lastBounds.width || bounds.height != m_lastBounds.height);
    bool themeChanged = (theme != m_lastTheme);

    if (boundsChanged && m_panel) {
        m_panel->setPosition(bounds.x, bounds.y);
        m_panel->setSize(bounds.width, bounds.height);
    }

    if (themeChanged) {
        if (m_panel) m_panel->setTheme(theme);
        // Set theme on all widgets
        Widget* widgets[] = {
            m_posXSlider.get(), m_posYSlider.get(), m_posZSlider.get(),
            m_colorPicker.get(), m_intensitySlider.get(), m_radiusSlider.get(),
            m_angularDiameterSlider.get(), m_sizeXSlider.get(), m_sizeYSlider.get(),
            m_baseColorPicker.get(), m_metallicSlider.get(), m_roughnessSlider.get(),
            m_emissivePicker.get(), m_emissiveIntensitySlider.get()
        };
        for (Widget* w : widgets) {
            if (w) w->setTheme(theme);
        }
    }

    m_lastBounds = bounds;
    m_lastTheme = theme;
}

//------------------------------------------------------------------------------
// addWidget - The core layout function
// Positions widget, sizes it, renders it, and advances contentY
//------------------------------------------------------------------------------
void PropertyPanel::addWidget(Widget* widget, float height, std::vector<UIQuad>& outQuads, text::TextLayout* textLayout) {
    if (!widget) {
        m_contentY += height;  // Advance even if null to maintain spacing
        return;
    }
    if (!widget->isVisible()) {
        return;  // Don't position or advance for hidden widgets
    }

    // Skip if entirely off-screen (culling)
    if (m_contentY + height < m_visibleMinY || m_contentY > m_visibleMaxY) {
        m_contentY += height;
        return;
    }

    // Position widget relative to PropertyPanel
    // Use setPositionDirect/setSizeDirect to avoid markDirty cascade during layout
    float widgetY = m_contentY - m_bounds.y;
    float2 newPos = make_float2(PADDING, widgetY);
    float2 newSize = make_float2(m_contentWidth, height);

    // Use direct setters to avoid dirty propagation during layout
    widget->setPositionDirect(newPos);
    widget->setSizeDirect(newSize);

    // Render widget
    widget->collectGeometry(outQuads, textLayout);

    // Advance contentY
    m_contentY += height;
}

void PropertyPanel::drawHeader(std::vector<UIQuad>& outQuads, text::TextLayout* textLayout,
                               const std::string& title, const Theme* theme, float depth) {
    // Skip if entirely off-screen (culling)
    if (m_contentY + HEADER_HEIGHT < m_visibleMinY || m_contentY > m_visibleMaxY) {
        m_contentY += HEADER_HEIGHT;
        return;
    }

    Rect headerRect = { m_bounds.x, m_contentY, m_bounds.width, HEADER_HEIGHT };
    outQuads.push_back(makeSolidQuad(headerRect, theme->propertyHeader, depth + 0.001f));

    if (textLayout) {
        float textY = m_contentY + (HEADER_HEIGHT - textLayout->getLineHeight(0.55f)) * 0.5f;
        textLayout->layout(title, make_float2(m_bounds.x + PADDING, textY), 0.55f,
                          theme->textPrimary, TextAlign::Left, depth + 0.002f, outQuads);
    }
    m_contentY += HEADER_HEIGHT;
}

void PropertyPanel::drawPropertyRow(std::vector<UIQuad>& outQuads, text::TextLayout* textLayout,
                                     const std::string& label, const std::string& value,
                                     const Theme* theme, float depth) {
    // Skip if entirely off-screen (culling)
    if (m_contentY + ROW_HEIGHT < m_visibleMinY || m_contentY > m_visibleMaxY) {
        m_contentY += ROW_HEIGHT;
        return;
    }

    if (textLayout) {
        float textY = m_contentY + (ROW_HEIGHT - textLayout->getLineHeight(0.5f)) * 0.5f;
        textLayout->layout(label, make_float2(m_bounds.x + PADDING, textY), 0.5f,
                          theme->propertyLabel, TextAlign::Left, depth + 0.001f, outQuads);
        textLayout->layout(value, make_float2(m_bounds.x + m_bounds.width * 0.45f, textY), 0.5f,
                          theme->propertyValue, TextAlign::Left, depth + 0.001f, outQuads);
    }
    m_contentY += ROW_HEIGHT;
}

void PropertyPanel::drawSeparator(std::vector<UIQuad>& outQuads, const Theme* theme, float depth) {
    Rect sepRect = { m_bounds.x + PADDING, m_contentY + 4.0f, m_bounds.width - PADDING * 2, 1.0f };
    outQuads.push_back(makeSolidQuad(sepRect, theme->propertySeparator, depth + 0.001f));
    m_contentY += SECTION_SPACING;
}

void PropertyPanel::drawLabel(std::vector<UIQuad>& outQuads, text::TextLayout* textLayout,
                              const std::string& text, const Theme* theme, float depth) {
    // Skip if entirely off-screen (culling)
    if (m_contentY + ROW_HEIGHT < m_visibleMinY || m_contentY > m_visibleMaxY) {
        m_contentY += ROW_HEIGHT;
        return;
    }

    if (textLayout) {
        float textY = m_contentY + (ROW_HEIGHT - textLayout->getLineHeight(0.5f)) * 0.5f;
        textLayout->layout(text, make_float2(m_bounds.x + PADDING, textY), 0.5f,
                          theme->propertyLabel, TextAlign::Left, depth + 0.001f, outQuads);
    }
    m_contentY += ROW_HEIGHT;
}

void PropertyPanel::drawTexturePreview(std::vector<UIQuad>& outQuads, text::TextLayout* textLayout,
                                       const std::string& label, uint32_t textureIndex, bool hasTexture,
                                       const Theme* theme, float depth) {
    // Skip if entirely off-screen (culling)
    if (m_contentY + TEXTURE_ROW_HEIGHT < m_visibleMinY || m_contentY > m_visibleMaxY) {
        m_contentY += TEXTURE_ROW_HEIGHT;
        return;
    }

    // Draw label
    if (textLayout) {
        float textY = m_contentY + 4.0f;
        textLayout->layout(label, make_float2(m_bounds.x + PADDING, textY), 0.45f,
                          theme->propertyLabel, TextAlign::Left, depth + 0.001f, outQuads);
    }

    // Calculate preview rect (centered below label)
    float previewX = m_bounds.x + PADDING;
    float previewY = m_contentY + 20.0f;
    Rect previewRect = { previewX, previewY, TEXTURE_PREVIEW_SIZE, TEXTURE_PREVIEW_SIZE };

    // Draw border around preview
    Rect borderRect = { previewX - 1.0f, previewY - 1.0f,
                        TEXTURE_PREVIEW_SIZE + 2.0f, TEXTURE_PREVIEW_SIZE + 2.0f };
    outQuads.push_back(makeSolidQuad(borderRect, theme->panelBorder, depth + 0.001f));

    if (hasTexture && textureIndex != UINT32_MAX) {
        // Draw actual texture preview
        outQuads.push_back(makeTexturedQuad(previewRect, textureIndex, depth + 0.002f));
    } else {
        // Draw placeholder (checkerboard pattern represented by gray)
        outQuads.push_back(makeSolidQuad(previewRect, make_float4(0.3f, 0.3f, 0.3f, 1.0f), depth + 0.002f));

        // Draw "No Texture" text
        if (textLayout) {
            float noTexY = previewY + TEXTURE_PREVIEW_SIZE * 0.5f - textLayout->getLineHeight(0.35f) * 0.5f;
            textLayout->layout("No Tex", make_float2(previewX + TEXTURE_PREVIEW_SIZE * 0.5f, noTexY), 0.35f,
                              theme->textSecondary, TextAlign::Center, depth + 0.003f, outQuads);
        }
    }

    m_contentY += TEXTURE_ROW_HEIGHT;
}

//------------------------------------------------------------------------------
// Scroll support
//------------------------------------------------------------------------------
bool PropertyPanel::needsScrolling() const {
    return m_totalContentHeight > (m_bounds.height - HEADER_HEIGHT);
}

void PropertyPanel::clampScrollOffset() {
    float maxScroll = std::max(0.0f, m_totalContentHeight - (m_bounds.height - HEADER_HEIGHT - PADDING));
    m_scrollOffset = std::clamp(m_scrollOffset, 0.0f, maxScroll);
}

float PropertyPanel::getScrollbarHeight() const {
    float viewHeight = m_bounds.height - HEADER_HEIGHT;
    if (m_totalContentHeight <= 0.0f) return viewHeight;
    float ratio = viewHeight / m_totalContentHeight;
    return std::max(20.0f, viewHeight * ratio);
}

float PropertyPanel::getScrollbarY() const {
    float viewHeight = m_bounds.height - HEADER_HEIGHT;
    float maxScroll = std::max(1.0f, m_totalContentHeight - viewHeight);
    float scrollRatio = m_scrollOffset / maxScroll;
    float trackHeight = viewHeight - getScrollbarHeight();
    return m_bounds.y + HEADER_HEIGHT + scrollRatio * trackHeight;
}

Rect PropertyPanel::getScrollbarRect() const {
    return {
        m_bounds.right() - SCROLLBAR_WIDTH,
        getScrollbarY(),
        SCROLLBAR_WIDTH,
        getScrollbarHeight()
    };
}

Rect PropertyPanel::getContentClipBounds() const {
    return {
        m_bounds.x,
        m_bounds.y + HEADER_HEIGHT,
        m_bounds.width - SCROLLBAR_WIDTH,  // Always clip to leave room for scrollbar
        m_bounds.height - HEADER_HEIGHT
    };
}

void PropertyPanel::drawScrollbar(std::vector<UIQuad>& outQuads, const Theme* theme, float depth) {
    if (!needsScrolling()) return;
    
    // Scrollbar track
    Rect trackRect = {
        m_bounds.right() - SCROLLBAR_WIDTH,
        m_bounds.y + HEADER_HEIGHT,
        SCROLLBAR_WIDTH,
        m_bounds.height - HEADER_HEIGHT
    };
    outQuads.push_back(makeSolidQuad(trackRect, theme->scrollbarTrack, depth + 0.01f));
    
    // Scrollbar thumb
    Rect thumbRect = getScrollbarRect();
    float4 thumbColor = m_scrollbarDragging ? theme->scrollbarThumbActive : theme->scrollbarThumb;
    outQuads.push_back(makeSolidQuad(thumbRect, thumbColor, depth + 0.02f));
}

void PropertyPanel::collectGeometry(std::vector<UIQuad>& outQuads, text::TextLayout* textLayout) {
    if (!m_visible) return;

    // Cache bounds to avoid repeated getAbsoluteBounds() calls
    m_bounds = getAbsoluteBounds();

    // Update panel position/theme only if dirty
    updatePanelAndTheme();

    // Render panel background
    if (m_panel) {
        m_panel->collectGeometry(outQuads, textLayout);
    }

    // Setup layout state
    const Theme* theme = getTheme();
    float depth = getEffectiveDepth();
    
    // Always reserve scrollbar space to avoid layout oscillation
    // (content width doesn't change when scrollbar appears/disappears)
    m_contentWidth = m_bounds.width - PADDING * 2 - SCROLLBAR_WIDTH;
    
    // Apply scroll offset to starting Y
    float contentStartY = m_bounds.y + HEADER_HEIGHT + PADDING;
    m_contentY = contentStartY - m_scrollOffset;
    
    // Track where content starts for clipping
    size_t contentStartIdx = outQuads.size();
    Rect clipBounds = getContentClipBounds();

    // Calculate visible Y range for culling
    m_visibleMinY = m_bounds.y + HEADER_HEIGHT;
    m_visibleMaxY = m_bounds.y + m_bounds.height;

    if (m_displayMode == DisplayMode::Instance) {
        // Cache instance ID string to avoid per-frame allocation
        if (m_instanceInfo.instanceId != m_cachedInstanceId) {
            m_cachedIdStr = std::to_string(m_instanceInfo.instanceId);
            m_cachedInstanceId = m_instanceInfo.instanceId;
        }

        // Instance header
        drawHeader(outQuads, textLayout, "Instance", theme, depth);
        drawPropertyRow(outQuads, textLayout, "ID", m_cachedIdStr, theme, depth);
        drawPropertyRow(outQuads, textLayout, "Model", m_instanceInfo.modelName, theme, depth);
        if (!m_instanceInfo.meshName.empty()) {
            drawPropertyRow(outQuads, textLayout, "Mesh", m_instanceInfo.meshName, theme, depth);
        }

        // Material section
        drawSeparator(outQuads, theme, depth);
        drawHeader(outQuads, textLayout, "Material", theme, depth);

        // Base Color picker (needs full height for 3 RGB sliders)
        addSpacing(4.0f);
        drawLabel(outQuads, textLayout, "Base Color", theme, depth);
        addWidget(m_baseColorPicker.get(), ColorPicker::SLIDER_HEIGHT * 3 + ColorPicker::SLIDER_SPACING * 2 + 8.0f, outQuads, textLayout);
        addSpacing(4.0f);

        // Material sliders
        addWidget(m_metallicSlider.get(), ROW_HEIGHT, outQuads, textLayout);
        addSpacing(2.0f);

        addWidget(m_roughnessSlider.get(), ROW_HEIGHT, outQuads, textLayout);
        addSpacing(4.0f);

        // Emissive picker (needs full height for 3 RGB sliders)
        drawLabel(outQuads, textLayout, "Emissive", theme, depth);
        addWidget(m_emissivePicker.get(), ColorPicker::SLIDER_HEIGHT * 3 + ColorPicker::SLIDER_SPACING * 2 + 8.0f, outQuads, textLayout);
        addSpacing(4.0f);

        addWidget(m_emissiveIntensitySlider.get(), ROW_HEIGHT, outQuads, textLayout);

        // Textures section - only show if textures are actually loaded
        bool hasAnyTexture = m_instanceInfo.hasBaseColorTex || m_instanceInfo.hasNormalTex ||
                             m_instanceInfo.hasMetallicRoughnessTex || m_instanceInfo.hasEmissiveTex;

        if (hasAnyTexture) {
            drawSeparator(outQuads, theme, depth);
            drawHeader(outQuads, textLayout, "Textures", theme, depth);
            addSpacing(4.0f);
            
            // Only show previews for textures that are actually loaded
            if (m_instanceInfo.hasBaseColorTex) {
                drawTexturePreview(outQuads, textLayout, "Base Color", 
                                   m_instanceInfo.baseColorTexIndex, true, theme, depth);
            }
            
            if (m_instanceInfo.hasNormalTex) {
                drawTexturePreview(outQuads, textLayout, "Normal", 
                                   m_instanceInfo.normalTexIndex, true, theme, depth);
            }
            
            if (m_instanceInfo.hasMetallicRoughnessTex) {
                drawTexturePreview(outQuads, textLayout, "Metal/Rough", 
                                   m_instanceInfo.metallicRoughnessTexIndex, true, theme, depth);
            }
            
            if (m_instanceInfo.hasEmissiveTex) {
                drawTexturePreview(outQuads, textLayout, "Emissive", 
                                   m_instanceInfo.emissiveTexIndex, true, theme, depth);
            }
        }

    } else if (m_displayMode == DisplayMode::Light) {
        // Light header
        std::string lightTypeName;
        switch (m_lightInfo.type) {
            case SceneNodeType::PointLight: lightTypeName = "Point Light"; break;
            case SceneNodeType::DirectionalLight: lightTypeName = "Directional Light"; break;
            case SceneNodeType::AreaLight: lightTypeName = "Area Light"; break;
            default: lightTypeName = "Light"; break;
        }
        drawHeader(outQuads, textLayout, lightTypeName, theme, depth);

        // Position (for point/area lights)
        if (m_lightInfo.type != SceneNodeType::DirectionalLight) {
            addSpacing(4.0f);
            drawLabel(outQuads, textLayout, "Position", theme, depth);
            addWidget(m_posXSlider.get(), ROW_HEIGHT, outQuads, textLayout);
            addSpacing(2.0f);
            addWidget(m_posYSlider.get(), ROW_HEIGHT, outQuads, textLayout);
            addSpacing(2.0f);
            addWidget(m_posZSlider.get(), ROW_HEIGHT, outQuads, textLayout);
            addSpacing(SECTION_SPACING);
        }

        // Color
        drawLabel(outQuads, textLayout, "Color", theme, depth);
        addWidget(m_colorPicker.get(), ColorPicker::SLIDER_HEIGHT * 3 + 8.0f, outQuads, textLayout);
        addSpacing(4.0f);

        // Intensity
        addWidget(m_intensitySlider.get(), ROW_HEIGHT, outQuads, textLayout);
        addSpacing(SECTION_SPACING);

        // Type-specific
        if (m_lightInfo.type == SceneNodeType::PointLight) {
            addWidget(m_radiusSlider.get(), ROW_HEIGHT, outQuads, textLayout);
        } else if (m_lightInfo.type == SceneNodeType::DirectionalLight) {
            addWidget(m_angularDiameterSlider.get(), ROW_HEIGHT, outQuads, textLayout);
        } else if (m_lightInfo.type == SceneNodeType::AreaLight) {
            drawLabel(outQuads, textLayout, "Size", theme, depth);
            addWidget(m_sizeXSlider.get(), ROW_HEIGHT, outQuads, textLayout);
            addSpacing(2.0f);
            addWidget(m_sizeYSlider.get(), ROW_HEIGHT, outQuads, textLayout);
        }

    } else {
        // Empty state
        if (textLayout) {
            float textY = m_bounds.y + m_bounds.height * 0.5f - textLayout->getLineHeight(0.5f) * 0.5f;
            textLayout->layout("Select an object", make_float2(m_bounds.x + m_bounds.width * 0.5f, textY), 0.5f,
                              theme->textSecondary, TextAlign::Center, depth + 0.001f, outQuads);
        }
    }

    // Calculate total content height (for scroll)
    m_totalContentHeight = (m_contentY + m_scrollOffset) - contentStartY + PADDING;
    clampScrollOffset();
    
    // Apply clip bounds to all content quads (but not the panel background)
    for (size_t i = contentStartIdx; i < outQuads.size(); i++) {
        outQuads[i].clipMinX = clipBounds.x;
        outQuads[i].clipMinY = clipBounds.y;
        outQuads[i].clipMaxX = clipBounds.right();
        outQuads[i].clipMaxY = clipBounds.bottom();
    }
    
    // Draw scrollbar on top (no clipping)
    drawScrollbar(outQuads, theme, depth);

    m_dirty = false;
}

void PropertyPanel::generateGeometry(std::vector<UIQuad>& outQuads, text::TextLayout* textLayout) {
    // Panel handles background rendering
}

bool PropertyPanel::onMouseDown(float2 pos, int button) {
    if (!m_visible || !m_enabled) return false;

    // Forward to Panel first (handles dragging)
    if (m_panel && m_panel->onMouseDown(pos, button)) return true;

    // Check scrollbar interaction
    if (button == 0 && needsScrolling()) {
        Rect scrollbarRect = getScrollbarRect();
        if (scrollbarRect.contains(pos)) {
            m_scrollbarDragging = true;
            m_dragStartOffset = m_scrollOffset;
            m_dragStartY = pos.y;
            markDirty();
            return true;
        }
        
        // Click on track - jump to position
        Rect trackRect = {
            m_bounds.right() - SCROLLBAR_WIDTH,
            m_bounds.y + HEADER_HEIGHT,
            SCROLLBAR_WIDTH,
            m_bounds.height - HEADER_HEIGHT
        };
        if (trackRect.contains(pos)) {
            float viewHeight = m_bounds.height - HEADER_HEIGHT;
            float clickRatio = (pos.y - (m_bounds.y + HEADER_HEIGHT)) / viewHeight;
            float maxScroll = std::max(1.0f, m_totalContentHeight - viewHeight);
            m_scrollOffset = clickRatio * maxScroll;
            clampScrollOffset();
            markDirty();
            return true;
        }
    }

    // Forward to visible widgets
    Widget* widgets[] = {
        m_posXSlider.get(), m_posYSlider.get(), m_posZSlider.get(),
        m_colorPicker.get(), m_intensitySlider.get(), m_radiusSlider.get(),
        m_angularDiameterSlider.get(), m_sizeXSlider.get(), m_sizeYSlider.get(),
        m_baseColorPicker.get(), m_metallicSlider.get(), m_roughnessSlider.get(),
        m_emissivePicker.get(), m_emissiveIntensitySlider.get()
    };
    for (Widget* w : widgets) {
        if (w && w->isVisible() && w->onMouseDown(pos, button)) return true;
    }

    return containsPoint(pos);
}

bool PropertyPanel::onMouseUp(float2 pos, int button) {
    if (!m_visible || !m_enabled) return false;

    bool consumed = false;

    // Handle scrollbar drag end
    if (m_scrollbarDragging) {
        m_scrollbarDragging = false;
        markDirty();
        consumed = true;
    }

    // Forward to Panel
    if (m_panel) {
        bool wasDragging = m_panel->isDragging();
        consumed = m_panel->onMouseUp(pos, button) || consumed;
        if (wasDragging) {
            Rect panelBounds = m_panel->getAbsoluteBounds();
            setPosition(panelBounds.x, panelBounds.y);
            markDirty();
        }
    }

    // Forward to widgets
    Widget* widgets[] = {
        m_posXSlider.get(), m_posYSlider.get(), m_posZSlider.get(),
        m_colorPicker.get(), m_intensitySlider.get(), m_radiusSlider.get(),
        m_angularDiameterSlider.get(), m_sizeXSlider.get(), m_sizeYSlider.get(),
        m_baseColorPicker.get(), m_metallicSlider.get(), m_roughnessSlider.get(),
        m_emissivePicker.get(), m_emissiveIntensitySlider.get()
    };
    for (Widget* w : widgets) {
        if (w) consumed = w->onMouseUp(pos, button) || consumed;
    }

    return consumed || containsPoint(pos);
}

bool PropertyPanel::onMouseMove(float2 pos) {
    if (!m_visible || !m_enabled) return false;

    bool consumed = false;

    // Handle scrollbar dragging
    if (m_scrollbarDragging) {
        float viewHeight = m_bounds.height - HEADER_HEIGHT;
        float trackHeight = viewHeight - getScrollbarHeight();
        if (trackHeight > 0) {
            float deltaY = pos.y - m_dragStartY;
            float maxScroll = std::max(1.0f, m_totalContentHeight - viewHeight);
            float scrollDelta = (deltaY / trackHeight) * maxScroll;
            m_scrollOffset = m_dragStartOffset + scrollDelta;
            clampScrollOffset();
            markDirty();
        }
        return true;
    }

    // Forward to Panel
    if (m_panel) {
        consumed = m_panel->onMouseMove(pos);
        if (m_panel->isDragging()) {
            Rect panelBounds = m_panel->getAbsoluteBounds();
            setPosition(panelBounds.x, panelBounds.y);
            markDirty();
        }
    }

    // Forward to widgets
    Widget* widgets[] = {
        m_posXSlider.get(), m_posYSlider.get(), m_posZSlider.get(),
        m_colorPicker.get(), m_intensitySlider.get(), m_radiusSlider.get(),
        m_angularDiameterSlider.get(), m_sizeXSlider.get(), m_sizeYSlider.get(),
        m_baseColorPicker.get(), m_metallicSlider.get(), m_roughnessSlider.get(),
        m_emissivePicker.get(), m_emissiveIntensitySlider.get()
    };
    for (Widget* w : widgets) {
        if (w) consumed = w->onMouseMove(pos) || consumed;
    }

    return consumed;
}

bool PropertyPanel::onMouseScroll(float2 pos, float delta) {
    if (!m_visible || !m_enabled) return false;
    
    if (containsPoint(pos) && needsScrolling()) {
        m_scrollOffset -= delta * m_scrollSpeed;
        clampScrollOffset();
        markDirty();
        return true;
    }
    
    // Only consume if we're over the panel AND it could scroll (just not needed now)
    // Don't consume scroll events unnecessarily
    return false;
}

} // namespace ui
} // namespace spectra
