#pragma once

#include "widget.h"
#include "panel.h"
#include "label.h"
#include "slider.h"
#include "color_picker.h"
#include "scroll_view.h"
#include "../scene/scene_hierarchy.h"
#include "../core/shared_types.h"
#include <memory>
#include <functional>
#include <string>

namespace spectra {
namespace ui {

//------------------------------------------------------------------------------
// Light Info - Data structure for light property editing
//------------------------------------------------------------------------------
struct LightInfo {
    SceneNodeType type;
    uint32_t index;

    // Position (for point/area lights)
    float3 position;

    // Direction (for directional lights)
    float3 direction;

    // Color/emission
    float3 color;

    // Intensity/irradiance magnitude (extracted from color for easier editing)
    float intensity;

    // Type-specific
    float radius;           // Point light radius
    float angularDiameter;  // Directional light angular diameter
    float2 size;            // Area light size
};

//------------------------------------------------------------------------------
// Instance Info - Data structure for instance property display and editing
//------------------------------------------------------------------------------
struct InstanceInfo {
    uint32_t instanceId;
    uint32_t materialIndex;
    std::string modelName;
    std::string meshName;

    // Material properties (editable)
    float4 baseColor;
    float metallic;
    float roughness;
    float3 emissive;

    // Texture paths (full paths for display)
    std::string baseColorTexPath;
    std::string normalTexPath;
    std::string metallicRoughnessTexPath;
    std::string emissiveTexPath;
    std::string occlusionTexPath;

    // Texture handles for preview (indices into UI texture array)
    uint32_t baseColorTexIndex = UINT32_MAX;
    uint32_t normalTexIndex = UINT32_MAX;
    uint32_t metallicRoughnessTexIndex = UINT32_MAX;
    uint32_t emissiveTexIndex = UINT32_MAX;
    uint32_t occlusionTexIndex = UINT32_MAX;

    // For tracking if material has textures
    bool hasBaseColorTex = false;
    bool hasNormalTex = false;
    bool hasMetallicRoughnessTex = false;
    bool hasEmissiveTex = false;
};

//------------------------------------------------------------------------------
// PropertyPanel - Shows and edits properties for selected objects
//------------------------------------------------------------------------------
class PropertyPanel : public Widget {
public:
    PropertyPanel();
    ~PropertyPanel() override = default;

    // Show properties for an instance
    void showInstanceProperties(const InstanceInfo& info);

    // Show properties for a light
    void showLightProperties(const LightInfo& info);

    // Clear the panel (no selection)
    void clearProperties();

    // Light edit callback
    using LightEditCallback = std::function<void(SceneNodeType, uint32_t, const LightInfo&)>;
    void setOnLightEdit(LightEditCallback callback) { m_onLightEdit = callback; }

    // Material edit callback
    using MaterialEditCallback = std::function<void(uint32_t instanceId, const GpuMaterial& material)>;
    void setOnMaterialEdit(MaterialEditCallback callback) { m_onMaterialEdit = callback; }

    // Close callback
    using CloseCallback = std::function<void()>;
    void setOnClose(CloseCallback callback) { m_onClose = callback; }

    // Draggable panel (delegates to internal Panel)
    void setDraggable(bool draggable);
    bool isDraggable() const;

    // Override to collect child geometry
    void collectGeometry(std::vector<UIQuad>& outQuads, text::TextLayout* textLayout) override;

    // Override input handling
    bool onMouseDown(float2 pos, int button) override;
    bool onMouseUp(float2 pos, int button) override;
    bool onMouseMove(float2 pos) override;
    bool onMouseScroll(float2 pos, float delta) override;

protected:
    void generateGeometry(std::vector<UIQuad>& outQuads, text::TextLayout* textLayout) override;

private:
    void createWidgets();
    void updatePanelAndTheme();
    void updateLightFromSliders();
    void updateMaterialFromSliders();

    // Layout state - tracks current Y position during collectGeometry
    float m_contentY = 0.0f;
    float m_contentWidth = 0.0f;
    Rect m_bounds;
    
    // Scroll support
    float m_scrollOffset = 0.0f;
    float m_totalContentHeight = 0.0f;
    float m_scrollSpeed = 30.0f;
    bool m_scrollbarDragging = false;
    float m_dragStartOffset = 0.0f;
    float m_dragStartY = 0.0f;
    
    // Cached values for optimization
    Rect m_lastBounds = {0.0f, 0.0f, 0.0f, 0.0f};
    const Theme* m_lastTheme = nullptr;

    // Cached instance ID string to avoid per-frame std::to_string allocations
    mutable std::string m_cachedIdStr;
    mutable uint32_t m_cachedInstanceId = UINT32_MAX;

    // Visible bounds for culling
    float m_visibleMinY = 0.0f;
    float m_visibleMaxY = 0.0f;

    // Add widget: positions, sizes, renders, and advances contentY automatically
    void addWidget(Widget* widget, float height, std::vector<UIQuad>& outQuads, text::TextLayout* textLayout);
    
    // Add spacing
    void addSpacing(float amount) { m_contentY += amount; }

    // Section rendering helpers
    void drawHeader(std::vector<UIQuad>& outQuads, text::TextLayout* textLayout,
                    const std::string& title, const Theme* theme, float depth);
    void drawPropertyRow(std::vector<UIQuad>& outQuads, text::TextLayout* textLayout,
                         const std::string& label, const std::string& value,
                         const Theme* theme, float depth);
    void drawSeparator(std::vector<UIQuad>& outQuads, const Theme* theme, float depth);
    void drawLabel(std::vector<UIQuad>& outQuads, text::TextLayout* textLayout,
                   const std::string& text, const Theme* theme, float depth);
    void drawTexturePreview(std::vector<UIQuad>& outQuads, text::TextLayout* textLayout,
                            const std::string& label, uint32_t textureIndex, bool hasTexture,
                            const Theme* theme, float depth);
    void drawScrollbar(std::vector<UIQuad>& outQuads, const Theme* theme, float depth);
    
    // Scroll helpers
    bool needsScrolling() const;
    void clampScrollOffset();
    float getScrollbarHeight() const;
    float getScrollbarY() const;
    Rect getScrollbarRect() const;
    Rect getContentClipBounds() const;

    // Panel container
    std::unique_ptr<Panel> m_panel;

    // Current display mode
    enum class DisplayMode {
        Empty,
        Instance,
        Light
    };
    DisplayMode m_displayMode = DisplayMode::Empty;

    // Cached info
    InstanceInfo m_instanceInfo;
    LightInfo m_lightInfo;

    // Light editing sliders
    std::unique_ptr<Slider> m_posXSlider;
    std::unique_ptr<Slider> m_posYSlider;
    std::unique_ptr<Slider> m_posZSlider;
    std::unique_ptr<ColorPicker> m_colorPicker;
    std::unique_ptr<Slider> m_intensitySlider;
    std::unique_ptr<Slider> m_radiusSlider;
    std::unique_ptr<Slider> m_angularDiameterSlider;
    std::unique_ptr<Slider> m_sizeXSlider;
    std::unique_ptr<Slider> m_sizeYSlider;

    // Material editing sliders
    std::unique_ptr<ColorPicker> m_baseColorPicker;
    std::unique_ptr<Slider> m_metallicSlider;
    std::unique_ptr<Slider> m_roughnessSlider;
    std::unique_ptr<ColorPicker> m_emissivePicker;
    std::unique_ptr<Slider> m_emissiveIntensitySlider;

    // Callbacks
    LightEditCallback m_onLightEdit;
    MaterialEditCallback m_onMaterialEdit;
    CloseCallback m_onClose;

    // Layout constants
    static constexpr float HEADER_HEIGHT = 28.0f;
    static constexpr float ROW_HEIGHT = 22.0f;
    static constexpr float SECTION_SPACING = 12.0f;
    static constexpr float PADDING = 8.0f;
    static constexpr float TEXTURE_PREVIEW_SIZE = 160.0f;
    static constexpr float TEXTURE_ROW_HEIGHT = 188.0f;  // Preview + label + padding
    static constexpr float SCROLLBAR_WIDTH = 8.0f;
};

} // namespace ui
} // namespace spectra
