# Real-Time OptiX Ray Tracing Engine - Phase 4 Specification

## Phase 4: Custom Ray-Traced UI System

### Objective

Build a custom UI system rendered entirely through OptiX ray tracing. Implement scene hierarchy panel for object selection and basic interaction. Use optimized architecture: retained mode widgets, dual raygen pipeline, SDF text rendering, and screen-space coordinates.

**Success Criteria:**
- Top bar with button to toggle scene hierarchy panel
- Scene hierarchy panel displays model/mesh tree structure
- Click on tree items to select objects
- Selected objects highlighted in viewport
- Light and dark theme support
- Scene save/load to JSON (preserves transforms, camera, settings)
- UI renders at < 1ms overhead
- 60 FPS maintained with UI visible
- Text is crisp and readable at any scale

---

## System Architecture

### Complete UI Pipeline

```
Input Events (Mouse) → Widget System (Retained Mode) → Geometry Updates
                                                              ↓
                                                    GPU Upload & BVH Update
                                                              ↓
                    ┌─────────────────────────────────────────┴─────────┐
                    ↓                                                     ↓
            Raygen Pass 1: Scene                                 Raygen Pass 2: UI
         (Perspective camera)                              (Orthographic projection)
         Trace scene IAS                                   Trace UI IAS
         Output: scene_buffer                              Sample SDF atlas
                    ↓                                      Output: ui_buffer (RGBA)
                    └─────────────────────────────────────────┬─────────┘
                                                              ↓
                                                    Composite Pass
                                          final = scene + ui * ui.alpha
                                                              ↓
                                                        Display Output
```

---

## Project Structure

```
project/
├── src/
│   ├── ui/
│   │   ├── ui_manager.h/cpp           # Central UI coordinator
│   │   ├── widget.h/cpp               # Base widget class
│   │   ├── panel.h/cpp                # Container widget
│   │   ├── button.h/cpp               # Clickable button
│   │   ├── label.h/cpp                # Text display
│   │   ├── tree_node.h/cpp            # Tree view item
│   │   ├── ui_geometry.h/cpp          # GPU geometry management
│   │   ├── theme.h/cpp                # Light/dark themes
│   │   └── input_system.h/cpp         # Mouse event handling
│   │
│   ├── text/
│   │   ├── font_atlas.h/cpp           # SDF atlas loading
│   │   ├── text_layout.h/cpp          # String → glyph quads
│   │   └── glyph_cache.h              # Glyph metrics lookup
│   │
│   ├── scene/
│   │   ├── scene_serializer.h/cpp     # JSON save/load
│   │   └── selection_manager.h/cpp    # Selection & highlighting
│   │
│   └── (existing files from Phase 1-3)
│
├── optix_programs/
│   ├── raygen_scene.cu                # Scene rendering (existing)
│   ├── raygen_ui.cu                   # UI rendering (NEW)
│   ├── closesthit_ui.cu               # UI shading (NEW)
│   ├── miss_ui.cu                     # UI miss (NEW)
│   └── composite.cu                   # Blend scene+UI (NEW)
│
├── assets/
│   ├── fonts/
│   │   ├── roboto_sdf.png            # SDF atlas texture
│   │   └── roboto.json               # Glyph metrics
│   └── scenes/
│       └── default.json              # Default scene
│
└── tools/
    └── generate_sdf_atlas.py         # Atlas generation script
```

---

## Implementation Checklist

### 1. SDF Font Atlas Generation (Offline)

**Reference:** https://github.com/Chlumsky/msdfgen

**Tools Setup:**
- [ ] Install msdfgen (binary or build from source)
- [ ] Create Python automation script
- [ ] Choose font (Roboto recommended for UI)

**Atlas Generation:**
- [ ] Character set: ASCII printable (32-126) + common symbols
- [ ] Atlas size: 1024x1024 (or 512x512 for smaller fonts)
- [ ] Pixel range: 4-8 (controls SDF quality)
- [ ] Output: PNG atlas (RGB multi-channel) + JSON metrics

**JSON Metrics Format:**
- Atlas dimensions (width, height)
- Font size and line height
- Per-glyph data:
  - UV coordinates in atlas
  - Advance width
  - Bearing X/Y
  - Glyph width/height

**Validation:**
- [ ] Verify all characters rendered correctly in atlas
- [ ] Check JSON contains all required glyphs
- [ ] Test different pixel ranges if quality issues

---

### 2. Font Atlas Loading

**Font Atlas Structure:**
- [ ] Load PNG texture using stb_image
- [ ] Upload to CUDA texture object (RGB8, linear filtering)
- [ ] Parse JSON metrics into glyph lookup map
- [ ] Store font size, line height
- [ ] Validate all ASCII characters present

**Glyph Metrics Data:**
- UV rectangle in atlas
- Horizontal advance to next character
- Bearing offsets (horizontal and vertical)
- Glyph dimensions

---

### 3. Text Layout System

**Text Layout Algorithm:**
- [ ] Input: string, position, font size, alignment
- [ ] For each character:
  - Look up glyph metrics
  - Calculate quad position (cursor + bearing + scale)
  - Generate 4 vertices with UVs from metrics
  - Advance cursor by glyph advance * scale
- [ ] Output: array of positioned quads

**Text Measurement:**
- [ ] Measure string width (sum of advances)
- [ ] Measure string height (line height)
- [ ] Support alignment (left, center, right)
- [ ] Scale from atlas font size to desired size

**Alignment Implementation:**
- Left: Start at position.x
- Center: position.x - width/2
- Right: position.x - width

---

### 4. Widget System Foundation

**Base Widget Class:**

**Core Functionality:**
- Position and size (screen-space pixels)
- Depth (Z-order for layering)
- Visibility and enabled state
- Hover and active state tracking
- Dirty flag (needs geometry regeneration)
- Geometry storage (vertices, indices)
- Children list (for containers)

**Virtual Methods:**
- update(dt) - Update logic
- render() - Generate geometry if dirty
- onMouseMove(x, y) - Hover detection
- onMouseDown(x, y, button) - Click handling
- onMouseUp(x, y, button) - Release handling
- regenerateGeometry() - Pure virtual, widget-specific

**Helper Methods:**
- containsPoint(x, y) - Hit testing
- markDirty() - Set dirty flag, propagate to parent
- setPosition(x, y)
- setSize(width, height)

**Geometry Structure:**
- Vertices: position (x, y, depth), UV, color
- Indices: Triangle list
- Widget ID: For ray picking

---

### 5. Panel Widget (Container)

**Purpose:** Rectangular background container for child widgets

**Properties:**
- [ ] Background color
- [ ] Border color and width
- [ ] Padding (optional, can skip for Phase 4)

**Geometry:**
- Background: Single quad (2 triangles)
- Border: Four edge quads (optional)
- Depth: Set to ensure proper layering

**Child Management:**
- [ ] Add/remove children
- [ ] Position children relative to panel origin
- [ ] Propagate events to children
- [ ] Collect geometry from all children

---

### 6. Label Widget (Text Display)

**Purpose:** Display static or dynamic text

**Properties:**
- [ ] Text string
- [ ] Font size
- [ ] Text color
- [ ] Alignment (left, center, right)

**Geometry:**
- [ ] Use TextLayout to generate quads for each glyph
- [ ] Position based on alignment
- [ ] Set vertex colors to text color
- [ ] UVs from glyph metrics
- [ ] Depth slightly in front of background

**Dynamic Updates:**
- [ ] setText() marks widget dirty
- [ ] Regenerates geometry on next render
- [ ] Efficient for status text, labels

---

### 7. Button Widget (Interactive)

**Purpose:** Clickable button with text label

**Properties:**
- [ ] Label text
- [ ] Normal, hover, active, disabled colors
- [ ] Click callback function

**States:**
- Normal: Default appearance
- Hovered: Mouse over button
- Active: Mouse pressed down
- Disabled: Non-interactive, grayed out

**Geometry:**
- [ ] Background quad (color varies by state)
- [ ] Centered text label
- [ ] Optional border
- [ ] Regenerate on state change

**Event Handling:**
- [ ] onMouseMove: Check if hovering, update state
- [ ] onMouseDown: Set active state
- [ ] onMouseUp: If still hovering, invoke callback

---

### 8. TreeNode Widget (Hierarchical)

**Purpose:** Expandable/collapsible tree item for scene hierarchy

**Properties:**
- [ ] Label text
- [ ] Expanded/collapsed state
- [ ] Selected state
- [ ] Depth level (for indentation)
- [ ] Child nodes

**Visual Elements:**
- [ ] Expand/collapse icon (triangle: right=collapsed, down=expanded)
- [ ] Indentation based on depth (e.g., 20px per level)
- [ ] Selection highlight background
- [ ] Label text

**Geometry:**
- [ ] Background quad (if selected or hovered)
- [ ] Icon triangle at left
- [ ] Label text after icon
- [ ] Recursively render children if expanded

**Event Handling:**
- [ ] Click on icon: Toggle expanded state
- [ ] Click on label: Set selected, invoke callback
- [ ] Recursive hit testing (check self, then children)

**Layout:**
- [ ] Children positioned below parent
- [ ] Y offset accumulates for each child
- [ ] Only visible children contribute to height

---

### 9. UI Geometry Management

**UIGeometry System:**

**Purpose:** Manage all UI geometry on GPU

**Responsibilities:**
- [ ] Pre-allocate large vertex/index buffers
- [ ] Collect geometry from all widgets each frame
- [ ] Upload changed geometry to GPU
- [ ] Build and maintain UI IAS
- [ ] Map primitives to widget IDs

**Buffer Management:**
- [ ] Allocate max size buffers (e.g., 100K vertices)
- [ ] Track current usage
- [ ] Sub-allocate to widgets
- [ ] Reallocate if exceeded (rare)

**Upload Strategy:**
- [ ] Collect all dirty widget geometry
- [ ] Batch into single CPU buffer
- [ ] Single cudaMemcpy to GPU
- [ ] Update primitive-to-widget ID mapping

**BVH Construction:**
- [ ] Build IAS with ALLOW_UPDATE flag
- [ ] Use optixAccelUpdate when only positions change
- [ ] Full rebuild when topology changes (widgets added/removed)
- [ ] Separate from scene IAS

**Performance Target:**
- Geometry upload: < 0.2ms
- BVH update: < 0.3ms
- Total: < 0.5ms per frame

---

### 10. Input System

**Event Types:**
- MouseMove: Position change
- MouseDown: Button pressed
- MouseUp: Button released
- MouseScroll: Wheel movement (optional for Phase 4)

**Input Manager:**
- [ ] Register GLFW callbacks
- [ ] Queue events for processing
- [ ] Track mouse position and button states
- [ ] Convert GLFW coordinates to screen-space

**Event Processing:**
- [ ] Process queue at frame start
- [ ] For each event, perform hit testing (ray picking)
- [ ] Dispatch to appropriate widget
- [ ] Update hover states (only one widget hovered at a time)

**Hit Testing (Ray Picking):**
- [ ] Generate orthographic ray from mouse position
- [ ] Trace against UI IAS
- [ ] Get hit primitive index
- [ ] Map to widget ID
- [ ] Return widget pointer

---

### 11. UI Manager (Central Coordinator)

**UIManager Responsibilities:**

**Widget Management:**
- [ ] Store root widgets (top bar, panels)
- [ ] Update all widgets each frame
- [ ] Collect geometry from widgets
- [ ] Upload to GPU and maintain BVH

**Input Handling:**
- [ ] Receive events from InputManager
- [ ] Perform hit testing
- [ ] Dispatch to widgets
- [ ] Manage hover states globally

**Rendering:**
- [ ] Collect dirty widget geometry
- [ ] Upload to UIGeometry
- [ ] Trigger BVH update if needed
- [ ] Provide IAS handle to OptiX

**Theme Management:**
- [ ] Store current theme (light or dark)
- [ ] Provide theme to widgets
- [ ] Handle theme switching (mark all widgets dirty)

**Initialization:**
- [ ] Load font atlas
- [ ] Create initial widgets (top bar, hierarchy panel)
- [ ] Set up callbacks
- [ ] Initialize geometry system

---

### 12. Theme System

**Theme Structure:**

**Color Palette:**
- Panel: background, border
- Button: normal, hover, active, disabled
- Text: normal, disabled, selected
- Tree: background, selected, hover
- Misc: border, highlight

**Predefined Themes:**

**Light Theme:**
- Bright backgrounds (0.95 gray)
- Dark text (0.1 gray)
- Subtle highlights (blue-ish)

**Dark Theme:**
- Dark backgrounds (0.2 gray)
- Light text (0.9 gray)
- Vibrant highlights (brighter blue)

**Theme Switching:**
- [ ] Button in top bar
- [ ] Mark all widgets dirty on switch
- [ ] Widgets use theme colors in regenerateGeometry()
- [ ] Immediate visual update

---

### 13. Dual Raygen Pipeline

**OptiX Pipeline Changes:**

**Pass 1: Scene Rendering**
- Existing raygen_scene program
- Perspective camera
- Trace scene IAS
- Full material evaluation and lighting
- Output to scene_buffer (RGBA32F)

**Pass 2: UI Rendering**
- New raygen_ui program
- Orthographic projection (screen-space)
- Convert pixel coords to NDC
- Generate parallel rays (direction = [0, 0, -1])
- Trace UI IAS
- Output to ui_buffer (RGBA32F with alpha)

**Closest Hit UI:**
- [ ] Interpolate vertex attributes (UV, color)
- [ ] Sample SDF atlas texture
- [ ] Calculate median of RGB channels (signed distance)
- [ ] Apply smooth threshold to get alpha (smoothstep)
- [ ] Modulate vertex color alpha with SDF alpha
- [ ] Return RGBA via payload

**Miss UI:**
- [ ] Return transparent (RGBA = 0, 0, 0, 0)
- [ ] No UI at this pixel location

**Compositing:**
- [ ] Can be done in OptiX program or OpenGL shader
- [ ] Blend: final = scene * (1 - ui.alpha) + ui * ui.alpha
- [ ] Or: final = scene + ui * ui.alpha (if ui is premultiplied)
- [ ] Output to display buffer

**Launch Sequence:**
- [ ] Launch raygen_scene (entire viewport)
- [ ] Launch raygen_ui (entire viewport or UI regions only)
- [ ] Composite (single pass, per-pixel blend)
- [ ] Display final buffer

---

### 14. Scene Hierarchy Implementation

**Top Bar Widget:**

**Layout:**
- [ ] Fixed at top of screen (y=0)
- [ ] Full width
- [ ] Height: 40 pixels
- [ ] Dark background with border

**Contents:**
- [ ] Application title (left side)
- [ ] "Scene" button (toggles hierarchy panel)
- [ ] Theme toggle button (right side)

**Scene Hierarchy Panel:**

**Layout:**
- [ ] Fixed position (e.g., x=10, y=50)
- [ ] Fixed size (e.g., 300x400 pixels)
- [ ] Visible/hidden based on toggle button
- [ ] Semi-transparent background

**Contents:**
- [ ] Title bar ("Scene Hierarchy")
- [ ] Tree view of loaded models and meshes
- [ ] Scroll container if content exceeds height (optional, can skip)

**Tree Structure:**
- [ ] Root nodes: Loaded models
- [ ] Children: Meshes within each model
- [ ] Expand/collapse models
- [ ] Select individual meshes

**Integration:**
- [ ] Build tree from SceneManager data
- [ ] Update tree when models loaded/unloaded
- [ ] On selection: notify SelectionManager
- [ ] Highlight selected in viewport

---

### 15. Selection & Highlighting

**Selection Manager:**

**Responsibilities:**
- [ ] Track currently selected object (mesh/instance ID)
- [ ] Provide selection state to widgets
- [ ] Trigger highlighting in viewport

**Selection Methods:**
- Tree selection: User clicks in hierarchy panel
- Viewport selection: User clicks on object (optional for Phase 4)

**Highlighting Implementation:**

**Option A: Material Override (Simplest)**
- [ ] In closesthit_scene: check if primitive is selected
- [ ] If selected: tint color (e.g., multiply by highlight color)
- [ ] Easy to implement, always visible

**Option B: Wireframe Overlay**
- [ ] Render selected object edges as lines
- [ ] Requires edge geometry generation
- [ ] Second render pass or line primitives in IAS

**Option C: Outline Shader**
- [ ] Post-process pass
- [ ] Detect edges in depth/ID buffer
- [ ] Draw outline around selected object
- [ ] More complex but looks professional

**Recommendation:** Start with Option A (material tint), upgrade later if desired.

---

### 16. Scene Serialization

**Reference:** https://github.com/nlohmann/json

**Library:** nlohmann/json (header-only, excellent API)

**Scene File Format (JSON):**

**Top-Level Structure:**
- Version number
- Camera state
- Loaded models (paths, not geometry)
- Instance data (transforms, visibility, material overrides)
- Lights
- Render settings

**What to Save:**
- [ ] Camera: position, look-at, up, FOV
- [ ] Models: file paths relative to scene file
- [ ] Instances: per-instance transforms (4x4 matrix)
- [ ] Material overrides: if user changed any materials
- [ ] Lights: positions, colors, intensities, types
- [ ] Settings: quality mode, shadow enable, theme
- [ ] UI state: panel open/closed, positions (optional)

**Serialization:**
- [ ] SceneSerializer::save(filepath, scene)
- [ ] Traverse scene graph
- [ ] Convert to JSON object
- [ ] Write to file with pretty formatting

**Deserialization:**
- [ ] SceneSerializer::load(filepath) → Scene
- [ ] Parse JSON
- [ ] Load models (relative paths resolved)
- [ ] Create instances with transforms
- [ ] Restore camera and settings
- [ ] Build BVH

**Auto-save Strategy:**
- [ ] Save on significant changes (model loaded, transform changed)
- [ ] Debounce (don't save during continuous drag)
- [ ] Save to temp file first, atomic rename on success
- [ ] Option: Auto-save every N seconds

**Startup Behavior:**
- [ ] Check for last saved scene
- [ ] If exists: load automatically
- [ ] If not: Start with empty scene or default

---

## Performance Targets

**UI Rendering Overhead:**
- Geometry update: < 0.2ms
- BVH update: < 0.3ms
- UI raygen pass: < 0.5ms (mostly misses, very fast)
- Compositing: < 0.1ms
- **Total: < 1ms overhead**

**Overall Performance:**
- Scene rendering: 5-12ms (from Phase 3)
- UI rendering: < 1ms
- **Total: 6-13ms → 75-165 FPS**
- Target maintained: 60+ FPS

**Memory Usage:**
- Font atlas: ~1-4 MB (1024x1024 RGB)
- UI geometry: ~1-2 MB (vertex/index buffers)
- UI IAS: ~500 KB
- Widget metadata: < 1 MB
- **Total additional: < 10 MB**

---

## Common Issues & Solutions

**Text Looks Blurry:**
- Verify SDF atlas generated correctly
- Check pixel range in msdfgen (try 6-8)
- Ensure linear filtering on texture
- Verify smooth threshold in shader

**UI Not Responding to Clicks:**
- Check hit testing ray direction (should be [0,0,-1])
- Verify UI IAS built correctly
- Check widget containsPoint() logic
- Print hit primitive ID to debug

**Geometry Not Updating:**
- Check dirty flag propagation
- Verify geometry upload happening
- Ensure BVH update called when needed
- Check if widget marked visible

**Performance Issues:**
- Profile each component separately
- Check BVH rebuild vs update (should be update mostly)
- Verify only dirty widgets regenerate geometry
- Reduce UI complexity if needed

**Theme Switching Not Working:**
- Ensure all widgets marked dirty on theme change
- Check widget using theme colors in regenerateGeometry()
- Verify theme pointer passed to widgets

**Selection Not Highlighting:**
- Check SelectionManager state
- Verify selected ID passed to render
- Test material tint logic in closesthit
- Print selected ID to debug

---

## Validation Tests

**Font Rendering:**
- [ ] Text displays correctly at various sizes
- [ ] Text is sharp and readable (no blur)
- [ ] Special characters render correctly
- [ ] Text measurement accurate (alignment works)

**Widget Interaction:**
- [ ] Buttons respond to hover (color change)
- [ ] Buttons respond to click (callback fires)
- [ ] Tree nodes expand/collapse on icon click
- [ ] Tree nodes select on label click

**Scene Hierarchy:**
- [ ] All loaded models appear in tree
- [ ] Meshes appear as children
- [ ] Selection state updates correctly
- [ ] Selected object highlighted in viewport

**Theme Switching:**
- [ ] Toggle button switches theme
- [ ] All UI elements update colors
- [ ] Text remains readable in both themes
- [ ] No visual artifacts

**Scene Serialization:**
- [ ] Save scene to JSON file
- [ ] Load scene restores all state
- [ ] Model paths resolved correctly
- [ ] Camera and settings restored

**Performance:**
- [ ] UI overhead < 1ms
- [ ] 60 FPS maintained with UI visible
- [ ] No stuttering during interaction
- [ ] Smooth theme switching

---

## Reference Documentation

**SDF Text Rendering:**
- Valve Paper: https://steamcdn-a.akamaihd.net/apps/valve/2007/SIGGRAPH2007_AlphaTestedMagnification.pdf
- msdfgen: https://github.com/Chlumsky/msdfgen

**OptiX:**
- Dual Launch: https://raytracing-docs.nvidia.com/optix7/guide/index.html#program_pipeline_creation

**JSON:**
- nlohmann/json: https://github.com/nlohmann/json

**UI Patterns:**
- Dear ImGui (reference): https://github.com/ocornut/imgui

---

## Next Steps After Phase 4

Once custom UI is working:

**Future Enhancements:**
- Additional panels (properties, render settings, stats)
- Viewport gizmos (transform tools)
- Drag-and-drop in hierarchy
- Animation timeline (future)
- Material editor panel
- Light editor panel
- Performance profiler overlay

**Advanced Lighting (Future Phase):**
- ReSTIR for efficient many-light sampling
- Global illumination (indirect bounces)
- OptiX denoiser integration

This completes Phase 4 specification.