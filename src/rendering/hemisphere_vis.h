#pragma once

#include <glad/glad.h>
#include <filesystem>
#include <cstdint>
#include <vector>

namespace spectra {

// Renders vMF PDF visualization as OpenGL texture inset overlays.
// Shows 3 hemispheres: combined mixture, lobe 0 only, lobe 1 only.
// CPU-side evaluation of vMF over hemisphere, mapped to a cool-to-hot colormap.
class HemisphereVis {
public:
    HemisphereVis() = default;
    ~HemisphereVis();

    HemisphereVis(const HemisphereVis&) = delete;
    HemisphereVis& operator=(const HemisphereVis&) = delete;

    bool init(const std::filesystem::path& shaderDir, uint32_t resolution = 128);
    void shutdown();

    // Update the hemisphere textures with new vMF lobe parameters
    // pi0 = mixture weight for lobe 0 (lobe 1 weight = 1-pi0)
    void update(float theta0, float phi0, float kappa0,
                float theta1, float phi1, float kappa1,
                float pi0 = 0.5f);

    // Render all hemisphere insets (combined + individual lobes)
    void render(float screenX, float screenY, float size,
                uint32_t viewportW, uint32_t viewportH);

    bool isInitialized() const { return m_program != 0; }

private:
    void renderQuad(GLuint texture, float screenX, float screenY, float size,
                    uint32_t viewportW, uint32_t viewportH);

    uint32_t m_resolution = 128;
    GLuint m_program = 0;
    GLuint m_vao = 0;
    GLuint m_vbo = 0;
    GLuint m_texCombined = 0;   // Combined mixture
    GLuint m_texLobe0 = 0;      // Lobe 0 only
    GLuint m_texLobe1 = 0;      // Lobe 1 only
    GLint m_transformLoc = -1;
    GLint m_textureLoc = -1;
    std::vector<float> m_pixelsCombined;
    std::vector<float> m_pixelsLobe0;
    std::vector<float> m_pixelsLobe1;
    bool m_dirty = false;
    bool m_lobe1Active = false;  // Whether lobe 1 has data to show
};

} // namespace spectra
