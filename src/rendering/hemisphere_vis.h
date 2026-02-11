#pragma once

#include <glad/glad.h>
#include <filesystem>
#include <cstdint>
#include <vector>

namespace spectra {

// Renders a vMF PDF visualization as an OpenGL texture inset overlay.
// CPU-side evaluation of 2-lobe vMF over hemisphere, mapped to a cool-to-hot colormap.
class HemisphereVis {
public:
    HemisphereVis() = default;
    ~HemisphereVis();

    HemisphereVis(const HemisphereVis&) = delete;
    HemisphereVis& operator=(const HemisphereVis&) = delete;

    bool init(const std::filesystem::path& shaderDir, uint32_t resolution = 128);
    void shutdown();

    // Update the hemisphere texture with new vMF lobe parameters
    // pi0 = mixture weight for lobe 0 (lobe 1 weight = 1-pi0)
    void update(float theta0, float phi0, float kappa0,
                float theta1, float phi1, float kappa1,
                float pi0 = 0.5f);

    // Render the hemisphere inset at the given screen position
    void render(float screenX, float screenY, float size,
                uint32_t viewportW, uint32_t viewportH);

    bool isInitialized() const { return m_program != 0; }

private:
    uint32_t m_resolution = 128;
    GLuint m_program = 0;
    GLuint m_vao = 0;
    GLuint m_vbo = 0;
    GLuint m_texture = 0;
    GLint m_transformLoc = -1;
    GLint m_textureLoc = -1;
    std::vector<float> m_pixels;  // RGB float per pixel
    bool m_dirty = false;
};

} // namespace spectra
