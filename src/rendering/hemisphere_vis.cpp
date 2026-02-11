#include "hemisphere_vis.h"
#include "shader_utils.h"
#include <cmath>
#include <iostream>

namespace spectra {

namespace {

// vMF PDF: C3(kappa) * exp(kappa * cosTheta)
float vmfPdfCpu(float kappa, float cosTheta) {
    if (kappa <= 1e-6f) return 1.0f / (4.0f * 3.14159265f);
    float sinhK = std::sinh(kappa);
    if (sinhK < 1e-10f) return 1.0f / (4.0f * 3.14159265f);
    float C3 = kappa / (4.0f * 3.14159265f * sinhK);
    return C3 * std::exp(kappa * cosTheta);
}

// Spherical to Cartesian (theta from +Y, phi around XZ)
void sphericalToCartesian(float theta, float phi, float& mx, float& my, float& mz) {
    float st = std::sin(theta);
    mx = st * std::cos(phi);
    my = std::cos(theta);
    mz = st * std::sin(phi);
}

// Cool-to-hot colormap (blue -> cyan -> green -> yellow -> red)
void coolToHot(float t, float& r, float& g, float& b) {
    t = std::fmax(0.0f, std::fmin(1.0f, t));
    if (t < 0.25f) {
        float s = t / 0.25f;
        r = 0.0f; g = s; b = 1.0f;
    } else if (t < 0.5f) {
        float s = (t - 0.25f) / 0.25f;
        r = 0.0f; g = 1.0f; b = 1.0f - s;
    } else if (t < 0.75f) {
        float s = (t - 0.5f) / 0.25f;
        r = s; g = 1.0f; b = 0.0f;
    } else {
        float s = (t - 0.75f) / 0.25f;
        r = 1.0f; g = 1.0f - s; b = 0.0f;
    }
}

// Quad vertices: position (x,y) + texcoord (u,v)
float quadVertices[] = {
    // x, y, u, v
    0.0f, 0.0f, 0.0f, 0.0f,
    1.0f, 0.0f, 1.0f, 0.0f,
    1.0f, 1.0f, 1.0f, 1.0f,
    0.0f, 0.0f, 0.0f, 0.0f,
    1.0f, 1.0f, 1.0f, 1.0f,
    0.0f, 1.0f, 0.0f, 1.0f,
};

} // anonymous namespace

HemisphereVis::~HemisphereVis() {
    shutdown();
}

bool HemisphereVis::init(const std::filesystem::path& shaderDir, uint32_t resolution) {
    shutdown();
    m_resolution = resolution;
    m_pixels.resize(resolution * resolution * 3, 0.0f);

    // Load shaders
    auto vertPath = shaderDir / "hemisphere.vert";
    auto fragPath = shaderDir / "hemisphere.frag";

    m_program = createProgramFromFiles(vertPath, fragPath);
    if (m_program == 0) {
        std::cerr << "[HemisphereVis] Failed to create shader program\n";
        return false;
    }

    m_transformLoc = glGetUniformLocation(m_program, "uTransform");
    m_textureLoc = glGetUniformLocation(m_program, "uTexture");

    // Create texture
    glGenTextures(1, &m_texture);
    glBindTexture(GL_TEXTURE_2D, m_texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, resolution, resolution, 0, GL_RGB, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glBindTexture(GL_TEXTURE_2D, 0);

    // Create VAO/VBO for quad
    glGenVertexArrays(1, &m_vao);
    glGenBuffers(1, &m_vbo);

    glBindVertexArray(m_vao);
    glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), quadVertices, GL_STATIC_DRAW);

    // Position (location 0)
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), nullptr);
    // Texcoord (location 1)
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), reinterpret_cast<void*>(2 * sizeof(float)));

    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    std::cout << "[HemisphereVis] Initialized (" << resolution << "x" << resolution << ")\n";
    return true;
}

void HemisphereVis::shutdown() {
    if (m_texture) { glDeleteTextures(1, &m_texture); m_texture = 0; }
    if (m_vbo) { glDeleteBuffers(1, &m_vbo); m_vbo = 0; }
    if (m_vao) { glDeleteVertexArrays(1, &m_vao); m_vao = 0; }
    if (m_program) { glDeleteProgram(m_program); m_program = 0; }
    m_pixels.clear();
}

void HemisphereVis::update(float theta0, float phi0, float kappa0,
                            float theta1, float phi1, float kappa1,
                            float pi0) {
    // Determine which lobes are active (kappa > threshold)
    bool lobe0Active = (kappa0 > 0.1f);
    bool lobe1Active = (kappa1 > 0.1f);

    uint32_t res = m_resolution;

    // If neither lobe is trained, show a neutral gray circle
    if (!lobe0Active && !lobe1Active) {
        for (uint32_t iy = 0; iy < res; iy++) {
            for (uint32_t ix = 0; ix < res; ix++) {
                size_t pixIdx = (iy * res + ix) * 3;
                float u = (static_cast<float>(ix) + 0.5f) / static_cast<float>(res) * 2.0f - 1.0f;
                float v = (static_cast<float>(iy) + 0.5f) / static_cast<float>(res) * 2.0f - 1.0f;
                float val = (u * u + v * v > 1.0f) ? 0.05f : 0.15f;
                m_pixels[pixIdx + 0] = val;
                m_pixels[pixIdx + 1] = val;
                m_pixels[pixIdx + 2] = val;
            }
        }
        m_dirty = true;
        return;
    }

    // Compute mu directions for active lobes
    float mu0x = 0, mu0y = 1, mu0z = 0;
    float mu1x = 0, mu1y = 1, mu1z = 0;
    if (lobe0Active) sphericalToCartesian(theta0, phi0, mu0x, mu0y, mu0z);
    if (lobe1Active) sphericalToCartesian(theta1, phi1, mu1x, mu1y, mu1z);

    // Mixture weights from fitted pi_0
    float w0, w1;
    if (lobe0Active && lobe1Active) {
        w0 = (pi0 > 0.0f && pi0 < 1.0f) ? pi0 : 0.5f;
        w1 = 1.0f - w0;
    } else {
        w0 = lobe0Active ? 1.0f : 0.0f;
        w1 = lobe1Active ? 1.0f : 0.0f;
    }

    float maxPdf = 0.0f;

    // First pass: compute PDF values and find max
    std::vector<float> pdfValues(res * res, 0.0f);
    for (uint32_t iy = 0; iy < res; iy++) {
        for (uint32_t ix = 0; ix < res; ix++) {
            float u = (static_cast<float>(ix) + 0.5f) / static_cast<float>(res) * 2.0f - 1.0f;
            float v = (static_cast<float>(iy) + 0.5f) / static_cast<float>(res) * 2.0f - 1.0f;
            float r2 = u * u + v * v;
            if (r2 > 1.0f) continue;

            // Equal-area projection to hemisphere direction
            float z = std::sqrt(1.0f - r2);
            float dx = u;
            float dy = z;  // Up = +Y hemisphere
            float dz = v;

            float pdf = 0.0f;
            if (lobe0Active) {
                float cosAngle0 = dx * mu0x + dy * mu0y + dz * mu0z;
                pdf += w0 * vmfPdfCpu(kappa0, cosAngle0);
            }
            if (lobe1Active) {
                float cosAngle1 = dx * mu1x + dy * mu1y + dz * mu1z;
                pdf += w1 * vmfPdfCpu(kappa1, cosAngle1);
            }

            pdfValues[iy * res + ix] = pdf;
            if (pdf > maxPdf) maxPdf = pdf;
        }
    }

    // Second pass: normalize and apply colormap
    if (maxPdf < 1e-8f) maxPdf = 1.0f;
    for (uint32_t iy = 0; iy < res; iy++) {
        for (uint32_t ix = 0; ix < res; ix++) {
            size_t pixIdx = (iy * res + ix) * 3;
            float u = (static_cast<float>(ix) + 0.5f) / static_cast<float>(res) * 2.0f - 1.0f;
            float v = (static_cast<float>(iy) + 0.5f) / static_cast<float>(res) * 2.0f - 1.0f;
            float r2 = u * u + v * v;

            if (r2 > 1.0f) {
                m_pixels[pixIdx + 0] = 0.05f;
                m_pixels[pixIdx + 1] = 0.05f;
                m_pixels[pixIdx + 2] = 0.05f;
            } else {
                float t = pdfValues[iy * res + ix] / maxPdf;
                coolToHot(t, m_pixels[pixIdx + 0], m_pixels[pixIdx + 1], m_pixels[pixIdx + 2]);
            }
        }
    }

    m_dirty = true;
}

void HemisphereVis::render(float screenX, float screenY, float size,
                            uint32_t viewportW, uint32_t viewportH) {
    if (!m_program || !m_texture || viewportW == 0 || viewportH == 0) return;

    // Upload texture if dirty
    if (m_dirty) {
        glBindTexture(GL_TEXTURE_2D, m_texture);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, m_resolution, m_resolution,
                        GL_RGB, GL_FLOAT, m_pixels.data());
        glBindTexture(GL_TEXTURE_2D, 0);
        m_dirty = false;
    }

    // Save GL state
    GLboolean depthTest, blend;
    glGetBooleanv(GL_DEPTH_TEST, &depthTest);
    glGetBooleanv(GL_BLEND, &blend);

    glDisable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glUseProgram(m_program);

    // Transform: pixel coords to NDC
    // NDC = (pixel / viewport) * 2 - 1
    float scaleX = 2.0f * size / static_cast<float>(viewportW);
    float scaleY = 2.0f * size / static_cast<float>(viewportH);
    float offsetX = 2.0f * screenX / static_cast<float>(viewportW) - 1.0f;
    float offsetY = 2.0f * screenY / static_cast<float>(viewportH) - 1.0f;

    // uTransform = vec4(scaleX, scaleY, offsetX, offsetY)
    if (m_transformLoc != -1) {
        glUniform4f(m_transformLoc, scaleX, scaleY, offsetX, offsetY);
    }

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, m_texture);
    if (m_textureLoc != -1) {
        glUniform1i(m_textureLoc, 0);
    }

    glBindVertexArray(m_vao);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    glBindVertexArray(0);

    glBindTexture(GL_TEXTURE_2D, 0);
    glUseProgram(0);

    // Restore state
    if (depthTest) glEnable(GL_DEPTH_TEST);
    if (!blend) glDisable(GL_BLEND);
}

} // namespace spectra
