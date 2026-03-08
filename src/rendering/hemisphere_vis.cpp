#include "hemisphere_vis.h"
#include "shader_utils.h"
#include <cmath>
#include <iostream>

namespace spectra {

namespace {

// vMF PDF: numerically stable form kappa/(2pi) * exp(kappa*(cosTheta-1)) / (1-exp(-2*kappa))
float vmfPdfCpu(float kappa, float cosTheta) {
    if (kappa <= 1e-6f) return 1.0f / (4.0f * 3.14159265f);
    float exp_neg2k = std::exp(-2.0f * kappa);
    float denom = 1.0f - exp_neg2k;
    if (denom < 1e-10f) denom = 1.0f;
    float pdf = (kappa / (2.0f * 3.14159265f)) * std::exp(kappa * (cosTheta - 1.0f)) / denom;
    return std::max(pdf, 0.0f);
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

// Render a single-lobe vMF PDF into a pixel buffer
void renderLobePdf(std::vector<float>& pixels, uint32_t res,
                   float muX, float muY, float muZ, float kappa, bool active) {
    if (!active) {
        // Inactive lobe: dark gray circle
        for (uint32_t iy = 0; iy < res; iy++) {
            for (uint32_t ix = 0; ix < res; ix++) {
                size_t pixIdx = (iy * res + ix) * 3;
                float u = (static_cast<float>(ix) + 0.5f) / static_cast<float>(res) * 2.0f - 1.0f;
                float v = (static_cast<float>(iy) + 0.5f) / static_cast<float>(res) * 2.0f - 1.0f;
                float val = (u * u + v * v > 1.0f) ? 0.05f : 0.1f;
                pixels[pixIdx + 0] = val;
                pixels[pixIdx + 1] = val;
                pixels[pixIdx + 2] = val;
            }
        }
        return;
    }

    float maxPdf = 0.0f;
    std::vector<float> pdfValues(res * res, 0.0f);

    for (uint32_t iy = 0; iy < res; iy++) {
        for (uint32_t ix = 0; ix < res; ix++) {
            float u = (static_cast<float>(ix) + 0.5f) / static_cast<float>(res) * 2.0f - 1.0f;
            float v = (static_cast<float>(iy) + 0.5f) / static_cast<float>(res) * 2.0f - 1.0f;
            float r2 = u * u + v * v;
            if (r2 > 1.0f) continue;

            float z = std::sqrt(1.0f - r2);
            float dx = u, dy = z, dz = v;
            float cosAngle = dx * muX + dy * muY + dz * muZ;
            float pdf = vmfPdfCpu(kappa, cosAngle);

            pdfValues[iy * res + ix] = pdf;
            if (pdf > maxPdf) maxPdf = pdf;
        }
    }

    if (maxPdf < 1e-8f) maxPdf = 1.0f;
    for (uint32_t iy = 0; iy < res; iy++) {
        for (uint32_t ix = 0; ix < res; ix++) {
            size_t pixIdx = (iy * res + ix) * 3;
            float u = (static_cast<float>(ix) + 0.5f) / static_cast<float>(res) * 2.0f - 1.0f;
            float v = (static_cast<float>(iy) + 0.5f) / static_cast<float>(res) * 2.0f - 1.0f;
            if (u * u + v * v > 1.0f) {
                pixels[pixIdx + 0] = 0.05f;
                pixels[pixIdx + 1] = 0.05f;
                pixels[pixIdx + 2] = 0.05f;
            } else {
                float t = pdfValues[iy * res + ix] / maxPdf;
                coolToHot(t, pixels[pixIdx + 0], pixels[pixIdx + 1], pixels[pixIdx + 2]);
            }
        }
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
    size_t pixelCount = resolution * resolution * 3;
    m_pixelsCombined.resize(pixelCount, 0.0f);
    m_pixelsLobe0.resize(pixelCount, 0.0f);
    m_pixelsLobe1.resize(pixelCount, 0.0f);

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

    // Create 3 textures (combined, lobe0, lobe1)
    GLuint textures[3];
    glGenTextures(3, textures);
    m_texCombined = textures[0];
    m_texLobe0 = textures[1];
    m_texLobe1 = textures[2];

    for (int i = 0; i < 3; i++) {
        glBindTexture(GL_TEXTURE_2D, textures[i]);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, resolution, resolution, 0, GL_RGB, GL_FLOAT, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    }
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

    std::cout << "[HemisphereVis] Initialized (" << resolution << "x" << resolution << ", 3 textures)\n";
    return true;
}

void HemisphereVis::shutdown() {
    if (m_texCombined) { glDeleteTextures(1, &m_texCombined); m_texCombined = 0; }
    if (m_texLobe0) { glDeleteTextures(1, &m_texLobe0); m_texLobe0 = 0; }
    if (m_texLobe1) { glDeleteTextures(1, &m_texLobe1); m_texLobe1 = 0; }
    if (m_vbo) { glDeleteBuffers(1, &m_vbo); m_vbo = 0; }
    if (m_vao) { glDeleteVertexArrays(1, &m_vao); m_vao = 0; }
    if (m_program) { glDeleteProgram(m_program); m_program = 0; }
    m_pixelsCombined.clear();
    m_pixelsLobe0.clear();
    m_pixelsLobe1.clear();
}

void HemisphereVis::update(float theta0, float phi0, float kappa0,
                            float theta1, float phi1, float kappa1,
                            float pi0) {
    bool lobe0Active = (kappa0 > 0.1f);
    bool lobe1Active = (kappa1 > 0.1f);
    m_lobe1Active = lobe1Active;

    uint32_t res = m_resolution;

    // Compute mu directions for active lobes
    float mu0x = 0, mu0y = 1, mu0z = 0;
    float mu1x = 0, mu1y = 1, mu1z = 0;
    if (lobe0Active) sphericalToCartesian(theta0, phi0, mu0x, mu0y, mu0z);
    if (lobe1Active) sphericalToCartesian(theta1, phi1, mu1x, mu1y, mu1z);

    // If neither lobe is active, show neutral gray for all
    if (!lobe0Active && !lobe1Active) {
        for (uint32_t iy = 0; iy < res; iy++) {
            for (uint32_t ix = 0; ix < res; ix++) {
                size_t pixIdx = (iy * res + ix) * 3;
                float u = (static_cast<float>(ix) + 0.5f) / static_cast<float>(res) * 2.0f - 1.0f;
                float v = (static_cast<float>(iy) + 0.5f) / static_cast<float>(res) * 2.0f - 1.0f;
                float val = (u * u + v * v > 1.0f) ? 0.05f : 0.15f;
                m_pixelsCombined[pixIdx + 0] = val;
                m_pixelsCombined[pixIdx + 1] = val;
                m_pixelsCombined[pixIdx + 2] = val;
            }
        }
        renderLobePdf(m_pixelsLobe0, res, mu0x, mu0y, mu0z, kappa0, false);
        renderLobePdf(m_pixelsLobe1, res, mu1x, mu1y, mu1z, kappa1, false);
        m_dirty = true;
        return;
    }

    // Mixture weights
    float w0, w1;
    if (lobe0Active && lobe1Active) {
        w0 = (pi0 > 0.0f && pi0 < 1.0f) ? pi0 : 0.5f;
        w1 = 1.0f - w0;
    } else {
        w0 = lobe0Active ? 1.0f : 0.0f;
        w1 = lobe1Active ? 1.0f : 0.0f;
    }

    // === Combined mixture ===
    float maxPdf = 0.0f;
    std::vector<float> pdfValues(res * res, 0.0f);
    for (uint32_t iy = 0; iy < res; iy++) {
        for (uint32_t ix = 0; ix < res; ix++) {
            float u = (static_cast<float>(ix) + 0.5f) / static_cast<float>(res) * 2.0f - 1.0f;
            float v = (static_cast<float>(iy) + 0.5f) / static_cast<float>(res) * 2.0f - 1.0f;
            float r2 = u * u + v * v;
            if (r2 > 1.0f) continue;

            float z = std::sqrt(1.0f - r2);
            float dx = u, dy = z, dz = v;

            float pdf = 0.0f;
            if (lobe0Active) pdf += w0 * vmfPdfCpu(kappa0, dx*mu0x + dy*mu0y + dz*mu0z);
            if (lobe1Active) pdf += w1 * vmfPdfCpu(kappa1, dx*mu1x + dy*mu1y + dz*mu1z);

            pdfValues[iy * res + ix] = pdf;
            if (pdf > maxPdf) maxPdf = pdf;
        }
    }

    if (maxPdf < 1e-8f) maxPdf = 1.0f;
    for (uint32_t iy = 0; iy < res; iy++) {
        for (uint32_t ix = 0; ix < res; ix++) {
            size_t pixIdx = (iy * res + ix) * 3;
            float u = (static_cast<float>(ix) + 0.5f) / static_cast<float>(res) * 2.0f - 1.0f;
            float v = (static_cast<float>(iy) + 0.5f) / static_cast<float>(res) * 2.0f - 1.0f;
            if (u * u + v * v > 1.0f) {
                m_pixelsCombined[pixIdx + 0] = 0.05f;
                m_pixelsCombined[pixIdx + 1] = 0.05f;
                m_pixelsCombined[pixIdx + 2] = 0.05f;
            } else {
                float t = pdfValues[iy * res + ix] / maxPdf;
                coolToHot(t, m_pixelsCombined[pixIdx + 0], m_pixelsCombined[pixIdx + 1], m_pixelsCombined[pixIdx + 2]);
            }
        }
    }

    // === Individual lobes ===
    renderLobePdf(m_pixelsLobe0, res, mu0x, mu0y, mu0z, kappa0, lobe0Active);
    renderLobePdf(m_pixelsLobe1, res, mu1x, mu1y, mu1z, kappa1, lobe1Active);

    m_dirty = true;
}

void HemisphereVis::renderQuad(GLuint texture, float screenX, float screenY, float size,
                                uint32_t viewportW, uint32_t viewportH) {
    float scaleX = 2.0f * size / static_cast<float>(viewportW);
    float scaleY = 2.0f * size / static_cast<float>(viewportH);
    float offsetX = 2.0f * screenX / static_cast<float>(viewportW) - 1.0f;
    float offsetY = 2.0f * screenY / static_cast<float>(viewportH) - 1.0f;

    if (m_transformLoc != -1) {
        glUniform4f(m_transformLoc, scaleX, scaleY, offsetX, offsetY);
    }

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texture);
    if (m_textureLoc != -1) {
        glUniform1i(m_textureLoc, 0);
    }

    glBindVertexArray(m_vao);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    glBindVertexArray(0);
    glBindTexture(GL_TEXTURE_2D, 0);
}

void HemisphereVis::render(float screenX, float screenY, float size,
                            uint32_t viewportW, uint32_t viewportH) {
    if (!m_program || !m_texCombined || viewportW == 0 || viewportH == 0) return;

    // Upload textures if dirty
    if (m_dirty) {
        glBindTexture(GL_TEXTURE_2D, m_texCombined);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, m_resolution, m_resolution,
                        GL_RGB, GL_FLOAT, m_pixelsCombined.data());
        glBindTexture(GL_TEXTURE_2D, m_texLobe0);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, m_resolution, m_resolution,
                        GL_RGB, GL_FLOAT, m_pixelsLobe0.data());
        glBindTexture(GL_TEXTURE_2D, m_texLobe1);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, m_resolution, m_resolution,
                        GL_RGB, GL_FLOAT, m_pixelsLobe1.data());
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

    // Layout: Combined (large) on top, Lobe 0 and Lobe 1 (smaller) below side-by-side
    float smallSize = size * 0.48f;
    float gap = size * 0.04f;

    // Combined mixture (main, full size)
    renderQuad(m_texCombined, screenX, screenY, size, viewportW, viewportH);

    // Lobe 0 (bottom-left of combined, labeled by blue tint in border)
    float lobeY = screenY - smallSize - gap;
    renderQuad(m_texLobe0, screenX, lobeY, smallSize, viewportW, viewportH);

    // Lobe 1 (bottom-right of combined)
    float lobe1X = screenX + smallSize + gap;
    renderQuad(m_texLobe1, lobe1X, lobeY, smallSize, viewportW, viewportH);

    glUseProgram(0);

    // Restore state
    if (depthTest) glEnable(GL_DEPTH_TEST);
    if (!blend) glDisable(GL_BLEND);
}

} // namespace spectra
