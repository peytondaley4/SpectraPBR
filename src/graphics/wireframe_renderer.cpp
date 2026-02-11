#include "wireframe_renderer.h"
#include "shader_utils.h"
#include <glm/gtc/type_ptr.hpp>
#include <iostream>

namespace spectra {

WireframeRenderer::~WireframeRenderer() {
    shutdown();
}

bool WireframeRenderer::init(const std::filesystem::path& shaderDir) {
    shutdown();

    // Load shaders
    auto vertPath = shaderDir / "wireframe.vert";
    auto fragPath = shaderDir / "wireframe.frag";

    m_program = createProgramFromFiles(vertPath, fragPath);
    if (m_program == 0) {
        std::cerr << "[WireframeRenderer] Failed to create shader program\n";
        return false;
    }

    // Get uniform locations
    m_viewProjLoc = glGetUniformLocation(m_program, "uViewProj");
    m_colorLoc = glGetUniformLocation(m_program, "uColor");

    if (m_viewProjLoc == -1) {
        std::cerr << "[WireframeRenderer] Warning: uViewProj uniform not found\n";
    }
    if (m_colorLoc == -1) {
        std::cerr << "[WireframeRenderer] Warning: uColor uniform not found\n";
    }

    // Create VAO and VBO
    glGenVertexArrays(1, &m_vao);
    glGenBuffers(1, &m_vbo);

    glBindVertexArray(m_vao);
    glBindBuffer(GL_ARRAY_BUFFER, m_vbo);

    // Position attribute (location 0)
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), nullptr);

    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    std::cout << "[WireframeRenderer] Initialized successfully\n";
    return true;
}

void WireframeRenderer::shutdown() {
    if (m_vbo) {
        glDeleteBuffers(1, &m_vbo);
        m_vbo = 0;
    }
    if (m_vao) {
        glDeleteVertexArrays(1, &m_vao);
        m_vao = 0;
    }
    if (m_program) {
        glDeleteProgram(m_program);
        m_program = 0;
    }
    m_vertexCount = 0;
}

void WireframeRenderer::updateVertices(const std::vector<float>& vertices) {
    m_vertexCount = vertices.empty() ? 0 : static_cast<uint32_t>(vertices.size() / 3);

    glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
    if (vertices.empty()) {
        glBufferData(GL_ARRAY_BUFFER, 0, nullptr, GL_DYNAMIC_DRAW);
    } else {
        glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_DYNAMIC_DRAW);
    }
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void WireframeRenderer::render(const glm::mat4& viewProj, const glm::vec3& color) {
    if (m_program == 0 || m_vertexCount == 0) {
        return;
    }

    // Save state
    GLboolean depthTestEnabled;
    glGetBooleanv(GL_DEPTH_TEST, &depthTestEnabled);

    // Disable depth test for overlay
    glDisable(GL_DEPTH_TEST);

    glUseProgram(m_program);

    if (m_viewProjLoc != -1) {
        glUniformMatrix4fv(m_viewProjLoc, 1, GL_FALSE, glm::value_ptr(viewProj));
    }
    if (m_colorLoc != -1) {
        glUniform3fv(m_colorLoc, 1, glm::value_ptr(color));
    }

    glBindVertexArray(m_vao);
    glDrawArrays(GL_LINES, 0, m_vertexCount);
    glBindVertexArray(0);

    glUseProgram(0);

    // Restore state
    if (depthTestEnabled) {
        glEnable(GL_DEPTH_TEST);
    }
}

} // namespace spectra
