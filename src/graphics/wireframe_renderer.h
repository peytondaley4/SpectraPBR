#pragma once

#include <glad/glad.h>
#include <glm/glm.hpp>
#include <filesystem>
#include <vector>
#include <cstdint>

namespace spectra {

class WireframeRenderer {
public:
    WireframeRenderer() = default;
    ~WireframeRenderer();

    WireframeRenderer(const WireframeRenderer&) = delete;
    WireframeRenderer& operator=(const WireframeRenderer&) = delete;

    // Initialize shaders and VAO
    bool init(const std::filesystem::path& shaderDir);

    // Cleanup
    void shutdown();

    // Update vertex data from edge vertices (x, y, z per vertex)
    // vertices should contain pairs of vertices for GL_LINES (2 vertices per edge)
    void updateVertices(const std::vector<float>& vertices);

    // Render the wireframe with the given view-projection matrix
    void render(const glm::mat4& viewProj, const glm::vec3& color = glm::vec3(0.2f, 0.7f, 1.0f));

    // Get vertex count
    uint32_t getVertexCount() const { return m_vertexCount; }

    bool isInitialized() const { return m_program != 0; }

private:
    GLuint m_program = 0;
    GLuint m_vao = 0;
    GLuint m_vbo = 0;
    GLint m_viewProjLoc = -1;
    GLint m_colorLoc = -1;
    uint32_t m_vertexCount = 0;
};

} // namespace spectra
