#version 450 core

// Fullscreen triangle technique - no vertex buffer needed
// Generates a triangle that covers the entire screen
// Vertex 0: (-1, -1)  Vertex 1: (3, -1)  Vertex 2: (-1, 3)
// The triangle extends beyond the screen and gets clipped

out vec2 vTexCoord;

void main() {
    // Generate positions using vertex ID
    // This creates a triangle that covers the screen when clipped
    vec2 pos = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);

    // Map from [0, 2] to [-1, 1] for clip space
    gl_Position = vec4(pos * 2.0 - 1.0, 0.0, 1.0);

    // Texture coordinates: flip Y for correct orientation
    // OpenGL textures have origin at bottom-left, but our output has origin at top-left
    vTexCoord = vec2(pos.x, 1.0 - pos.y);
}
