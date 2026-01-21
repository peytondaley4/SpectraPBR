#version 450 core

in vec2 vTexCoord;
out vec4 fragColor;

uniform sampler2D uTexture;

void main() {
    // Sample the OptiX output texture
    vec4 color = texture(uTexture, vTexCoord);

    // For Phase 1, output directly (no tone mapping yet)
    // Later phases will add HDR tone mapping here
    fragColor = vec4(color.rgb, 1.0);
}
