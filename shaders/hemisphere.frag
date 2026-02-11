#version 450 core

in vec2 vTexCoord;
out vec4 FragColor;

uniform sampler2D uTexture;

void main() {
    // Apply circular mask with slight border
    vec2 centered = vTexCoord * 2.0 - 1.0;
    float dist = dot(centered, centered);

    if (dist > 1.0) {
        discard;
    }

    vec3 color = texture(uTexture, vTexCoord).rgb;

    // Slight border darkening near edge
    float border = smoothstep(0.85, 1.0, dist);
    color *= (1.0 - border * 0.5);

    FragColor = vec4(color, 1.0);
}
