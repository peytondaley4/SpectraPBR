#version 450 core

in vec2 vTexCoord;
out vec4 fragColor;

uniform sampler2D uSceneTexture;
uniform sampler2D uUITexture;
uniform int uUIEnabled;

void main() {
    // Sample the OptiX scene texture
    vec4 scene = texture(uSceneTexture, vTexCoord);

    // If UI is enabled, composite UI on top
    if (uUIEnabled != 0) {
        vec4 ui = texture(uUITexture, vTexCoord);

        // Alpha blend: result = ui * ui.a + scene * (1 - ui.a)
        vec3 blended = ui.rgb * ui.a + scene.rgb * (1.0 - ui.a);
        fragColor = vec4(blended, 1.0);
    } else {
        fragColor = vec4(scene.rgb, 1.0);
    }
}
