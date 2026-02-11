#version 450 core

in vec2 vTexCoord;
out vec4 fragColor;

uniform sampler2D uSceneTexture;
uniform sampler2D uUITexture;
uniform int uUIEnabled;
uniform float uExposure;  // Exposure multiplier (default 1.0)

// ACES filmic tone mapping (Narkowicz 2015 fit)
// Maps HDR linear values to [0,1] with a natural film-like shoulder and toe
vec3 acesTonemap(vec3 x) {
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

// Linear to sRGB gamma encoding
vec3 linearToSrgb(vec3 linear) {
    // Exact sRGB transfer function
    vec3 lo = linear * 12.92;
    vec3 hi = 1.055 * pow(linear, vec3(1.0 / 2.4)) - 0.055;
    return mix(lo, hi, step(vec3(0.0031308), linear));
}

void main() {
    // Sample the OptiX scene texture (linear HDR)
    vec4 scene = texture(uSceneTexture, vTexCoord);

    // Apply exposure and tone mapping to scene (linear HDR -> linear LDR)
    vec3 mapped = acesTonemap(scene.rgb * uExposure);

    // Gamma encode (linear -> sRGB)
    vec3 srgb = linearToSrgb(mapped);

    // If UI is enabled, composite UI on top (UI is already in sRGB/display space)
    if (uUIEnabled != 0) {
        vec4 ui = texture(uUITexture, vTexCoord);

        // Alpha blend: result = ui * ui.a + scene * (1 - ui.a)
        vec3 blended = ui.rgb * ui.a + srgb * (1.0 - ui.a);
        fragColor = vec4(blended, 1.0);
    } else {
        fragColor = vec4(srgb, 1.0);
    }
}
