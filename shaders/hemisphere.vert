#version 450 core

layout(location = 0) in vec2 aPos;
layout(location = 1) in vec2 aTexCoord;

uniform vec4 uTransform;  // (scaleX, scaleY, offsetX, offsetY)

out vec2 vTexCoord;

void main() {
    vec2 ndc = aPos * uTransform.xy + uTransform.zw;
    gl_Position = vec4(ndc, 0.0, 1.0);
    vTexCoord = aTexCoord;
}
