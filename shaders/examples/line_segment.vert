#version 450

layout (location = 0) in vec4 a_pos;
layout (location = 1) in vec2 a_uv;

layout (location = 0) out vec2 v_uv;

layout (set = 0, binding = 0) uniform Transform {
    mat4 m;
} transform;

void main() {
    v_uv = a_uv;
    gl_Position = transform.m * a_pos;
}