#version 450

layout (location = 0) in flat uint v_segment_ix;
layout (location = 1) in vec2 v_uv;

layout (location = 0) out vec4 f_color;

void main() {
    f_color = vec4(v_uv.xy, 0.2, 1.0);
}