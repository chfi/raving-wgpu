#version 450

layout(location = 0) in vec2 v_tex_coords;

layout(location = 0) out vec4 f_color;
// layout(location = 1) out vec4 f2_color;

void main() {
    f_color = vec4(v_tex_coords.x, 0.0, v_tex_coords.y, 1.0);
}