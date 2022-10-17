#version 450

layout(location = 0) in vec2 v_tex_coords;

layout(location = 0) out vec4 f_color;
layout(location = 1) out vec4 f_color_2;
layout(location = 2) out uint lol;

layout(set = 0, binding = 0) uniform texture2D t_diffuse;
layout(set = 0, binding = 1) uniform sampler s_diffuse;

void main() {
    lol = 69;
    f_color_2 = vec4(1.0, 0.0, 1.0, 0.69);
    f_color = texture(sampler2D(t_diffuse, s_diffuse), v_tex_coords);
}