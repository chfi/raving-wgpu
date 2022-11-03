#version 450

layout (location = 0) in vec2 v_uv;

layout (location = 0) out vec4 f_color;

layout (set = 0, binding = 1, r32ui) uniform readonly uimage2D r_color;

void main() {
    ivec2 pixel = ivec2(v_uv * 256.0);
    uint tex = imageLoad(r_color, pixel).r;
    float v = float(tex.x) / 255.0;
    f_color = vec4(1.0 - (v * 5.0), 1.0 - (v * 15.0), 1.0 - (v * 50.0), 1.0);
}

