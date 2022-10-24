#version 450

layout(location = 0) in vec2 v_tex_coords;

layout(location = 0) out vec4 f_color;

// layout(set = 0, binding = 0) uniform texture2D t_diffuse;
// layout(set = 0, binding = 1) uniform sampler s_diffuse;

layout(set = 0, binding = 0) buffer WorldCfg {
    uint columns;
    uint rows;
} config;

layout(set = 0, binding = 1) buffer GameWorld {
    uint blocks[];
} world;

// layout(push_constant) uniform Inputs {
//     vec2 output_size;
// } inputs;

// layout(push_constant) uniform Inputs {
//     vec4 color;
//     vec2 offset;
//     vec2 view_range;
// } inputs;

uvec2 get_index_at_pixel() {
    return uvec2(0, 0);
}

void main() {
    vec2 px = gl_FragCoord.xy;

    uint column = uint((px / 800.0) * float(config.columns));
    uint row = uint((px / 600.0) * float(config.rows));

    if ((column + row) % 2 == 0) {
        f_color = vec4(1.0);
    } else {
        f_color = vec4(0.3, 0.0, 0.7, 1.0);
    }


    // f_color = texture(sampler2D(t_diffuse, s_diffuse), v_tex_coords);
}