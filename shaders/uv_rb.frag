#version 450

layout(location = 0) in vec2 v_tex_coords;

layout(location = 0) out vec4 f_color;
// layout(location = 1) out vec4 f2_color;

layout (set = 0, binding = 0) uniform FragInput {
    vec4 color;
} frag_inputs;

// layout (push_constant) uniform Inputs {
//   vec4 color;
// } frag_inputs;

void main() {
    // vec4 base = vec4(v_tex_coords.x, 1.0, v_tex_coords.y, 1.0);
    // vec4 color = vec4
    f_color = frag_inputs.color;
    // f_color = base;
}