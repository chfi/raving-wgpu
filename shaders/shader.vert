#version 450

layout(location=0) in vec3 a_position;
layout(location=1) in vec2 a_tex_coords;

layout(location=0) out vec2 v_tex_coords;

layout (push_constant) uniform Inputs {
  vec4 offset;
//   vec2 offset;
//   vec4 color;
} vert_inputs;

void main() {
    v_tex_coords = a_tex_coords;
    vec3 pos = vec3(vert_inputs.offset.xy + a_position.xy, a_position.z);
    gl_Position = vec4(pos, 1.0);
    // gl_Position = vec4(a_position, 1.0);
}