#version 450

layout (location = 0) in vec2 a_p0;
layout (location = 1) in vec2 a_p1;
// layout (location = 2) in vec2 a_len;

layout (location = 0) out uint v_segment_ix;
layout (location = 1) out vec2 v_uv;

layout (set = 0, binding = 0) uniform Uniform {
    // vec2 viewport_size;
    float line_width;
    // float _pad;
    mat4 proj;
} in_uniform;

void main() {
  uint i = gl_VertexIndex % 6;
  vec4 p0 = in_uniform.proj * vec4(a_p0, 0.0, 1.0);
  vec4 p1 = in_uniform.proj * vec4(a_p1, 0.0, 1.0);

  p0.z = 0.0;
  p1.z = 0.0;

  vec4 u = p1 - p0;
  float len = length(u);
  vec4 n_u = u/len;

  v_segment_ix = gl_InstanceIndex;

  // float w = in_uniform.line_width / 100.0;
  float w = 0.1;
  vec4 a = p0 + n_u * w;
  vec4 b = p1 + n_u * w;
  vec4 c = p1 - n_u * w;
  vec4 d = p0 - n_u * w;
  

  if (i == 0) {
    // top left
    gl_Position = a;
    v_uv = vec2(0.0);
  } else if (i == 1 || i == 4) {
    // top right
    gl_Position = b;
    v_uv = vec2(0.0, 1.0);
  } else if (i == 2 || i == 5) {
    // bottom left
    gl_Position = d;
    v_uv = vec2(0.0, 1.0);
  } else {
    // bottom right
    gl_Position = c;
    v_uv = vec2(1.0);
  }
}