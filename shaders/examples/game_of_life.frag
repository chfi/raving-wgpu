#version 450

layout(location = 0) in vec2 v_tex_coords;

layout(location = 0) out vec4 f_color;

layout(set = 0, binding = 0) buffer WorldCfg {
    uint columns;
    uint rows;
    
    uvec2 viewport_size;

    vec2 view_offset;
    float scale;
    float _pad;
} config;

layout(set = 0, binding = 1) buffer GameWorld {
    uint blocks[];
} world;

#define BLOCK_COLUMNS 8
#define BLOCK_ROWS 4

vec2 cell_pt_at_px(vec2 px) {
    return config.view_offset + px / config.scale;
}

void main() {
    vec2 px = gl_FragCoord.xy;

    vec2 cell_pt = cell_pt_at_px(px);

    uint col = uint(floor(cell_pt.x));
    uint row = uint(floor(cell_pt.y));

    bool in_world = col < config.columns && row < config.rows;

    if (!in_world) {
        f_color = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    uint block_cols = config.columns / BLOCK_COLUMNS;
    uvec2 blk = uvec2(col / BLOCK_COLUMNS, row / BLOCK_ROWS);

    uint blk_i = blk.x + blk.y * block_cols;

    uint block = world.blocks[blk_i];

    uvec2 loc = uvec2(col % BLOCK_COLUMNS, row % BLOCK_ROWS);
    uint loc_i = loc.x + loc.y * BLOCK_COLUMNS;

    bool cell_alive = ((block >> loc_i) & 1) == 1;

    uint v_mod = 32;
    float v_div = 64.0;

    float g = float(blk_i % v_mod) / v_div;
    float r = float(col % v_mod) / v_div;
    float b = float(row % v_mod) / v_div;

    bool alive = cell_alive;

    vec4 alive_color = vec4(0.9, 0.9, 0.9, 1.0);
    vec4 bg_color = vec4(r, g, b, 1.0);

    f_color = alive ? alive_color : bg_color;
}