#version 450

layout(location = 0) in vec2 v_tex_coords;

layout(location = 0) out vec4 f_color;

// layout(set = 0, binding = 0) uniform texture2D t_diffuse;
// layout(set = 0, binding = 1) uniform sampler s_diffuse;

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

// layout(push_constant) uniform Inputs {
//     vec4 color;
//     vec2 offset;
//     vec2 view_range;
// } inputs;

#define BLOCK_COLUMNS 8
#define BLOCK_ROWS 4

bool alive_cell(uint col, uint row) {
    uint blk_col = col / BLOCK_COLUMNS;
    uint blk_row = row / BLOCK_ROWS;

    uint blk_cols = config.columns / BLOCK_COLUMNS;
    uint blk_rows = config.rows / BLOCK_ROWS;

    uint blk_counts = blk_cols * blk_rows;

    uint blk_index = (blk_col + blk_row * blk_cols) % blk_counts;

    uint block = world.blocks[blk_index];

    uint i_col = col % BLOCK_COLUMNS;
    uint i_row = row % BLOCK_ROWS;
    // uvec2 local_index = uvec2(col % BLOCK_COLUMNS, row % BLOCK_ROWS);
    // uint index = blk_col + blk_row * config.columns;
    // uint len = config.columns * config.rows;

    bool result = alive(block, i_col, i_row);

    return result;
}

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