#version 450

layout(location = 0) in vec2 v_tex_coords;

layout(location = 0) out vec4 f_color;

// layout(set = 0, binding = 0) uniform texture2D t_diffuse;
// layout(set = 0, binding = 1) uniform sampler s_diffuse;

layout(set = 0, binding = 0) buffer WorldCfg {
    uint columns;
    uint rows;

    // uvec2 view_offset;
    uvec2 view_size;
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


uvec2 get_cell_index_at_pixel(vec2 px) {
    uint col = uint((px.x / config.view_size.x)
                     * config.columns);
    uint row = uint((px.y / config.view_size.y)
                     * config.rows);
    return uvec2(col, row);
}
// uint get_local_index_at_pixel() {

// }
// uint get_index_at_pixel() {

//     col = col % 8;
//     row = row % 4;
    
//     // first, shift so the row is correct
//     uint cell = (block >> row);

//     return (cell >> col) == 1;
//     return uvec2(0, 0);
// }


bool alive(uint block, uint col, uint row) {
    // the game board is represented using 32-bit unsigned integers,
    // treating each cell as a single bit.

    // each `uint` thus corresponds to a 4 row, 8 column block of cells,
    // in row-major order (not that that really matters at this scale,
    // other than for consistency)

    // the easy way to avoid OOB
    col = col % 8;
    row = row % 4;
    
    // first, shift so the row is correct
    uint cell = (block >> row);
    return (cell >> col) == 1;
}

vec4 cell_color(uint col, uint row) {
    // col = col % config.columns;
    // row = row % config.rows;
    col = col % BLOCK_COLUMNS;
    row = row % BLOCK_ROWS;
    uint n = BLOCK_COLUMNS * BLOCK_ROWS;
    uint ix = col + row * config.columns;
    float v = (ix % n) * 1.0 / float(n);
    return vec4(v, v, v, 1.0);
}

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

void main() {
    vec2 px = gl_FragCoord.xy;

    uvec2 cell_i = get_cell_index_at_pixel(px);

    // vec4 base_color = cell_color(cell_i.x, cell_i.y);

    f_color = alive_cell(cell_i.x, cell_i.y) 
                ? vec4(1.0) : vec4(0.0, 0.0, 0.0, 1.0);
}