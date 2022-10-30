

struct WorldBuf {
    blocks: array<u32>,
}

struct WorldCfg {
    columns: u32,
    rows: u32,

    viewport_size: vec2<u32>,

    view_offset: vec2<f32>,

    scale: f32,
    _pad: f32,
}

let BLOCK_ROWS : u32 = 8u;
let BLOCK_COLUMNS : u32 = 4u;
 
@group(0) @binding(0) 
var<uniform> config: WorldCfg;

@group(1) @binding(0)
var<storage, read> src: array<atomic<u32>>;

@group(1) @binding(1)
var<storage, read_write> dst: array<atomic<u32>>;

var<workgroup> workcells: array<atomic<u32>, 2>;
var<workgroup> neighborhood: array<atomic<u32>, 10>;

fn load_block(blk_i: vec2<i32>) -> u32 {
    if blk_i.x < 0 || blk_i.y < 0
        || blk_i.x > i32(config.columns)
        || blk_i.y > i32(config.rows) {
            return 0u;
        }

    let ix = u32(blk_i.x) + u32(blk_i.y) * config.columns;
    return atomicLoad(&src[ix]);
}

// fn load_blocks_horiz(blk_i: vec2<i32>) -> vec4<u32> {
// }

fn block_index_for_cell(cell: vec2<u32>) -> u32 {
    let block_cols = config.columns / BLOCK_COLUMNS;
    let blk = vec2<u32>(cell.x / BLOCK_COLUMNS, cell.y / BLOCK_ROWS);
    return blk.x + blk.y * block_cols;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>) {
    let size = config.columns * config.rows;
    let ix = global_id.x + global_id.y * config.columns;

    if ix > size {
        return;
    }

    // let ix = ix % size;

    let cell = global_id.xy;

    let loc = local_id;

    // let blk_ix = block_index_for_cell(cell);

    // atomicMax(&dst[blk_ix], ix);


    if loc.y == 0u && loc.y == 0u {

        if loc.x == 0u {
            atomicStore(&dst[ix], !0u);
        } else if loc.x == 1u {
            atomicStore(&dst[ix], 1u + 3u + 5u + 7u + 9u + 11u);
        }
    }


    workgroupBarrier();

    // let lol = 5u - 1u;

    // var val = pack4x8snorm(vec4<f32>(0.1, 0.3, 0.5, 0.7));
    // val -= 1u;

    // if local_id.y > 0u {
        // return;
    // }

    // atomicStore(&dst[ix], ix);

}


// fn is_in_bounds_relative(origin: vec2<u32>, delta: vec2<i32>) -> bool {
    // return left;
// }

fn load_block_relative(origin_blk: vec2<u32>, delta: vec2<i32>) -> u32 {
    // let in_bounds = is_in_bounds_relative(origin, delta);

    return 0u;
}


fn get_alive_in_block(blck: u32, col: u32, row: u32) -> bool {
    let loc = vec2<u32>(col % BLOCK_COLUMNS, row % BLOCK_ROWS);
    let loc_i = loc.x + loc.y * BLOCK_COLUMNS;
    let cell_alive = ((blck >> loc_i) & 1u) == 1u;
    return cell_alive;
}

// fn index_in_bounds(cell: vec2<u32>) -> bool {

// }

// fn get_cell_alive(cell: vec2<u32>) -> bool {

// }