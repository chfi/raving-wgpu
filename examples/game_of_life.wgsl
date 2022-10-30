

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

fn load_block(blk_i: vec2<i32>) -> u32 {
    if blk_i.x < 0 || blk_i.y < 0
        || blk_i.x > i32(config.columns)
        || blk_i.y > i32(config.rows) {
            return 0u;
        }

    let ix = u32(blk_i.x) + u32(blk_i.y) * config.columns;
    return atomicLoad(&src[ix]);
}

fn load_blocks_horiz(blk_i: vec2<i32>) -> vec4<u32> {
    let x = vec2<i32>(1, 0);
    let a = load_block(blk_i);
    let b = load_block(blk_i + x);
    let c = load_block(blk_i + x + x);
    let d = load_block(blk_i + x + x + x);
    return vec4<u32>(a, b, c, d);
}

fn block_pos_for_cell(cell: vec2<u32>) -> vec2<u32> {
    let blk = vec2<u32>(cell.x / BLOCK_COLUMNS, cell.y / BLOCK_ROWS);
    return blk;
}

fn block_index_for_cell(cell: vec2<u32>) -> u32 {
    let block_cols = config.columns / BLOCK_COLUMNS;
    let blk = block_pos_for_cell(cell);
    return blk.x + blk.y * block_cols;
}


var<workgroup> nhood: array<array<atomic<u32>, 4>, 3>;

fn store_nhood_row(
    row: u32,
    vals: vec4<u32>,
) {
    let row = row % 3u;
    atomicStore(&nhood[row][0u], vals[0u]);
    atomicStore(&nhood[row][1u], vals[1u]);
    atomicStore(&nhood[row][2u], vals[2u]);
    atomicStore(&nhood[row][3u], vals[3u]);
}

fn get_alive_local(local_pos: vec2<u32>, offset: vec2<i32>) -> bool {
    // get the block
    // we store 12 blocks locally;
    // our coordinate space is X = [0..16], Y = [0..24]
    

    return true;
}

// `local_pos` is the local invocation ID, so in the [0..8] square
fn get_alive_neighbors(local_pos: vec2<u32>) -> u32 {
    let loc = vec2<i32>(local_pos);

    let it_l = loc + vec2(-1, -1);
    let it_c = loc + vec2(0, -1);
    let it_r = loc + vec2(1, -1);

    let im_l = loc + vec2(-1, 0);
    let im_c = loc + vec2(0, 0);
    let im_r = loc + vec2(1, 0);

    let ib_l = loc + vec2(-1, 1);
    let ib_c = loc + vec2(0, 1);
    let ib_r = loc + vec2(1, 1);

    return 0u;
}


@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>) {
    let size = config.columns * config.rows;
    let cell_ix = global_id.x + global_id.y * config.columns;

    if cell_ix > size {
        return;
    }

    let cell = global_id.xy;

    let loc = local_id;

    // let blk_loc = cell / vec2<u32>(config.columns, config.rows);

    let blk_loc = block_pos_for_cell(cell);
    let blk_ix = block_index_for_cell(cell);

    if loc.x == 0u && loc.y == 0u && loc.z == 0u {
        let blk_pos = vec2<i32>(blk_loc);

        let mid_left = blk_pos - vec2<i32>(1, 0);
        let top_left = mid_left - vec2<i32>(0, 1);
        let btm_left = mid_left + vec2<i32>(0, 1);

        let top_row = load_blocks_horiz(top_left);
        let mid_row = load_blocks_horiz(mid_left);
        let btm_row = load_blocks_horiz(btm_left);
        store_nhood_row(0u, top_row);
        store_nhood_row(1u, mid_row);
        store_nhood_row(2u, btm_row);
    }

    workgroupBarrier();

    if loc.x == 0u && loc.y == 0u && loc.z == 0u {
        let ix = 1u + loc.x;
        
        let val_0 = atomicLoad(&nhood[1u][1u]);
        let val_1 = atomicLoad(&nhood[1u][2u]);
        atomicStore(&dst[blk_ix], val_0);
        // let i = blk_ix + 
        atomicStore(&dst[blk_ix+1u], val_1);
    }

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