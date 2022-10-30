

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

@group(0) @binding(0) 
var<uniform> config: WorldCfg;

@group(1) @binding(0)
var<storage, read> src: array<atomic<u32>>;

@group(1) @binding(1)
var<storage, read_write> dst: array<atomic<u32>>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let size = config.columns * config.rows;
    let ix = global_id.x + global_id.y * config.columns;
    let ix = ix % size;
    // atomicStore(&dst[ix], atomic<u32>(0));

    atomicExchange(&dst[ix], pack4x8snorm(vec4<f32>(8.0, 1.0, 1.0, 1.0)));

}