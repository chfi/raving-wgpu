use std::sync::atomic::AtomicBool;

use std::sync::Arc;

use anyhow::Result;
use raving_wgpu::{
    shader::render::{FragmentShader, VertexShader},
    NodeId,
};
use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    BufferUsages, Extent3d, ImageCopyTexture, Origin3d,
};

use datafrog::{Iteration, Relation, Variable};

pub async fn run() -> anyhow::Result<()> {
    let (event_loop, window, mut state) = raving_wgpu::initialize().await?;

    let size = window.inner_size();

    let dims = [size.width, size.height];

    let mut graph = raving_wgpu::graph::example_graph(&mut state, dims)?;

    println!(" tutturu~~");

    let vert_src = include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/shaders/shader.vert.spv"
    ));

    let vert = VertexShader::from_spirv(&state, vert_src, "main")?;

    let frag_src = include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/shaders/shader.frag.spv"
    ));

    let frag = FragmentShader::from_spirv(&state, frag_src, "main")?;

    Ok(())
}

pub fn main() {
    env_logger::builder()
        .filter_level(log::LevelFilter::Info)
        .init();

    log::error!("logging!");
    if let Err(e) = pollster::block_on(run()) {
        log::error!("{:?}", e);
    }
}
