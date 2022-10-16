use std::sync::atomic::AtomicBool;

use std::sync::Arc;

use anyhow::Result;
use raving_wgpu::{NodeId, shader::render::VertexShader};
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

    let shader_src = include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/shaders/shader.vert.spv"
    ));

    let vx = VertexShader::from_spirv(&state, shader_src, "main")?;


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
