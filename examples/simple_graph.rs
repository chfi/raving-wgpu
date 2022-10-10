use anyhow::Result;

pub async fn run() -> anyhow::Result<()> {
    let (event_loop, window, mut state) = raving_wgpu::initialize().await?;

    let size = window.inner_size();

    let dims = [size.width, size.height];

    let mut graph = raving_wgpu::graph::example_graph(&mut state, dims)?;

    Ok(())
}

pub fn main() {
    env_logger::init();
    if let Err(e) = pollster::block_on(run()) {
        log::error!("{:?}", e);
    }
}
