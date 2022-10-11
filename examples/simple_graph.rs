use anyhow::Result;
use raving_wgpu::NodeId;

pub async fn run() -> anyhow::Result<()> {
    let (event_loop, window, mut state) = raving_wgpu::initialize().await?;

    let size = window.inner_size();

    let dims = [size.width, size.height];

    let mut graph = raving_wgpu::graph::example_graph(&mut state, dims)?;

    let done = graph.prepare_node(0.into())?;
    let done = graph.prepare_node(1.into())?;

    graph.allocate_node_resources(&state)?;

    log::error!("Complete!");

    println!("Node 0:\n{:#?}", graph.nodes[0]);
    println!();
    println!("------------------------");
    println!();
    println!("Node 1:\n{:#?}", graph.nodes[1]);
    println!();
    
    if let Some(b) = graph.nodes[1].bind.as_ref() {
        let node = &graph.nodes[1];
        let bind_groups = b.create_bind_groups(node, &state, &graph.resources)?;
        println!("created {} bind groups", bind_groups.len());
    }

    Ok(())
}

pub fn main() {
    env_logger::init();
    log::error!("logging!");
    if let Err(e) = pollster::block_on(run()) {
        log::error!("{:?}", e);
    }
}
