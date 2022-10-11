use anyhow::Result;
use raving_wgpu::NodeId;
use wgpu::{ImageCopyTexture, Origin3d, Extent3d};

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
    
    let bind_groups = if let Some(b) = graph.nodes[1].execute.as_ref() {
        let node = &graph.nodes[1];
        let bind_groups = b.create_bind_groups(node, &state, &graph.resources)?;
        println!("created {} bind groups", bind_groups.len());
        bind_groups
    } else {
        panic!("nope");
    };

    if let Some(b) = graph.nodes[1].execute.as_mut() {
        b.set_bind_groups(bind_groups);
    }

    let mut encoder = state.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Render Encoder"),
    });

    if let Some(e) = graph.nodes[1].execute.as_ref() {
        e.execute(&graph, &mut encoder)?;
    }

    let output = state.surface.get_current_texture()?;

    let src_texture = match &graph.resources[0] {
        raving_wgpu::Resource::Buffer { buffer, size, usage } => {
            unreachable!();
        }
        raving_wgpu::Resource::Texture { texture, size, format, usage } => texture,
    };

    let source = ImageCopyTexture {
        texture: &src_texture.as_ref().unwrap().texture,
        mip_level: 0,
        origin: Origin3d::ZERO,
        aspect: wgpu::TextureAspect::All,
    };

    let destination = ImageCopyTexture {
        texture: &output.texture,
        mip_level: 0,
        origin: Origin3d::ZERO,
        aspect: wgpu::TextureAspect::All,
    };

    let copy_size = Extent3d {
        width: 800,
        height: 600,
        depth_or_array_layers: 1,
    };

    // encoder.copy_texture_to_texture(source, destination, copy_size);
 
    state.queue.submit(Some(encoder.finish()));
    output.present();

    state.device.poll(wgpu::Maintain::Wait);

    std::thread::sleep(std::time::Duration::from_millis(2000));
    
    /*
    event_loop.run(move |event, _, control_flow| {
        match event {
            Event::WindowEvent {
                ref event,
                window_id,
            } if window_id == window.id() => {
                if !state.input(event) {
                    // UPDATED!
                    match event {
                        WindowEvent::CloseRequested
                        | WindowEvent::KeyboardInput {
                            input:
                                KeyboardInput {
                                    state: ElementState::Pressed,
                                    virtual_keycode: Some(VirtualKeyCode::Escape),
                                    ..
                                },
                            ..
                        } => *control_flow = ControlFlow::Exit,
                        WindowEvent::Resized(physical_size) => {
                            state.resize(*physical_size);
                        }
                        WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                            state.resize(**new_inner_size);
                        }
                        _ => {}
                    }
                }
            }
            _ => {}
        }
    });
    */

    Ok(())
}

pub fn main() {
    env_logger::init();
    log::error!("logging!");
    if let Err(e) = pollster::block_on(run()) {
        log::error!("{:?}", e);
    }
}
