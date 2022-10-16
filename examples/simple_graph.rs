use std::sync::atomic::AtomicBool;

use std::sync::Arc;

use anyhow::Result;
use raving_wgpu::NodeId;
use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    BufferUsages, Extent3d, ImageCopyTexture, Origin3d,
};

pub async fn run() -> anyhow::Result<()> {
    let (event_loop, window, mut state) = raving_wgpu::initialize().await?;

    let size = window.inner_size();

    let dims = [size.width, size.height];

    dbg!();
    let mut graph = raving_wgpu::graph::example_graph(&mut state, dims)?;

    let done = graph.prepare_node(0.into())?;
    let done = graph.prepare_node(2.into())?;
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
        let bind_groups =
            b.create_bind_groups(node, &state, &graph.resources)?;
        println!("created {} bind groups", bind_groups.len());
        bind_groups
    } else {
        panic!("nope");
    };

    if let Some(b) = graph.nodes[1].execute.as_mut() {
        b.set_bind_groups(bind_groups);
    }

    let mut encoder =
        state
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

    graph.execute_node(NodeId::from(1), &mut encoder)?;
    // if let Some(e) = graph.nodes[1].execute.as_ref() {
        // e.execute
        // e.execute(&graph, &mut encoder)?;
    // }

    let output = state.surface.get_current_texture()?;

    let src_texture = match &graph.resources[0] {
        raving_wgpu::Resource::Buffer {
            buffer,
            size,
            usage,
        } => {
            unreachable!();
        }
        raving_wgpu::Resource::Texture {
            texture,
            size,
            format,
            usage,
        } => texture,
    };

    /*
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
    */

    // encoder.copy_texture_to_texture(source, destination, copy_size);
    let src_buf = if let raving_wgpu::Resource::Buffer { buffer, .. } =
        &graph.resources[1]
    {
        buffer.as_ref().unwrap()
    } else {
        todo!();
    };

    let bytes = [0xF0; 512];
    let dst_buf = state.device.create_buffer_init(&BufferInitDescriptor {
        label: None,
        contents: bytes.as_slice(),
        usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
    });

    dbg!();
    encoder.copy_buffer_to_buffer(src_buf, 0, &dst_buf, 0, 512);

    dbg!();
    // let buf_dst =

    state.queue.submit(Some(encoder.finish()));

    log::error!("mapping destination buffer");

    let spinner = Arc::new(AtomicBool::new(true));

    let inner = spinner.clone();

    dst_buf
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            log::error!("mapped!");
            inner.store(false, std::sync::atomic::Ordering::SeqCst);
        });

    loop {
        state.device.poll(wgpu::Maintain::Poll);
        if !spinner.load(std::sync::atomic::Ordering::SeqCst) {
            break;
        }
    }
    let range = dst_buf.slice(..).get_mapped_range();
    let u32_range: &[u32] = bytemuck::cast_slice(&range[0..256]);
    log::error!("range[0..32] = {:?}", &range[0..32]);
    log::error!("u32_range: {:?}", u32_range);

    dbg!();
    output.present();
    dbg!();
    state.device.poll(wgpu::Maintain::Wait);

    // use raving_wgpu::shader::interface::GroupBindings;

    // std::thread::sleep(std::time::Duration::from_millis(2000));

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
