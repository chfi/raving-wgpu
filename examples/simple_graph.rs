use std::{collections::HashMap, sync::atomic::AtomicBool};

use std::sync::Arc;

use anyhow::Result;
use raving_wgpu::{
    shader::render::{
        FragmentShader, FragmentShaderInstance, GraphicsPipeline, VertexShader,
        VertexShaderInstance,
    },
    NodeId,
};
use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    BufferUsages, Extent3d, ImageCopyTexture, Origin3d,
};

pub async fn run_compute() -> anyhow::Result<()> {
    let (event_loop, window, mut state) = raving_wgpu::initialize().await?;

    let size = window.inner_size();

    let dims = [size.width, size.height];

    dbg!();
    let mut graph = raving_wgpu::graph::example_graph(&mut state, dims)?;

    let done = graph.prepare_node(0.into())?;
    let done = graph.prepare_node(2.into())?;
    let done = graph.prepare_node(1.into())?;

    // graph.nodes[2].

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
    state.device.poll(wgpu::Maintain::Wait);

    // use raving_wgpu::shader::interface::GroupBindings;

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

pub async fn run_old() -> anyhow::Result<()> {
    let (event_loop, window, mut state) = raving_wgpu::initialize().await?;

    dbg!();
    let size = window.inner_size();

    let dims = [size.width, size.height];

    // texture is in node 0

    dbg!();
    let vert_src = include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/shaders/shader.vert.spv"
    ));

    dbg!();
    let vert = VertexShader::from_spirv(&state, vert_src, "main")?;

    dbg!();
    let frag_src = include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        // "/shaders/shader.frag.spv"
        "/shaders/uv_rb.frag.spv"
    ));

    dbg!();
    let frag = FragmentShader::from_spirv(&state, frag_src, "main")?;

    let vert = Arc::new(vert);
    let frag = Arc::new(frag);

    // vertices are (vec3 pos, vec2 uv)

    let vert_inst = VertexShaderInstance::from_shader_single_buffer(
        &vert,
        wgpu::VertexStepMode::Vertex,
    );
    dbg!();

    let frag_inst = FragmentShaderInstance::from_shader(
        &frag,
        &[wgpu::TextureFormat::Bgra8UnormSrgb],
    )?;
    dbg!();

    let graphics = GraphicsPipeline::new(&state, vert_inst, frag_inst)?;

    fn vx(p: [f32; 3], uv: [f32; 2]) -> [u8; 5 * 4] {
        let mut buf = [0u8; 20];
        buf[0..(3 * 4)].clone_from_slice(bytemuck::cast_slice(&p));
        buf[(3 * 4)..].clone_from_slice(bytemuck::cast_slice(&uv));
        buf
    }

    let vertices = vec![
        vx([0.0, 0.0, 0.0], [0.0, 0.0]),
        vx([0.0, 1.0, 0.0], [0.0, 1.0]),
        vx([1.0, 0.0, 0.0], [1.0, 0.0]),
        vx([0.0, 1.0, 0.0], [0.0, 1.0]),
        vx([1.0, 1.0, 0.0], [1.0, 1.0]),
        vx([1.0, 0.0, 0.0], [1.0, 0.0]),
    ];

    let vertex_buffer =
        state.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("vertex buffer"),
            contents: bytemuck::cast_slice(vertices.as_slice()),
            usage: wgpu::BufferUsages::VERTEX,
        });

    let output = state.surface.get_current_texture()?;
    let view = output
        .texture
        .create_view(&wgpu::TextureViewDescriptor::default());

    let mut encoder =
        state
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

    {
        let attch = wgpu::RenderPassColorAttachment {
            view: &view,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color {
                    r: 0.0,
                    g: 0.0,
                    b: 0.0,
                    a: 1.0,
                }),
                store: true,
            },
        };

        let desc = wgpu::RenderPassDescriptor {
            label: Some("render pass"),
            color_attachments: &[Some(attch)],
            depth_stencil_attachment: None,
        };
        let mut pass = encoder.begin_render_pass(&desc);

        pass.set_pipeline(&graphics.pipeline);
        pass.set_vertex_buffer(0, vertex_buffer.slice(..));

        // pass.set_bind_group(index, bind_group, offsets)

        pass.draw(0..6, 0..1);
    }

    state.queue.submit(std::iter::once(encoder.finish()));
    output.present();

    std::thread::sleep(std::time::Duration::from_millis(2000));

    dbg!();

    Ok(())
}

pub async fn run() -> anyhow::Result<()> {
    use raving_wgpu::graph::dfrog::{
        Graph, InputResource, Node, NodeSchema, NodeSchemaId,
    };

    let (event_loop, window, mut state) = raving_wgpu::initialize().await?;

    dbg!();
    let size = window.inner_size();

    let dims = [size.width, size.height];

    let mut graph = Graph::new();

    graph.add_schemas()?;

    let img_s = NodeSchemaId(0);
    let gfx_s = NodeSchemaId(1);
    let comp_s = NodeSchemaId(2);

    let transient_res: HashMap<String, InputResource<'_>> = HashMap::default();

    let img_n = graph.add_node(img_s);
    let gfx_n = graph.add_node(gfx_s);
    let comp_n = graph.add_node(comp_s);

    graph.add_link(img_n, 0, gfx_n, 0);
    graph.add_link(gfx_n, 1, comp_n, 0);

    let mut graph_scalars = rhai::Map::default();
    graph_scalars.insert("dimensions".into(), rhai::Dynamic::from(dims));

    let valid = graph.validate(&transient_res, &graph_scalars)?;

    if valid {
        println!("validation successful");
    } else {
        log::error!("graph validation error");
    }

    Ok(())
}

pub fn main() {
    env_logger::builder()
        .filter_level(log::LevelFilter::Warn)
        .init();

    if let Err(e) = pollster::block_on(run()) {
        log::error!("{:?}", e);
    }
}
