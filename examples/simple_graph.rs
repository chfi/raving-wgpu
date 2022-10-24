use std::{collections::HashMap, sync::atomic::AtomicBool};

use std::sync::Arc;

use anyhow::Result;
use raving_wgpu::dfrog::{ResourceMeta, SocketMetadataSource};
use raving_wgpu::DataType;
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
use winit::event::{
    ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent,
};
use winit::event_loop::ControlFlow;

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

    let vert_src = include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/shaders/shader.vert.spv"
    ));
    let frag_src = include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/shaders/uv_rb.frag.spv"
    ));

    let gfx_schema = graph.add_graphics_schema(
        &state,
        vert_src,
        frag_src,
        ["vertex_in"],
        &[state.surface_format],
    )?;

    let help_s = graph.add_custom_schema(["input", "output"], |schema| {
        /*
        This adds a custom resource node, with pretty minimal boilerplate,
        that creates an image of a given format and usage, with the
        size taken from an input image.

        The input image data is never actually used, since this node
        doesn't do anything other than allocate the resource for
        its second node.
        */

        schema.source_sockets.push((1, DataType::Texture));

        use wgpu::TextureUsages as Usage;
        use SocketMetadataSource as Rule;

        schema.default_sources.insert(
            1,
            ResourceMeta::Texture {
                size: None,
                format: Some(wgpu::TextureFormat::Rgba8Unorm),
                usage: Some(Usage::STORAGE_BINDING | Usage::COPY_SRC),
            },
        );

        schema.source_rules_sockets.push((1, Rule::texture_size(0)));
    });

    let comp_src = include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/shaders/shader.comp.spv"
    ));

    let comp_s = graph.add_custom_compute_schema(&state, comp_src, |schema| {
        log::warn!("adding custom compute schema");
    });

    let img_s = NodeSchemaId(0);
    let gfx_s = NodeSchemaId(1);
    let comp_s = NodeSchemaId(2);

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

    let uniform_buffer =
        state.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("uniform buffer"),
            contents: bytemuck::cast_slice(&[1.0f32, 0.0, 0.0, 1.0]),
            usage: wgpu::BufferUsages::UNIFORM
            | wgpu::BufferUsages::COPY_DST,
        });

    log::warn!("updating transient cache");

    // let img_n1 = graph.add_node(img_s);

    let gfx_n = graph.add_node(gfx_schema);
    let gfx_n_2 = graph.add_node(gfx_schema);

    let help_n = graph.add_node(help_s);

    // graph.add_link(img_n1, 0, gfx_n, 0);
    // graph.add_link(img_n2, 0, gfx_n, 1);
    // graph.add_link(gfx_n, 1, comp_n, 0);

    graph.add_link_from_transient("swapchain", help_n, 0);

    graph.add_link_from_transient("vertices", gfx_n, 0);
    graph.add_link_from_transient("swapchain", gfx_n, 1);

    graph.add_link_from_transient("uniform", gfx_n, 2);
    graph.add_link_from_transient("uniform", gfx_n_2, 2);

    graph.add_link_from_transient("vertices", gfx_n_2, 0);
    graph.add_link(gfx_n, 1, gfx_n_2, 1);

    // graph.add_link(comp_n, 1, gfx_n, 0);

    let mut graph_scalars = rhai::Map::default();
    graph_scalars.insert("dimensions".into(), rhai::Dynamic::from(dims));

    let start_t = std::time::Instant::now();

    state.resize(winit::dpi::PhysicalSize {
        width: 800,
        height: 600,
    });
    std::thread::sleep(std::time::Duration::from_millis(50));

    let mut first_resize = true;

    event_loop.run(move |event, _, control_flow| {
        match event {
            Event::WindowEvent {
                ref event,
                window_id,
            } if window_id == window.id() => match event {
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
                    // for some reason i get a validation error if i actually attempt
                    // to execute the first resize
                    if first_resize {
                        first_resize = false;
                    } else {
                        state.resize(*physical_size);
                    }
                }
                WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                    state.resize(**new_inner_size);
                }
                _ => {}
            },
            Event::RedrawRequested(window_id) if window_id == window.id() => {
                if let Ok(output) = state.surface.get_current_texture() {
                    let output_view = output
                        .texture
                        .create_view(&wgpu::TextureViewDescriptor::default());

                    let mut transient_res: HashMap<String, InputResource<'_>> =
                        HashMap::default();

                    {
                        let w_size = window.inner_size();
                        let size = [w_size.width, w_size.height];
                        let format = state.surface_format;

                        transient_res.insert(
                            "swapchain".into(),
                            InputResource::Texture {
                                size,
                                format,
                                texture: None,
                                view: Some(&output_view),
                                sampler: None,
                            },
                        );

                        transient_res.insert(
                            "vertices".into(),
                            InputResource::Buffer {
                                size: vertex_buffer.size() as usize,
                                stride: Some(4 * 5),
                                buffer: &vertex_buffer,
                            },
                        );

                        transient_res.insert(
                            "uniform".into(),
                            InputResource::Buffer {
                                size: uniform_buffer.size() as usize,
                                stride: None,
                                buffer: &uniform_buffer,
                            },
                        );
                    }

                    {
                        if let Some(consts) =
                            graph.ops.node_op_state.get_mut(&gfx_n_2).and_then(
                                |s| {
                                    s.push_constants
                                        .get_mut(&naga::ShaderStage::Vertex)
                                },
                            )
                        {
                            let x = -0.5f32;
                            let y = -0.5;
                            consts
                                .write_field_bytes(
                                    "offset",
                                    bytemuck::cast_slice(&[x, y]),
                                )
                                .unwrap();
                        }
                    }

                    graph.update_transient_cache(&transient_res);

                    // log::warn!("validating graph");
                    let valid =
                        graph.validate(&transient_res, &graph_scalars).unwrap();

                    if !valid {
                        log::error!("graph validation error");
                    }

                    // log::warn!("executing graph");
                    let sub_index = graph
                        .execute(&state, &transient_res, &graph_scalars)
                        .unwrap();
                    state.device.poll(
                        wgpu::MaintainBase::WaitForSubmissionIndex(sub_index),
                    );

                    output.present();
                } else {
                    state.resize(state.size);
                }

                // state.update();
                /*
                match state.render() {
                    Ok(_) => {}
                    // Reconfigure the surface if it's lost or outdated
                    Err(
                        wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated,
                    ) => state.resize(state.size),
                    // The system is out of memory, we should probably quit
                    Err(wgpu::SurfaceError::OutOfMemory) => {
                        *control_flow = ControlFlow::Exit
                    }
                    // We're ignoring timeouts
                    Err(wgpu::SurfaceError::Timeout) => {
                        log::warn!("Surface timeout")
                    }
                }
                 */
            }
            Event::MainEventsCleared => {
                // RedrawRequested will only trigger once, unless we manually
                // request it.
                window.request_redraw();
            }

            _ => {}
        }
    })
}

pub fn main() {
    env_logger::builder()
        .filter_level(log::LevelFilter::Warn)
        .init();

    if let Err(e) = pollster::block_on(run()) {
        log::error!("{:?}", e);
    }
}
