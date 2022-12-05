use egui_winit::EventResponse;

use lyon::lyon_tessellation::geometry_builder::SimpleBuffersBuilder;
use lyon::lyon_tessellation::{
    BuffersBuilder, FillOptions, FillTessellator, FillVertex, VertexBuffers,
};
use lyon::math::point;
use lyon::path::FillRule;

use raving_wgpu::camera::{DynamicCamera2d, TouchHandler, TouchOutput};
use raving_wgpu::gui::EguiCtx;
use std::collections::HashMap;
use wgpu::util::DeviceExt;
use winit::event::{ElementState, Event, VirtualKeyCode, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop, EventLoopWindowTarget};
use winit::window::Window;

use raving_wgpu::graph::dfrog::{Graph, InputResource};
use raving_wgpu::{NodeId, State};

use anyhow::Result;

use ultraviolet::*;

#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct GpuVertex {
    pos: [f32; 2],
    // tex_coord: [f32; 2],
}

struct LyonRenderer {
    render_graph: Graph,
    egui: EguiCtx,

    camera: DynamicCamera2d,
    touch: TouchHandler,

    graph_scalars: rhai::Map,

    uniform_buf: wgpu::Buffer,
    path_buffers: Option<LyonBuffers>,
    // uniform_buf: wgpu::Buffer,
    // vertex_buf: wgpu::Buffer,
    // index_buf: wgpu::Buffer,
    draw_node: NodeId,
}

struct LyonBuffers {
    vertices: usize,
    indices: usize,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    // num_instances: u32,
}

impl LyonBuffers {
    fn example(state: &State) -> Result<Self> {
        let mut geometry: VertexBuffers<GpuVertex, u32> = VertexBuffers::new();

        let tolerance = 0.02;

        let mut fill_tess = FillTessellator::new();

        let mut builder = lyon::path::Path::builder();

        builder.begin(point(0.0, 0.0));
        builder.line_to(point(0.5, 0.5));
        builder.line_to(point(0.5, 0.75));
        builder.line_to(point(0.75, 0.75));
        builder.line_to(point(0.75, 0.5));
        builder.end(true);

        let path = builder.build();
        let opts =
            FillOptions::tolerance(tolerance).with_fill_rule(FillRule::NonZero);

        let mut buf_build =
            BuffersBuilder::new(&mut geometry, |vx: FillVertex| GpuVertex {
                pos: vx.position().to_array(),
            });

        fill_tess.tessellate_path(&path, &opts, &mut buf_build)?;

        eprintln!(
            " -- {} vertices {} indices",
            geometry.vertices.len(),
            geometry.indices.len()
        );

        let vertices = geometry.vertices.len();
        let indices = geometry.indices.len();

        let vertex_buffer = state.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Vertex Buffer"),
                contents: bytemuck::cast_slice(&geometry.vertices),
                usage: wgpu::BufferUsages::VERTEX,
            },
        );

        let index_buffer = state.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Index Buffer"),
                contents: bytemuck::cast_slice(&geometry.indices),
                usage: wgpu::BufferUsages::INDEX,
            },
        );

        // let num_instances = geometry.v

        Ok(Self {
            vertices,
            indices,
            vertex_buffer,
            index_buffer,
            //     num_instances: todo!(),
        })
    }
}

impl LyonRenderer {
    fn init(
        event_loop: &EventLoopWindowTarget<()>,
        state: &State,
    ) -> Result<Self> {
        let mut graph = Graph::new();

        let draw_schema = {
            let vert_src = include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/shaders/examples/lyon.vert.spv"
            ));
            let frag_src = include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/shaders/examples/flat.frag.spv"
            ));

            let primitive = wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Cw,
                cull_mode: None,
                // cull_mode: Some(wgpu::Face::Front),
                polygon_mode: wgpu::PolygonMode::Fill,

                strip_index_format: None,
                unclipped_depth: false,
                conservative: false,
            };

            graph.add_graphics_schema_custom(
                state,
                vert_src,
                frag_src,
                primitive,
                ["vertex_in"],
                Some("indices"),
                &[state.surface_format],
            )?
        };

        let camera =
            DynamicCamera2d::new(Vec2::new(0.0, 0.0), Vec2::new(4.0, 3.0));

        let touch = TouchHandler::default();

        let egui = EguiCtx::init(event_loop, state, None);

        let uniform_data = camera.to_matrix();

        let uniform_buf = state.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Uniform Buffer"),
                contents: bytemuck::cast_slice(&[uniform_data]),
                usage: wgpu::BufferUsages::UNIFORM
                    | wgpu::BufferUsages::COPY_DST,
            },
        );

        let draw_node = graph.add_node(draw_schema);
        
        graph.add_link_from_transient("vertices", draw_node, 0);
        graph.add_link_from_transient("indices", draw_node, 1);
        graph.add_link_from_transient("swapchain", draw_node, 2);

        // set 0, binding 0, transform matrix
        graph.add_link_from_transient("transform", draw_node, 3);

        let path_buffers = LyonBuffers::example(state)?;

        Ok(Self {
            render_graph: graph,
            egui,
            camera,
            touch,
            graph_scalars: rhai::Map::default(),
            uniform_buf,
            path_buffers: Some(path_buffers),
            draw_node,
        })
    }

    fn render(&mut self, state: &mut State) -> Result<()> {
        let dims = state.size;
        let size = [dims.width, dims.height];

        let mut transient_res: HashMap<String, InputResource<'_>> =
            HashMap::default();

        let Some(buffers) = self.path_buffers.as_ref() else {
            return Ok(());
        };

        if let Ok(output) = state.surface.get_current_texture() {
            {
                let uniform_data = self.camera.to_matrix();
                state.queue.write_buffer(
                    &self.uniform_buf,
                    0,
                    bytemuck::cast_slice(&[uniform_data]),
                );
            }

            let output_view = output
                .texture
                .create_view(&wgpu::TextureViewDescriptor::default());

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


            let stride = 8;
            let v_size = stride * buffers.vertices;
            let i_size = 4 * buffers.indices;

            transient_res.insert(
                "vertices".into(),
                InputResource::Buffer {
                    size: v_size,
                    stride: Some(stride),
                    buffer: &buffers.vertex_buffer,
                },
            );

            transient_res.insert(
                "indices".into(),
                InputResource::Buffer {
                    size: i_size,
                    stride: Some(4),
                    buffer: &buffers.index_buffer,
                },
            );

            transient_res.insert(
                "transform".into(),
                InputResource::Buffer {
                    size: 16 * 4,
                    stride: None,
                    buffer: &self.uniform_buf,
                },
            );

            self.render_graph.update_transient_cache(&transient_res);

            // log::warn!("validating graph");
            let valid = self
                .render_graph
                .validate(&transient_res, &self.graph_scalars)
                .unwrap();

            if !valid {
                log::error!("graph validation error");
            }

            let _sub_index = self
                .render_graph
                .execute(&state, &transient_res, &self.graph_scalars)
                .unwrap();

            let mut encoder = state.device.create_command_encoder(
                &wgpu::CommandEncoderDescriptor {
                    label: Some("egui render"),
                },
            );

            // self.egui.render(state, &output_view, &mut encoder);

            state.queue.submit(Some(encoder.finish()));

            // probably shouldn't be polling here, but the render graph
            // should probably not be submitting by itself, either:
            //  better to return the encoders that will be submitted
            state.device.poll(wgpu::MaintainBase::Wait);

            output.present();
        } else {
            state.resize(state.size);
        }

        Ok(())
    }
}

async fn run() -> anyhow::Result<()> {
    let (event_loop, window, mut state) = raving_wgpu::initialize().await?;

    // let size = window.inner_size();
    // let dims = [size.width, size.height];

    dbg!();
    let mut lyon = LyonRenderer::init(&event_loop, &state)?;
    dbg!();
    // let buffers = LyonBuffers::example(&mut state)?;
    dbg!();

    let mut first_resize = true;
    let mut prev_frame_t = std::time::Instant::now();

    event_loop.run(move |event, _, control_flow| {
        match &event {
            Event::WindowEvent { window_id, event } => {
                let mut consumed = false;

                if !consumed {
                    match &event {
                        WindowEvent::KeyboardInput { input, .. } => {
                            use VirtualKeyCode as Key;
                            if let Some(code) = input.virtual_keycode {
                                if let Key::Escape = code {
                                    *control_flow = ControlFlow::Exit;
                                }
                            }
                        }
                        WindowEvent::CloseRequested => {
                            *control_flow = ControlFlow::Exit
                        }
                        WindowEvent::Resized(phys_size) => {
                            // for some reason i get a validation error if i actually attempt
                            // to execute the first resize
                            if first_resize {
                                first_resize = false;
                            } else {
                                let old = state.size;
                                let new = *phys_size;

                                state.resize(*phys_size);

                                // let old = Vec2::new(
                                //     old.width as f32,
                                //     old.height as f32,
                                // );
                                // let new = Vec2::new(
                                //     new.width as f32,
                                //     new.height as f32,
                                // );

                                // let div = new / old;
                            }
                        }
                        WindowEvent::ScaleFactorChanged {
                            new_inner_size,
                            ..
                        } => {
                            state.resize(**new_inner_size);
                        }
                        _ => {}
                    }
                }
                // TODO
            }

            Event::RedrawRequested(window_id) if *window_id == window.id() => {
                let w_size = window.inner_size();
                let size = [w_size.width, w_size.height];

                lyon.render(&mut state).unwrap();

                // polyline.render(&mut state, size).unwrap();
                // gol.render(&mut state, size).unwrap();
            }
            Event::MainEventsCleared => {
                // RedrawRequested will only trigger once, unless we manually
                // request it.

                let dt = prev_frame_t.elapsed().as_secs_f32();
                prev_frame_t = std::time::Instant::now();

                // cube.update(&window, dt);

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
