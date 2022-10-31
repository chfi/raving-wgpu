use std::collections::HashMap;

use raving_wgpu::camera::DynamicCamera2d;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::BufferUsages;
use winit::event::{ElementState, Event, VirtualKeyCode, WindowEvent};
use winit::event_loop::ControlFlow;

use raving_wgpu::graph::dfrog::{Graph, InputResource};
use raving_wgpu::{NodeId, State};

use anyhow::Result;

use ultraviolet::*;

#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct PolylineVx {
    p0: Vec2,
    p1: Vec2,
}

#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct PolylineUniform {
    line_width: f32,
    _pad: [f32; 3],
    proj: Mat4,
}

struct Polyline {
    line_width: f32,
    line: Vec<Vec2>,
    camera: DynamicCamera2d,

    draw_node: NodeId,

    uniform_buffer: wgpu::Buffer,
    vertex_buffer: Option<wgpu::Buffer>,
    vertex_count: usize,

    graph: Graph,
    graph_scalars: rhai::Map,
}

impl Polyline {
    pub fn new(state: &State) -> Result<Self> {
        let line_width = 2.0;

        let mut line = Vec::new();
        let pts = [
            [-0.5f32, -0.5],
            [0.5, -0.5],
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.4],
            [0.4, 0.5],
            [50.0, 50.0],
            [55.0, 50.0],
            [55.0, 55.0],
        ];

        for [x, y] in pts {
            line.push(Vec2::new(x, y));
        }

        let camera = DynamicCamera2d::new(
            //
            Vec2::zero(),
            //
            // Vec2::new(10.0, 10.0),
            Vec2::new(1.0, 1.0),
            // Vec2::new(0.5, 0.5),
        );

        let uniform = {
            let proj = camera.to_matrix();
            println!("{proj:#?}");
            let data = PolylineUniform {
                line_width,
                proj,
                _pad: [0.0; 3],
            };

            let usage = BufferUsages::UNIFORM | BufferUsages::COPY_DST;

            let uniform =
                state.device.create_buffer_init(&BufferInitDescriptor {
                    label: None,
                    contents: bytemuck::cast_slice(&[data]),
                    usage,
                });

            uniform
        };

        let mut graph = Graph::new();

        let draw_schema = {
            let vert_src = include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/shaders/examples/polyline.vert.spv"
            ));
            let frag_src = include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/shaders/examples/polyline.frag.spv"
            ));

            graph.add_graphics_schema(
                state,
                vert_src,
                frag_src,
                ["vertex_in"],
                &[state.surface_format],
            )?
        };

        dbg!();
        let draw_node = graph.add_node(draw_schema);

        dbg!();
        graph.add_link_from_transient("vertex", draw_node, 0);
        graph.add_link_from_transient("swapchain", draw_node, 1);

        graph.add_link_from_transient("uniform", draw_node, 2);

        dbg!();
        let mut result = Self {
            graph,
            graph_scalars: rhai::Map::default(),

            line_width,
            line,

            draw_node,

            camera,

            uniform_buffer: uniform,
            vertex_buffer: None,
            vertex_count: 0,
        };

        let (vertex, vx_count) = {
            let usage = BufferUsages::VERTEX | BufferUsages::COPY_DST;

            let vertices = result.vertex_data();
            let vx_count = vertices.len();

            let vertex =
                state.device.create_buffer_init(&BufferInitDescriptor {
                    label: None,
                    contents: bytemuck::cast_slice(vertices.as_slice()),
                    usage,
                });

            (vertex, vx_count)
        };

        result.vertex_count = vx_count;
        result.vertex_buffer = Some(vertex);

        Ok(result)
    }

    fn render(
        &mut self,
        state: &mut State,
        window_dims: [u32; 2],
    ) -> Result<()> {
        if let Ok(output) = state.surface.get_current_texture() {
            let output_view = output
                .texture
                .create_view(&wgpu::TextureViewDescriptor::default());

            let mut transient_res: HashMap<String, InputResource<'_>> =
                HashMap::default();

            {
                let size = window_dims;
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
                    "uniform".into(),
                    InputResource::Buffer {
                        size: std::mem::size_of::<PolylineUniform>(),
                        stride: None,
                        buffer: &self.uniform_buffer,
                    },
                );

                if let Some(buffer) = self.vertex_buffer.as_ref() {
                    let vx_size = std::mem::size_of::<PolylineVx>();
                    transient_res.insert(
                        "vertex".into(),
                        InputResource::Buffer {
                            size: self.vertex_count * vx_size,
                            stride: Some(vx_size),
                            buffer,
                        },
                    );
                }
            }

            self.graph.update_transient_cache(&transient_res);

            let valid = self
                .graph
                .validate(&transient_res, &self.graph_scalars)
                .unwrap();

            if !valid {
                log::error!("graph validation error");
            }

            // log::warn!("executing graph");
            let sub_index = self
                .graph
                .execute(&state, &transient_res, &self.graph_scalars)
                .unwrap();

            state
                .device
                .poll(wgpu::MaintainBase::WaitForSubmissionIndex(sub_index));

            output.present();
        } else {
            state.resize(state.size);
        }

        Ok(())
    }

    fn vertex_data(&self) -> Vec<PolylineVx> {
        let mut segs_buf = Vec::new();
        let mut segs = self.line.iter().copied();
        let mut prev = segs.next();

        for p1 in segs {
            if let Some(p0) = prev {
                let seg = PolylineVx { p0, p1 };
                segs_buf.push(seg);
                prev = Some(p1);
            }
        }

        segs_buf
    }
}

async fn run() -> anyhow::Result<()> {
    let (event_loop, window, mut state) = raving_wgpu::initialize().await?;

    let size = window.inner_size();

    let dims = [size.width, size.height];
    // let mut gol = GameOfLife::new(&state)?;

    let mut polyline = Polyline::new(&state)?;

    let mut first_resize = true;

    let mut prev_frame_t = std::time::Instant::now();

    event_loop.run(move |event, _, control_flow| {
        match event {
            Event::WindowEvent {
                ref event,
                window_id,
            } if window_id == window.id() => match event {
                WindowEvent::Touch(touch) => {
                    // if !matches!(touch.phase, TouchPhase::Moved) {
                        // println!("{touch:#?}");
                    // }
                }
                WindowEvent::TouchpadPressure { device_id, pressure, stage } => {
                    //
                    println!("TouchPadPressure pressure: {pressure:?}\tstage: {stage:?}");
                }
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
                let w_size = window.inner_size();
                let size = [w_size.width, w_size.height];
                polyline.render(&mut state, size).unwrap();
                // gol.render(&mut state, size).unwrap();
            }
            Event::MainEventsCleared => {
                // RedrawRequested will only trigger once, unless we manually
                // request it.

                let dt = prev_frame_t.elapsed().as_secs_f32();
                prev_frame_t = std::time::Instant::now();

                // polyline.update(dt);

                // gol.update(dt);

                // {
                //     let w_size = window.inner_size();
                //     let size = [w_size.width, w_size.height];

                //     gol.cfg.viewport_size = size;

                //     state.queue.write_buffer(
                //         &gol.cfg_buffer,
                //         0,
                //         bytemuck::cast_slice(&[gol.cfg]),
                //     );
                // }

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
