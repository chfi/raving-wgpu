use std::collections::HashMap;
//
use raving_wgpu::camera::DynamicCamera2d;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::BufferUsages;
use winit::event::{ElementState, Event, VirtualKeyCode, WindowEvent};
use winit::event_loop::ControlFlow;

use raving_wgpu::graph::dfrog::{Graph, InputResource};
use raving_wgpu::{NodeId, State};

use anyhow::Result;

use ultraviolet::*;

#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct Vertex {
    pos: Vec4,
    tex_coord: Vec2,
}

struct CubeExample {
    graph: Graph,

    graph_scalars: rhai::Map,

    uniform_buf: wgpu::Buffer,
    vertex_buf: wgpu::Buffer,
    index_buf: wgpu::Buffer,

    draw_node: NodeId,
}

impl CubeExample {
    fn generate_matrix(aspect_ratio: f32) -> Mat4 {
        let vertical_fov = std::f32::consts::PI / 4.0;
        let projection = ultraviolet::projection::perspective_wgpu_dx(
            vertical_fov,
            aspect_ratio,
            1.0,
            10.0,
        );
        let view = Mat4::look_at(
            Vec3::new(1.5, -5.0, 3.0),
            Vec3::zero(),
            Vec3::unit_z(),
        );
        projection * view
    }

    fn init(state: &State) -> Result<Self> {
        let mut graph = Graph::new();

        let draw_schema = {
            let vert_src = include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/shaders/examples/cube.vert.spv"
            ));
            let frag_src = include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/shaders/examples/cube.frag.spv"
            ));

            graph.add_graphics_schema(
                state,
                vert_src,
                frag_src,
                ["vertex_in"],
                Some("indices"),
                &[state.surface_format],
            )?
        };

        let (vertex_data, index_data) = create_vertices();

        let vertex_buf = state.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Vertex Buffer"),
                contents: bytemuck::cast_slice(&vertex_data),
                usage: wgpu::BufferUsages::VERTEX,
            },
        );

        let index_buf = state.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Index Buffer"),
                contents: bytemuck::cast_slice(&index_data),
                usage: wgpu::BufferUsages::INDEX,
            },
        );

        let uniform_data = Self::generate_matrix(4.0 / 3.0);
        let uniform_buf = state.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Index Buffer"),
                contents: bytemuck::cast_slice(&[uniform_data]),
                usage: wgpu::BufferUsages::UNIFORM
                    | wgpu::BufferUsages::COPY_DST,
            },
        );

        let draw_node = graph.add_node(draw_schema);

        graph.add_link_from_transient("vertices", draw_node, 0);
        graph.add_link_from_transient("indices", draw_node, 1);
        graph.add_link_from_transient("image", draw_node, 2);

        let result = CubeExample {
            graph,
            graph_scalars: rhai::Map::default(),
            uniform_buf,
            vertex_buf,
            index_buf,
            draw_node,
        };

        Ok(result)
    }
}

async fn run() -> anyhow::Result<()> {
    let (event_loop, window, mut state) = raving_wgpu::initialize().await?;

    let size = window.inner_size();

    let dims = [size.width, size.height];

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
                // polyline.render(&mut state, size).unwrap();
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

fn create_vertices() -> (Vec<Vertex>, Vec<u32>) {
    fn vertex([x, y, z]: [i32; 3], [u, v]: [i32; 2]) -> Vertex {
        Vertex {
            pos: Vec4::new(x as f32, y as f32, z as f32, 1.0),
            tex_coord: Vec2::new(u as f32, v as f32),
        }
    }

    let vertex_data = [
        // top (0, 0, 1)
        vertex([-1, -1, 1], [0, 0]),
        vertex([1, -1, 1], [1, 0]),
        vertex([1, 1, 1], [1, 1]),
        vertex([-1, 1, 1], [0, 1]),
        // bottom (0, 0, -1)
        vertex([-1, 1, -1], [1, 0]),
        vertex([1, 1, -1], [0, 0]),
        vertex([1, -1, -1], [0, 1]),
        vertex([-1, -1, -1], [1, 1]),
        // right (1, 0, 0)
        vertex([1, -1, -1], [0, 0]),
        vertex([1, 1, -1], [1, 0]),
        vertex([1, 1, 1], [1, 1]),
        vertex([1, -1, 1], [0, 1]),
        // left (-1, 0, 0)
        vertex([-1, -1, 1], [1, 0]),
        vertex([-1, 1, 1], [0, 0]),
        vertex([-1, 1, -1], [0, 1]),
        vertex([-1, -1, -1], [1, 1]),
        // front (0, 1, 0)
        vertex([1, 1, -1], [1, 0]),
        vertex([-1, 1, -1], [0, 0]),
        vertex([-1, 1, 1], [0, 1]),
        vertex([1, 1, 1], [1, 1]),
        // back (0, -1, 0)
        vertex([1, -1, 1], [0, 0]),
        vertex([-1, -1, 1], [1, 0]),
        vertex([-1, -1, -1], [1, 1]),
        vertex([1, -1, -1], [0, 1]),
    ];

    let index_data: &[u32] = &[
        0, 1, 2, 2, 3, 0, // top
        4, 5, 6, 6, 7, 4, // bottom
        8, 9, 10, 10, 11, 8, // right
        12, 13, 14, 14, 15, 12, // left
        16, 17, 18, 18, 19, 16, // front
        20, 21, 22, 22, 23, 20, // back
    ];

    (vertex_data.to_vec(), index_data.to_vec())
}
