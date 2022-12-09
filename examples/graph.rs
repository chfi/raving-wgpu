use egui_winit::EventResponse;
use raving_wgpu::camera::{DynamicCamera2d, TouchHandler, TouchOutput};
use raving_wgpu::gui::EguiCtx;
use std::collections::HashMap;
use wgpu::util::DeviceExt;
use winit::event::{ElementState, Event, VirtualKeyCode, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop, EventLoopWindowTarget};
use winit::window::Window;

use raving_wgpu::graph::dfrog::{Graph as RenderGraph, InputResource};
use raving_wgpu::{NodeId, State};

use anyhow::Result;

use ultraviolet::*;

struct GraphExample {
    render_graph: RenderGraph,
    egui: EguiCtx,
    camera: DynamicCamera2d,
    touch: TouchHandler,

    graph_scalars: rhai::Map,
}

impl GraphExample {
    fn init(
        event_loop: &EventLoopWindowTarget<()>,
        state: &State,
    ) -> Result<Self> {
        let mut render = RenderGraph::new();

        let draw_schema = {
            let vert_src = include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/shaders/examples/cube.vert.spv"
            ));
            let frag_src = include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/shaders/examples/cube.frag.spv"
            ));

            let primitive = wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,

                strip_index_format: None,
                unclipped_depth: false,
                conservative: false,
            };

            render.add_graphics_schema_custom(
                state,
                vert_src,
                frag_src,
                primitive,
                wgpu::VertexStepMode::Vertex,
                ["vertex_in"],
                None,
                // Some("indices"),
                &[state.surface_format],
            )?
        };

        // TODO: vertices...

        let camera = DynamicCamera2d::new(
            Vec2::new(0.0, 0.0),
            // use a 1:1 scale for the moment
            Vec2::new(state.size.width as f32, state.size.height as f32),
        );


        todo!();
    }
}

async fn run() -> Result<()> {
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
            } => {
                let size = window.inner_size();
                let dims = [size.width, size.height];

                let consumed = false;
                // let egui_resp = cube.on_event(dims, event);
                // let consumed = egui_resp.consumed;

                if !consumed {
                    match event {
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

                                let old = Vec2::new(
                                    old.width as f32,
                                    old.height as f32,
                                );
                                let new = Vec2::new(
                                    new.width as f32,
                                    new.height as f32,
                                );

                                let div = new / old;

                                // cube.camera.resize_relative(div);
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
            }

            Event::RedrawRequested(window_id) if window_id == window.id() => {
                let w_size = window.inner_size();
                let size = [w_size.width, w_size.height];

                // cube.render(&mut state).unwrap();

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
