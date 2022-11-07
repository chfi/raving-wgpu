use egui_winit::EventResponse;
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
struct Vertex {
    pos: Vec4,
    tex_coord: Vec2,
}

struct CubeExample {
    graph: Graph,
    egui: EguiCtx,

    camera: DynamicCamera2d,
    touch: TouchHandler,

    graph_scalars: rhai::Map,

    uniform_buf: wgpu::Buffer,
    vertex_buf: wgpu::Buffer,
    index_buf: wgpu::Buffer,

    draw_node: NodeId,
}

impl CubeExample {
    fn on_event(
        &mut self,
        window_dims: [u32; 2],
        event: &WindowEvent,
    ) -> EventResponse {
        let mut resp = self.egui.on_event(event);

        let mut consume = false;

        if !resp.consumed {
            if self.touch.on_event(window_dims, event) {
                consume = true;
            }

            if let WindowEvent::KeyboardInput { input, .. } = event {
                if let Some(key) = input.virtual_keycode {
                    use winit::event::VirtualKeyCode as Key;
                    consume = true;

                    match key {
                        Key::Up => {
                            let scale = 1.1;
                            self.camera.scale_uniformly_around(
                                Vec2::new(-0.5, -0.5),
                                scale,
                            );
                            // self.camera.nudge(Vec2::unit_y());
                        }
                        Key::Down => {
                            let scale = 1.0 / 1.1;
                            self.camera.scale_uniformly_around(
                                Vec2::new(-0.5, -0.5),
                                scale,
                            );
                            // self.camera.nudge(-Vec2::unit_y());
                        }
                        Key::Left => {
                            self.camera.nudge(Vec2::unit_x());
                        }
                        Key::Right => {
                            self.camera.nudge(-Vec2::unit_x());
                        }
                        Key::Space => {
                            self.camera.stop();
                        }
                        Key::Escape => {
                            self.camera.set_position(Vec2::zero());
                        }
                        _ => consume = false,
                    }
                }
            }
        }

        if consume {
            resp.consumed = true;
        }

        resp
    }

    fn camera_window(ctx: &egui::Context, camera: &DynamicCamera2d) {
        egui::Window::new("Camera").show(ctx, |ui| {
            let p1 = camera.center;
            let p0 = camera.prev_center;

            let v = p1 - p0;
            let a = camera.accel;

            for (u, label) in [(p1, "pos"), (v, "vel"), (a, "accel")] {
                ui.label(&format!("{}: ({}, {})", label, u.x, u.y));
            }

            let width = camera.size.x;
            let height = camera.size.y;
            ui.label(&format!("size: ({width}, {height})"));

            if height > 0f32 {
                ui.label(&format!("aspect: {:4}", width / height));
            }
        });
    }

    fn update(&mut self, window: &winit::window::Window, dt: f32) {
        let dims = window.inner_size();
        let size = Vec2::new(dims.width as f32, dims.height as f32);

        let mut touches = self
            .touch
            .take()
            .map(TouchOutput::flip_y)
            .collect::<Vec<_>>();

        self.egui.run(window, |ctx| {
            Self::camera_window(ctx, &self.camera);

            let painter = ctx.debug_painter();

            // painter.circle_stroke(
            //     egui::pos2(200.0, 200.0),
            //     10.0,
            //     egui::Stroke::new(1.0, egui::Color32::WHITE),
            // );

            let touch_scr = touches.iter().map(|t| {
                let mut p = t.pos;
                p.y = 1.0 - p.y;
                (p, t.origin)
            });

            let mut prev_pt = None;

            for (tch, orig) in touch_scr {
                let p = tch * size;
                let p = egui::pos2(p.x, p.y);
                let stroke = egui::Stroke::new(1.0, egui::Color32::WHITE);
                // dbg!(&(p.x, p.y));
                painter.circle_stroke(p, 5.0, stroke);

                {
                    let o = orig * size;
                    let o = egui::pos2(o.x, o.y);
                    let mut stroke = stroke;
                    stroke.color = egui::Color32::RED;
                    painter.line_segment([o, p], stroke);
                }

                if let Some(prev) = prev_pt {
                    painter.line_segment([prev, p], stroke);
                }
                prev_pt = Some(p);
            }

            let mut touches = touches.iter();

            let p1 = touches.next();
            let p2 = touches.next();

            if let Some(p) = p1 {
                let mut p_ = p.pos;
                p_.y = 1.0 - p_.y;

                let text = format!("{p_:?}");
                // println!("{text}");
                painter.text(
                    egui::pos2(400.0, 50.0),
                    egui::Align2::CENTER_CENTER,
                    &text,
                    egui::FontId::monospace(12.0),
                    egui::Color32::WHITE,
                );

                let p_ = p_ * size;
                let p = egui::pos2(p_.x, p_.y);
                // dbg!(&(p.x, p.y));
                painter.circle_stroke(
                    p,
                    5.0,
                    egui::Stroke::new(1.0, egui::Color32::WHITE),
                );
            }
        });

        if !touches.is_empty() {
            // if first.is_some() {
            self.camera.stop();
            // } else {
            // println!("not stopping! displacement is: {:?}", self.camera.displacement());
            // }
        }

        let mut touches = touches.into_iter();
        let first = touches.next();
        let second = touches.next();

        // as long as there's one touch, we want to apply some friction

        self.camera.update(dt);

        // applying touches after update!
        // this makes flicking work without adding additional state

        let mut remove = true;

        match (first, second) {
            (Some(mut touch), None) => {
                // flip to flick in correct direction
                touch.delta *= -1.0;
                self.camera.blink(touch.delta);
                // self.camera.nudge(touch.delta);

                if touch.delta.mag() != 0.0 {
                    println!("touch delta mag: {}", touch.delta.mag());
                }

                // let pos = (touch.pos - Vec2::new(0.5, 0.5)) * size;
                let mut pos = touch.pos * size * 0.5;
            }
            (Some(mut fst), Some(mut snd)) => {
                let cam_size = self.camera.size;

                let n_ = (snd.pos - fst.pos).normalized();

                let v = n_.dot(snd.delta);

                let d = v;

                dbg!(&d);

                // self.camera.size -= d * cam_size;

                self.camera.pinch_anchored(
                    fst.pos,
                    snd.pos,
                    snd.pos + snd.delta,
                );
            }
            _ => (), // nothing
        }
    }

    fn init(
        event_loop: &EventLoopWindowTarget<()>,
        state: &State,
    ) -> Result<Self> {
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

            let primitive = wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
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

        let camera =
            DynamicCamera2d::new(Vec2::new(0.0, 0.0), Vec2::new(4.0, 3.0));

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
        // set 0, binding 1, r_color
        // graph.add_link_from_transient("cube_texture", draw_node, 4);

        let egui = EguiCtx::init(event_loop, state, None);

        let touch = TouchHandler::default();

        let result = CubeExample {
            graph,
            graph_scalars: rhai::Map::default(),
            uniform_buf,
            vertex_buf,
            index_buf,
            draw_node,

            camera,
            touch,

            egui,
        };

        Ok(result)
    }

    fn render(&mut self, state: &mut State) -> Result<()> {
        let dims = state.size;
        let size = [dims.width, dims.height];

        let mut transient_res: HashMap<String, InputResource<'_>> =
            HashMap::default();

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

            let stride = 6 * 4;
            let v_size = 24 * stride;

            transient_res.insert(
                "vertices".into(),
                InputResource::Buffer {
                    size: v_size,
                    stride: Some(stride),
                    buffer: &self.vertex_buf,
                },
            );

            transient_res.insert(
                "indices".into(),
                InputResource::Buffer {
                    size: 36,
                    stride: Some(4),
                    buffer: &self.index_buf,
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

            // add cube texture later

            self.graph.update_transient_cache(&transient_res);

            // log::warn!("validating graph");
            let valid = self
                .graph
                .validate(&transient_res, &self.graph_scalars)
                .unwrap();

            if !valid {
                log::error!("graph validation error");
            }

            let _sub_index = self
                .graph
                .execute(&state, &transient_res, &self.graph_scalars)
                .unwrap();

            let mut encoder = state.device.create_command_encoder(
                &wgpu::CommandEncoderDescriptor {
                    label: Some("egui render"),
                },
            );

            self.egui.render(state, &output_view, &mut encoder);

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

    let size = window.inner_size();
    let dims = [size.width, size.height];

    let mut cube = CubeExample::init(&event_loop, &state)?;

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
                let egui_resp = cube.on_event(dims, event);

                if !egui_resp.consumed {
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

                                cube.camera.resize_relative(div);
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

                cube.render(&mut state).unwrap();

                // polyline.render(&mut state, size).unwrap();
                // gol.render(&mut state, size).unwrap();
            }
            Event::MainEventsCleared => {
                // RedrawRequested will only trigger once, unless we manually
                // request it.

                let dt = prev_frame_t.elapsed().as_secs_f32();
                prev_frame_t = std::time::Instant::now();

                cube.update(&window, dt);

                window.request_redraw();
            }

            _ => {}
        }
    })
}

#[cfg_attr(
    target_os = "android",
    ndk_glue::main(
        backtrace = "on",
        logger(level = "debug", tag = "hello-world")
    )
)]
pub fn main() {
    #[cfg(target_os = "android")]
    {
        use ndk::trace;
        let _trace;
        if trace::is_trace_enabled() {
            _trace = trace::Section::new("ndk-rs example main").unwrap();
        }

        log::warn!("sleeping to handle android...");
        std::thread::sleep(std::time::Duration::from_millis(1000));
        log::warn!("awake!");
    }

    #[cfg(not(target_os = "android"))]
    {
        env_logger::builder()
            .filter_level(log::LevelFilter::Warn)
            .init();
    }

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
