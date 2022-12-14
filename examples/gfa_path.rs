use egui_winit::EventResponse;

use lyon::lyon_tessellation::geometry_builder::SimpleBuffersBuilder;
use lyon::lyon_tessellation::{
    BuffersBuilder, FillOptions, FillTessellator, FillVertex, StrokeOptions,
    StrokeTessellator, StrokeVertex, VertexBuffers,
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

struct GfaLayout {
    positions: Vec<Vec2>,
}

impl GfaLayout {
    fn from_layout_tsv(tsv_path: impl AsRef<std::path::Path>) -> Result<Self> {
        use std::fs::File;
        use std::io::{prelude::*, BufReader};
        let mut lines = File::open(tsv_path).map(BufReader::new)?.lines();

        let _header = lines.next();
        let mut positions = Vec::new();

        fn parse_row(line: &str) -> Option<Vec2> {
            let mut fields = line.split('\t');
            let _idx = fields.next();
            let x = fields.next()?.parse::<f32>().ok()?;
            let y = fields.next()?.parse::<f32>().ok()?;
            Some(Vec2::new(x, y))
        }

        for line in lines {
            let line = line?;
            if let Some(v) = parse_row(&line) {
                positions.push(v);
            }
        }

        Ok(GfaLayout { positions })
    }
}

struct PathRenderer {
    render_graph: Graph,
    egui: EguiCtx,

    camera: DynamicCamera2d,
    touch: TouchHandler,

    graph_scalars: rhai::Map,

    uniform_buf: wgpu::Buffer,

    annotations: Vec<(std::ops::Range<usize>, String)>,

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

fn load_layout_tsv(tsv_path: impl AsRef<std::path::Path>) -> Result<Vec<Vec2>> {
    use std::fs::File;
    use std::io::{prelude::*, BufReader};
    let mut lines = File::open(tsv_path).map(BufReader::new)?.lines();

    let _header = lines.next();
    let mut output = Vec::new();

    fn parse_row(line: &str) -> Option<Vec2> {
        let mut fields = line.split('\t');
        let _idx = fields.next();
        let x = fields.next()?.parse::<f32>().ok()?;
        let y = fields.next()?.parse::<f32>().ok()?;
        Some(Vec2::new(x, y))
    }

    for line in lines {
        let line = line?;
        if let Some(v) = parse_row(&line) {
            output.push(v);
        }
    }

    Ok(output)
}

fn parse_gfa_path(
    gfa_path: impl AsRef<std::path::Path>,
    path_name: &str,
) -> Result<Vec<(u32, bool)>> {
    use std::fs::File;
    use std::io::{prelude::*, BufReader};
    let lines = File::open(gfa_path).map(BufReader::new)?.lines();

    let mut output = Vec::new();

    for line in lines {
        let line = line?;
        if !line.starts_with("P") {
            continue;
        }

        let mut fields = line.trim().split('\t');
        let _ = fields.next();
        let name = fields.next();

        if Some(path_name) != name {
            continue;
        }

        let steps = fields.next().unwrap().split(',');

        for step in steps {
            let (seg, orient) = step.split_at(step.len() - 1);
            let seg_id = seg.parse::<u32>()?;
            let rev = orient == "-";
            output.push((seg_id, rev));
        }
    }

    Ok(output)
}

impl LyonBuffers {
    fn stroke_from_path(
        state: &State,
        points: impl IntoIterator<Item = Vec2>,
    ) -> Result<Self> {
        let mut geometry: VertexBuffers<GpuVertex, u32> = VertexBuffers::new();

        let tolerance = 10.0;

        let mut stroke_tess = StrokeTessellator::new();

        let mut builder = lyon::path::Path::builder();

        let mut points = points.into_iter();

        let first = points.next().unwrap();

        builder.begin(point(first.x, first.y));

        for p in points {
            builder.line_to(point(p.x, p.y));
        }

        builder.end(false);
        let path = builder.build();

        let opts = StrokeOptions::tolerance(tolerance).with_line_width(150.0);

        // FillOptions::tolerance(tolerance).with_fill_rule(FillRule::NonZero);

        // let mut buf_build =
        //     BuffersBuilder::new(&mut geometry, |vx: FillVertex| GpuVertex {
        //         pos: vx.position().to_array(),
        //     });

        let mut buf_build =
            BuffersBuilder::new(&mut geometry, |vx: StrokeVertex| GpuVertex {
                pos: vx.position().to_array(),
            });

        // fill_tess.tessellate_path(&path, &opts, &mut buf_build)?;
        stroke_tess.tessellate_path(&path, &opts, &mut buf_build)?;

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

impl PathRenderer {
    fn update(&mut self, window: &winit::window::Window, dt: f32) {
        let touches = self
            .touch
            .take()
            .map(TouchOutput::flip_y)
            .collect::<Vec<_>>();

        self.egui.run(window, |ctx| {
            let painter = ctx.debug_painter();

            let origin = Vec2::new(40000.0, 180000.0);
            let norm_p = self.camera.transform_world_to_screen(origin);

            let size = window.inner_size();
            let size = Vec2::new(size.width as f32, size.height as f32);
            let p = norm_p * size;

            let stroke = egui::Stroke::new(1.0, egui::Color32::WHITE);
            let p = egui::pos2(p.x, p.y);
            // dbg!(&p);
            painter.circle_stroke(p, 5.0, stroke);

            // for (range, )
        });

        if !touches.is_empty() {
            // let cam_size = self.camera.size;
            // dbg!(&cam_size);
            self.camera.stop();
        }

        self.camera.update(dt);

        match touches.len() {
            0 => {
                // drift
            }
            1 => {
                // pan
                let mut touch = touches[0];
                touch.delta *= -1.0;
                self.camera.blink(touch.delta);
            }
            n => {
                // pinch zoom (only use first two touches)
                let fst = touches[0];
                let snd = touches[1];

                let p0 = fst.pos;
                let p1 = snd.pos;

                let p0_ = fst.pos + fst.delta;
                let p1_ = snd.pos + snd.delta;

                let dist_pre = (p1 - p0).mag();
                let dist_post = (p1_ - p0_).mag();
                let del = (dist_post - dist_pre).abs();

                let cen = self.camera.center;
                let tl = cen - self.camera.size / 2.0;
                let br = cen + self.camera.size / 2.0;

                let cam_hyp = self.camera.size.dot(self.camera.size).sqrt();
                let del = del * cam_hyp;

                // if side_pre > side_post {
                if dist_pre > dist_post {
                    let tl = tl - Vec2::new(del, del);
                    let br = br + Vec2::new(del, del);
                    self.camera.fit_region_keep_aspect(tl, br);
                } else {
                    let tl = tl + Vec2::new(del, del);
                    let br = br - Vec2::new(del, del);
                    self.camera.fit_region_keep_aspect(tl, br);
                }
            }
        }
    }

    fn on_event(&mut self, window_dims: [u32; 2], event: &WindowEvent) -> bool {
        let mut consume = false;

        if self.touch.on_event(window_dims, event) {
            consume = true;
        }

        consume
    }

    fn init(
        event_loop: &EventLoopWindowTarget<()>,
        state: &State,
        points: Vec<Vec2>,
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
                wgpu::VertexStepMode::Vertex,
                ["vertex_in"],
                Some("indices"),
                &[state.surface_format],
            )?
        };

        let camera = {
            let mut min = Vec2::broadcast(f32::MAX);
            let mut max = Vec2::broadcast(f32::MIN);

            for &point in points.iter() {
                min = min.min_by_component(point);
                max = max.max_by_component(point);
            }
            let mid = min + (max - min) / 2.0;
            // dbg!(&mid);

            let center = Vec2::zero();
            let size = Vec2::new(4.0, 3.0);

            let mut camera = DynamicCamera2d::new(center, size);

            camera.fit_region_keep_aspect(min, max);
            camera
        };

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

        // let path_buffers = LyonBuffers::example2(state)?;

        let path_buffers = LyonBuffers::stroke_from_path(state, points)?;

        Ok(Self {
            render_graph: graph,
            egui,
            camera,
            touch,
            graph_scalars: rhai::Map::default(),
            uniform_buf,
            annotations: Vec::new(),
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

async fn run(points: Vec<Vec2>) -> anyhow::Result<()> {
    let (event_loop, window, mut state) = raving_wgpu::initialize().await?;

    let mut app = PathRenderer::init(&event_loop, &state, points)?;

    {
        let mk_ann = |a: usize, b: usize, s: &str| {
            (a..b, s.to_string())
        };
        let anns = [mk_ann(10_000, 15_000, "a region")];
        app.annotations.extend(anns);
    }

    let mut first_resize = true;
    let mut prev_frame_t = std::time::Instant::now();

    event_loop.run(move |event, _, control_flow| {
        match &event {
            Event::WindowEvent { window_id, event } => {
                let mut consumed = false;

                let size = window.inner_size();
                let dims = [size.width, size.height];
                consumed = app.on_event(dims, event);

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
            }

            Event::RedrawRequested(window_id) if *window_id == window.id() => {
                app.render(&mut state).unwrap();
            }
            Event::MainEventsCleared => {
                // RedrawRequested will only trigger once, unless we manually
                // request it.

                let dt = prev_frame_t.elapsed().as_secs_f32();
                prev_frame_t = std::time::Instant::now();

                app.update(&window, dt);

                window.request_redraw();
            }

            _ => {}
        }
    })
}

pub fn main() -> Result<()> {
    env_logger::builder()
        .filter_level(log::LevelFilter::Warn)
        .init();

    let args = std::env::args().collect::<Vec<_>>();

    if args.len() < 4 {
        println!("Usage: {} <gfa> <layout tsv> <path name>", &args[0]);
        std::process::exit(0);
    }

    let gfa = &args[1];
    let tsv = &args[2];
    let path = &args[3];

    // let gfa = "amy.29.gfa";
    // let path = "HG02080#1#JAHEOW010000048.1_16689583_17097287_0";
    // let tsv = "amy.29.tsv";

    let vertices = load_layout_tsv(tsv)?;

    let steps = parse_gfa_path(gfa, path).unwrap();

    println!("parsed {} steps", steps.len());

    let points = steps
        .into_iter()
        .flat_map(|(seg, rev)| {
            let ix = seg as usize - 1;
            let a = ix * 2;
            let b = a + 1;
            let va = vertices[a];
            let vb = vertices[b];
            if rev {
                [vb, va]
            } else {
                [va, vb]
            }
        })
        .collect::<Vec<_>>();

    if let Err(e) = pollster::block_on(run(points)) {
        log::error!("{:?}", e);
    }

    Ok(())
}
