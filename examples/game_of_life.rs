use std::collections::HashMap;

use anyhow::Result;
use raving_wgpu::input::{TouchHandler, TouchResult};
use raving_wgpu::NodeId;
use raving_wgpu::State;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use winit::event::{ElementState, Event, VirtualKeyCode, WindowEvent};
use winit::event_loop::ControlFlow;

use raving_wgpu::graph::dfrog::{Graph, InputResource};

struct ViewMachine {
    prev_x: f32,
    prev_y: f32,
    x: f32,
    y: f32,
    // vx: f32,
    // vy: f32,
    ax: f32,
    ay: f32,

    friction: f32,

    touches: TouchHandler,
    // first_touch: Option<u64>,
    // touches: FxHashMap<u64, PhysicalPosition<f64>>,
}

impl ViewMachine {
    pub fn new(friction: f32, x: f32, y: f32) -> Self {
        Self {
            prev_x: x,
            prev_y: y,
            x,
            y,
            ax: 0.0,
            ay: 0.0,

            friction,

            touches: TouchHandler::default(),
        }
    }

    pub fn handle_touch(
        &mut self,
        view_scale: f32,
        touch: &winit::event::Touch,
    ) {
        //
        let _result = self.touches.handle_touch(touch);
    }

    pub fn get(&self) -> [f32; 2] {
        [self.x, self.y]
    }

    pub fn update(&mut self, dt: f32) {
        let vx = self.x - self.prev_x;
        let vy = self.y - self.prev_y;

        self.prev_x = self.x;
        self.prev_y = self.y;

        let vx = vx * self.friction;
        let vy = vy * self.friction;

        self.x += vx + self.ax * dt * dt;
        self.y += vy + self.ay * dt * dt;

        self.ax = 0.0;
        self.ay = 0.0;
    }
}

#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct Config {
    columns: u32,
    rows: u32,
    viewport_size: [u32; 2],
    view_offset: [f32; 2],
    scale: f32,
    _pad: f32,
    // out_width: u32,
    // out_height: u32,
}

struct GameOfLife {
    view: ViewMachine,
    graph: Graph,
    graph_scalars: rhai::Map,

    running: bool,

    compute_node: NodeId,
    alt_compute_node: NodeId,
    view_node: NodeId,

    vertex_buffer: wgpu::Buffer,
    cfg_buffer: wgpu::Buffer,

    cfg: Config,
    blocks: Vec<u32>,
    iteration_step: usize,
    primary_world_buffer: wgpu::Buffer,
    secondary_world_buffer: wgpu::Buffer,
}

impl GameOfLife {
    const CFG_SIZE: usize = std::mem::size_of::<[u32; 2]>();

    const BLOCK_COLUMNS: usize = 8;
    const BLOCK_ROWS: usize = 4;

    pub fn len(&self) -> usize {
        // self.cfg.columns as usize
        // * self.cfg.rows as usize
        // *  Self::BLOCK_COLUMNS * Self::BLOCK_ROWS
        self.blocks.len() * Self::BLOCK_COLUMNS * Self::BLOCK_ROWS
    }

    fn block_index(column: u32, row: u32) -> usize {
        let column = column as usize % Self::BLOCK_COLUMNS;
        let row = row as usize % Self::BLOCK_ROWS;
        row + (column * Self::BLOCK_COLUMNS)
    }

    fn block_index_mask(column: u32, row: u32) -> u32 {
        let i = Self::block_index(column, row);
        1 << i
    }

    fn new(state: &State) -> Result<Self> {
        let mut graph = Graph::new();

        let view = ViewMachine::new(0.95, 0.0, 0.0);

        let cfg = Config {
            columns: 32 * 100,
            rows: 64 * 100,

            viewport_size: [800, 600],

            view_offset: view.get(),
            scale: 10.0,
            _pad: 0.0,
        };

        let b_rows = cfg.rows as usize / Self::BLOCK_ROWS;
        let b_cols = cfg.columns as usize / Self::BLOCK_COLUMNS;

        let c_rows = cfg.rows;
        let c_cols = cfg.columns;
        log::warn!("creating board with {c_rows} rows and {c_cols} columns");
        log::warn!("{b_rows} * {b_cols} blocks");
        let block_count = b_rows * b_cols;
        let blocks = vec![0u32; block_count];

        let cfg_buffer = {
            let buffer =
                state.device.create_buffer_init(&BufferInitDescriptor {
                    label: Some("config buffer"),
                    contents: bytemuck::cast_slice(&[cfg]),
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::UNIFORM
                        | wgpu::BufferUsages::COPY_DST,
                });

            buffer
        };

        let (primary_world_buffer, secondary_world_buffer) = {
            use rand::prelude::*;

            let mut rng = rand::thread_rng();

            // let mut data = vec![0b1111_0000_1111_1010u32; block_count];
            let mut data = vec![0u32; block_count];
            rng.fill_bytes(bytemuck::cast_slice_mut(data.as_mut_slice()));

            let primary =
                state.device.create_buffer_init(&BufferInitDescriptor {
                    label: Some("primary world buffer"),
                    contents: bytemuck::cast_slice(data.as_slice()),
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::UNIFORM,
                });

            let secondary =
                state.device.create_buffer_init(&BufferInitDescriptor {
                    label: Some("secondary work buffer"),
                    contents: bytemuck::cast_slice(data.as_slice()),
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::UNIFORM,
                });

            (primary, secondary)
        };

        let vertex_buffer = {
            let buffer =
                state.device.create_buffer_init(&BufferInitDescriptor {
                    label: Some("fullscreen vertex buffer"),
                    contents: [0u8; 4 * 3].as_slice(),
                    usage: wgpu::BufferUsages::VERTEX,
                });

            buffer
        };

        let draw_view_schema = {
            let vert_src = include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/shaders/fullscreen.vert.spv"
            ));
            let frag_src = include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/shaders/examples/game_of_life.frag.spv"
            ));

            graph.add_graphics_schema(
                state,
                vert_src,
                frag_src,
                ["vertex_in"],
                None,
                &[state.surface_format],
            )?
        };

        let comp_src = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/shaders/examples/game_of_life.comp.spv"
        ));
        let comp_schema =
            graph.add_custom_compute_schema(state, comp_src, |schema| {
                //
            })?;

        let alt_src =
            concat!(env!("CARGO_MANIFEST_DIR"), "/examples/game_of_life.wgsl");
        let alt_comp_schema =
            graph.add_compute_schema_wgsl(&state, &alt_src)?;

        let draw_n = graph.add_node(draw_view_schema);
        let comp_n = graph.add_node(comp_schema);

        let alt_comp_n = graph.add_node(alt_comp_schema);

        graph.set_node_disabled(comp_n, true);
        graph.set_node_disabled(alt_comp_n, true);

        {
            let rows = cfg.rows;
            let cols = cfg.columns;
            graph.set_node_preprocess_fn(comp_n, move |ctx, op_state| {
                let [sx, sy, _] = ctx.workgroup_size.unwrap();
                let x = cols / sx;
                let y = rows / sy;
                op_state.workgroup_count = Some([x, y, 1]);
            });

            graph.set_node_preprocess_fn(alt_comp_n, move |ctx, op_state| {
                let [sx, sy, _] = ctx.workgroup_size.unwrap();
                let x = cols / sx;
                let y = rows / sy;
                op_state.workgroup_count = Some([x, y, 1]);
            });
        }

        graph.add_link_from_transient("vertex", draw_n, 0);
        graph.add_link_from_transient("swapchain", draw_n, 1);
        graph.add_link_from_transient("cfg", draw_n, 2);

        graph.add_link_from_transient("dst_world", draw_n, 3);

        graph.add_link_from_transient("cfg", comp_n, 0);
        graph.add_link_from_transient("src_world", comp_n, 1);
        graph.add_link_from_transient("dst_world", comp_n, 2);

        graph.add_link_from_transient("cfg", alt_comp_n, 0);
        graph.add_link_from_transient("src_world", alt_comp_n, 1);
        graph.add_link_from_transient("dst_world", alt_comp_n, 2);

        let mut result = GameOfLife {
            view,
            running: false,

            graph_scalars: rhai::Map::default(),
            vertex_buffer,
            cfg_buffer,
            primary_world_buffer,
            secondary_world_buffer,

            cfg,
            blocks,

            // world,
            graph,

            compute_node: comp_n,
            alt_compute_node: alt_comp_n,
            view_node: draw_n,

            iteration_step: 0,
        };

        Ok(result)
    }

    fn update(&mut self, dt: f32) {
        let scale = self.cfg.scale;

        if let Some(touch_result) = self.view.touches.take_current_result() {
            match touch_result {
                TouchResult::Drag { pos, delta } => {
                    // let [x, y] = delta;
                    let ax = delta.x as f32 * scale;
                    let ay = delta.y as f32 * scale;

                    let a = delta.y.atan2(delta.x);

                    self.view.ax = ax;
                    self.view.ay = ay;
                }
                TouchResult::Pinch {
                    pos_0,
                    delta_0,
                    pos_1,
                    delta_1,
                } => {
                    let p0 = pos_0;
                    let p1 = pos_1;

                    let d0 = delta_0;
                    let d1 = delta_1;

                    let diff_0 = p1 - p0;
                    let dist_0 = diff_0.mag();

                    let p0_ = p0 + d0;
                    let p1_ = p1 + d1;

                    let diff_1 = p1_ - p0_;
                    let dist_1 = diff_1.mag();

                    let mid = p0 + diff_0 * 0.5;
                    let mid_ = p0_ + diff_1 * 0.5;

                    let quot = dist_1 / dist_0;

                    let new_scale = self.cfg.scale * quot as f32;

                    let delta = mid_ - mid;

                    // let mid__ = mid * quot as f32;

                    // let delta = mid__ - mid;

                    let q = quot as f32;

                    // let delta = Vec2::new(self.view.x * q, self.view.y * q);

                    self.view.x += delta.x as f32;
                    self.view.y += delta.y as f32;
                    self.view.prev_x = self.view.x;
                    self.view.prev_y = self.view.y;

                    self.cfg.scale = new_scale;

                    // self.cfg.scale = self.cfg.scale.max(1.0);

                    /*

                    let [d_x, d_y] = [x1 - x0, y1 - y0];

                    let dist_0 = (d_x * d_x + d_y * d_y).sqrt();

                    let x0_ = x0 + dx0;
                    let y0_ = y0 + dy0;
                    let x1_ = x1 + dx1;
                    let y1_ = y1 + dy1;

                    let [d_x, d_y] = [x1_ - x0_, y1_ - y0_];
                    let dist_1 = (d_x * d_x + d_y * d_y).sqrt();

                    let orig_x = self.view.x;
                    let orig_y = self.view.y;

                    self.cfg.scale *= quot as f32;
                    self.cfg.scale = self.cfg.scale.max(1.0);
                    */
                }
            }
        }

        self.view.update(dt);
        self.cfg.view_offset = self.view.get();
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
                // let w_size = window.inner_size();
                // let size = [w_size.width, w_size.height];
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
                    "cfg".into(),
                    InputResource::Buffer {
                        size: GameOfLife::CFG_SIZE,
                        stride: None,
                        buffer: &self.cfg_buffer,
                    },
                );

                transient_res.insert(
                    "vertex".into(),
                    InputResource::Buffer {
                        size: 3 * 4,
                        stride: Some(4),
                        buffer: &self.vertex_buffer,
                    },
                );

                let (src, dst) = if self.iteration_step % 2 == 0 {
                    (&self.primary_world_buffer, &self.secondary_world_buffer)
                } else {
                    (&self.secondary_world_buffer, &self.primary_world_buffer)
                };

                transient_res.insert(
                    "src_world".into(),
                    InputResource::Buffer {
                        size: self.len(),
                        stride: None,
                        buffer: src,
                    },
                );

                transient_res.insert(
                    "dst_world".into(),
                    InputResource::Buffer {
                        size: self.len(),
                        stride: None,
                        buffer: dst,
                    },
                );
            }
            self.graph.update_transient_cache(&transient_res);

            // log::warn!("validating graph");
            let valid = self
                .graph
                .validate(&transient_res, &self.graph_scalars)
                .unwrap();

            if !valid {
                log::error!("graph validation error");
            }

            if self.running {
                self.graph.set_node_disabled(self.alt_compute_node, false);
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

            if self.running {
                self.graph.set_node_disabled(self.alt_compute_node, true);
                self.running = false;
                self.iteration_step += 1;
            }
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
    let mut gol = GameOfLife::new(&state)?;

    let mut first_resize = true;

    let mut prev_frame_t = std::time::Instant::now();

    event_loop.run(move |event, _, control_flow| {
        match event {
            Event::WindowEvent {
                ref event,
                window_id,
            } if window_id == window.id() => match event {
                WindowEvent::Touch(touch) => {
                    gol.view.handle_touch(gol.cfg.scale, touch);
                    //

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

                        if let Key::Up = code {
                            gol.cfg.scale += 0.5;
                            // gol.cfg.scale = gol.cfg.scale.max(1.0);
                        } else if let Key::Down = code {
                            gol.cfg.scale -= 0.5;
                            gol.cfg.scale = gol.cfg.scale.max(1.0);
                        }

                        if let Key::Space = code {
                            if input.state == ElementState::Pressed {
                                gol.running = true;
                            }
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

                gol.render(&mut state, size).unwrap();
            }
            Event::MainEventsCleared => {
                // RedrawRequested will only trigger once, unless we manually
                // request it.

                let dt = prev_frame_t.elapsed().as_secs_f32();
                prev_frame_t = std::time::Instant::now();

                gol.update(dt);

                {
                    let w_size = window.inner_size();
                    let size = [w_size.width, w_size.height];

                    gol.cfg.viewport_size = size;

                    state.queue.write_buffer(
                        &gol.cfg_buffer,
                        0,
                        bytemuck::cast_slice(&[gol.cfg]),
                    );
                }

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
