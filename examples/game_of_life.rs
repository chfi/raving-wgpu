use std::{collections::HashMap, sync::atomic::AtomicBool};

use std::sync::Arc;

use anyhow::Result;
use raving_wgpu::dfrog::{ResourceMeta, SocketMetadataSource};
use raving_wgpu::{
    shader::render::{
        FragmentShader, FragmentShaderInstance, GraphicsPipeline, VertexShader,
        VertexShaderInstance,
    },
    NodeId,
};
use raving_wgpu::{DataType, State};
use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    BufferUsages, Extent3d, ImageCopyTexture, Origin3d,
};
use winit::event::{
    ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent,
};
use winit::event_loop::ControlFlow;

use raving_wgpu::graph::dfrog::{
    Graph, InputResource, Node, NodeSchema, NodeSchemaId,
};

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
    graph: Graph,
    graph_scalars: rhai::Map,

    // compute_node: NodeId,
    view_node: NodeId,

    cfg: Config,
    blocks: Vec<u32>,

    vertex_buffer: wgpu::Buffer,
    cfg_buffer: wgpu::Buffer,
    world_buffer: wgpu::Buffer,
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

        let cfg = Config {
            columns: 128,
            rows: 64,
            
            viewport_size: [800, 600],

            view_offset: [0.0, 0.0],
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

        let world_buffer = {
            use rand::prelude::*;

            let mut rng = rand::thread_rng();

            // let mut data = vec![15u32; block_count];
            let mut data = vec![0b11111111u32; block_count];
            // rng.fill_bytes(bytemuck::cast_slice_mut(data.as_mut_slice()));
            // let data = vec![0b1001_0000_1111_1010u16; bytes / 2];
            // let data = vec![0b1111_1111_1111_1111u16; bytes / 2];

            let buffer =
                state.device.create_buffer_init(&BufferInitDescriptor {
                    label: Some("world buffer"),
                    contents: bytemuck::cast_slice(data.as_slice()),
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::UNIFORM,
                });

            buffer
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
                "/shaders/game_of_life.frag.spv"
            ));

            graph.add_graphics_schema(
                state,
                vert_src,
                frag_src,
                ["vertex_in"],
                &[state.surface_format],
            )?
        };

        let comp_src = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/shaders/game_of_life.comp.spv"
        ));

        let comp_schema =
            graph.add_custom_compute_schema(state, comp_src, |schema| {
                //
            })?;

        // 1 attachment output, 2
        let draw_n = graph.add_node(draw_view_schema);

        graph.add_link_from_transient("vertex", draw_n, 0);
        graph.add_link_from_transient("swapchain", draw_n, 1);
        graph.add_link_from_transient("cfg", draw_n, 2);
        graph.add_link_from_transient("world", draw_n, 3);

        let mut result = GameOfLife {
            graph_scalars: rhai::Map::default(),
            vertex_buffer,
            cfg_buffer,
            world_buffer,

            cfg,
            blocks,

            // world,
            graph,

            view_node: draw_n,
        };

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
                    "world".into(),
                    InputResource::Buffer {
                        size: self.len(),
                        stride: None,
                        buffer: &self.world_buffer,
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
}

async fn run() -> anyhow::Result<()> {
    let (event_loop, window, mut state) = raving_wgpu::initialize().await?;

    let size = window.inner_size();

    let dims = [size.width, size.height];
    let mut gol = GameOfLife::new(&state)?;

    let mut first_resize = true;

    event_loop.run(move |event, _, control_flow| {
        match event {
            Event::WindowEvent {
                ref event,
                window_id,
            } if window_id == window.id() => match event {
                WindowEvent::KeyboardInput {
                    input,
                    ..
                } => {
                    use VirtualKeyCode as Key;
                    if let Some(code) = input.virtual_keycode {
                        if let Key::Up = code {
                            gol.cfg.scale += 0.5;
                            // gol.cfg.scale = gol.cfg.scale.max(1.0);
                        } else if let Key::Down = code {
                            gol.cfg.scale -= 0.5;
                            gol.cfg.scale = gol.cfg.scale.max(1.0);
                        }

                    }

                }
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
                let w_size = window.inner_size();
                let size = [w_size.width, w_size.height];

                gol.render(&mut state, size).unwrap();

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
