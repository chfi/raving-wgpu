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
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    num_instances: u32,
}

impl LyonBuffers {
    fn example(state: &mut State) -> Result<Self> {
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

        println!(
            " -- {} vertices {} indices",
            geometry.vertices.len(),
            geometry.indices.len()
        );

        std::process::exit(1);

        // Ok(Self {
        //     vertex_buffer: todo!(),
        //     index_buffer: todo!(),
        //     num_instances: todo!(),
        // })

        todo!();
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

        Ok(Self {
            render_graph: graph,
            egui,
            camera,
            touch,
            graph_scalars: rhai::Map::default(),
            uniform_buf,
            path_buffers: None,
            draw_node,
        })
    }
}

async fn run() -> anyhow::Result<()> {
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
