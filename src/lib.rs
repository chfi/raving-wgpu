// lib.rs

use winit::{
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::Window,
    window::WindowBuilder,
};

use anyhow::Result;

pub mod graph;
pub mod shader;
pub mod texture;

pub use graph::*;

#[derive(Debug, Clone)]
pub struct PushConstantEntry {
    pub kind: naga::ScalarKind,
    pub len: usize,
    pub range: std::ops::Range<u32>,
}

impl PushConstantEntry {
    pub fn size(&self) -> u32 {
        self.range.end - self.range.start
    }

    pub fn index_range(&self) -> std::ops::Range<usize> {
        (self.range.start as usize)..(self.range.end as usize)
    }
}

#[derive(Debug, Clone)]
pub struct PushConstants {
    buffer: Vec<u8>,
    pub stages: naga::ShaderStage,
    fields: Vec<(String, PushConstantEntry)>,
}

impl PushConstants {
    // pub fn write_field_float(
    //     &mut self,
    //     field_name: &str,
    //     data: &[f32],
    // ) -> Option<()> {
    //     todo!();
    // }

    pub fn write_field_bytes(
        &mut self,
        field_name: &str,
        data: &[u8],
    ) -> Option<()> {
        let field_ix = self.fields.iter().position(|(n, _)| n == field_name)?;
        let (_, entry) = &self.fields[field_ix];

        let size = entry.size();

        assert!(data.len() == size as usize);

        self.buffer[entry.index_range()].copy_from_slice(data);

        None
    }

    pub fn from_naga_struct(
        module: &naga::Module,
        s: &naga::TypeInner,
        stages: naga::ShaderStage,
    ) -> Result<Self> {
        if let naga::TypeInner::Struct { members, span } = s {
            let mut buffer = Vec::new();

            let mut fields = Vec::new();

            for mem in members {
                if let Some(name) = &mem.name {
                    let offset = mem.offset;

                    let ty = &module.types[mem.ty];

                    let size = ty.inner.size(&module.constants);

                    let kind = ty.inner.scalar_kind().unwrap();
                    // only supporting 32 bit values for now
                    let len = size as usize / 4;

                    let range = offset..(offset + size);

                    for _ in range.clone() {
                        buffer.push(0u8);
                    }

                    fields.push((
                        name.clone(),
                        PushConstantEntry { kind, len, range },
                    ));
                }
            }

            Ok(PushConstants {
                buffer,
                stages,
                fields,
            })
        } else {
            anyhow::bail!("Expected `TypeInner::Struct`, was: {:?}", s);
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 3],
    tex_coords: [f32; 2],
}

impl Vertex {
    fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        use std::mem;
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x2,
                },
            ],
        }
    }
}

pub async fn initialize(
) -> anyhow::Result<(winit::event_loop::EventLoop<()>, Window, State)> {
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new().build(&event_loop).unwrap();

    let state = State::new(&window).await?;

    Ok((event_loop, window, state))
}

pub async fn run() -> anyhow::Result<()> {
    env_logger::init();
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new().build(&event_loop).unwrap();

    let mut state = State::new(&window).await?;

    Ok(())
}

pub struct State {
    pub(crate) surface: wgpu::Surface,
    pub(crate) device: wgpu::Device,
    pub(crate) queue: wgpu::Queue,
    pub(crate) config: wgpu::SurfaceConfiguration,
    pub(crate) size: winit::dpi::PhysicalSize<u32>,
}

impl State {
    // Creating some of the wgpu types requires async code
    pub async fn new(window: &Window) -> Result<Self> {
        let size = window.inner_size();

        let instance = wgpu::Instance::new(wgpu::Backends::all());
        let surface = unsafe { instance.create_surface(window) };
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .ok_or(anyhow::anyhow!("Could not find compatible adapter"))?;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    features: wgpu::Features::empty(),
                    limits: if cfg!(target_arch = "wasm32") {
                        wgpu::Limits::downlevel_webgl2_defaults()
                    } else {
                        wgpu::Limits::default()
                    },
                    label: None,
                },
                None,
            )
            .await?;

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface.get_supported_formats(&adapter)[0],
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
        };

        surface.configure(&device, &config);

        Ok(State {
            surface,
            device,
            queue,
            config,
            size,
        })
    }

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
        }
    }

    pub fn input(&mut self, event: &WindowEvent) -> bool {
        false
    }

    pub fn update(&mut self) {
        todo!()
    }

    pub fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        todo!()
    }
}

/*
pub fn add(left: usize, right: usize) -> usize {
    left + right
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}
*/
