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

pub mod camera;
pub mod gui;
pub mod input;

pub mod util;

pub use graph::*;

pub async fn initialize_no_window(
) -> anyhow::Result<(winit::event_loop::EventLoop<()>, State)> {
    let event_loop = EventLoop::new();

    let state = State::new().await?;

    Ok((event_loop, state))
}

pub async fn initialize(
) -> anyhow::Result<(winit::event_loop::EventLoop<()>, State, WindowState)> {
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new().build(&event_loop).unwrap();

    let state = State::new().await?;

    let window_state = state.prepare_window(window)?;

    Ok((event_loop, state, window_state))
}

pub struct WindowState {
    pub window: winit::window::Window,
    pub surface: wgpu::Surface,
    pub config: wgpu::SurfaceConfiguration,
    pub size: winit::dpi::PhysicalSize<u32>,
    pub surface_format: wgpu::TextureFormat,
}

impl WindowState {
    pub fn resize(&mut self, device: &wgpu::Device) {
        let new_size = self.window.inner_size();
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(device, &self.config);
        }
    }
}

pub struct State {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,

    instance: wgpu::Instance,
    adapter: wgpu::Adapter,
}

impl State {
    pub fn new_window(
        &self,
        event_loop: &EventLoop<()>,
    ) -> Result<WindowState> {
        let window = WindowBuilder::new().build(event_loop).unwrap();
        self.prepare_window(window)
    }

    pub fn prepare_window(&self, window: Window) -> Result<WindowState> {
        let size = window.inner_size();

        let surface = unsafe { self.instance.create_surface(&window) }?;

        let surface_format = surface.get_capabilities(&self.adapter).formats[0];

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::COPY_DST,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            view_formats: vec![surface_format],
        };

        surface.configure(&self.device, &config);

        let window = WindowState {
            window,
            surface,
            config,
            size,
            surface_format,
        };

        Ok(window)
    }

    pub async fn new() -> Result<Self> {
        let backends = wgpu::util::backend_bits_from_env()
            .unwrap_or(wgpu::Backends::all());
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends,
            dx12_shader_compiler: Default::default(),
        });

        let adapter = wgpu::util::initialize_adapter_from_env_or_default(
            &instance, backends, None,
        )
        .await
        .ok_or(anyhow::anyhow!("Could not find compatible adapter"))?;

        let allowed_limits = adapter.limits();

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    features: wgpu::Features::empty(),
                    limits: if cfg!(target_arch = "wasm32") {
                        wgpu::Limits::downlevel_webgl2_defaults()
                    } else {
                        wgpu::Limits {
                            max_push_constant_size: allowed_limits
                                .max_push_constant_size,
                            ..wgpu::Limits::default()
                        }
                    },
                    label: None,
                },
                None,
            )
            .await?;

        Ok(Self {
            instance,
            device,
            queue,
            adapter,
        })
    }
}
