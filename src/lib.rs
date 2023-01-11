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
    pub fn prepare_window(&self, window: Window) -> Result<WindowState> {
        let size = window.inner_size();

        let surface = unsafe { self.instance.create_surface(&window) };

        let surface_format = surface.get_supported_formats(&self.adapter)[0];

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::COPY_DST,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
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
        let instance = wgpu::Instance::new(wgpu::Backends::all());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or(anyhow::anyhow!("Could not find compatible adapter"))?;

        // let available_features = adapter.features();

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    features: wgpu::Features::empty(),
                    limits: if cfg!(target_arch = "wasm32") {
                        wgpu::Limits::downlevel_webgl2_defaults()
                    } else {
                        wgpu::Limits {
                            max_push_constant_size: 128,
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
