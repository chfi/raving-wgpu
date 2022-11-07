use egui::{ClippedPrimitive, TexturesDelta};

use egui_wgpu::renderer::ScreenDescriptor;
use wgpu::CommandEncoder;
use winit::{event_loop::EventLoopWindowTarget, window::Window};

use crossbeam::atomic::AtomicCell;
use std::sync::Arc;

use crate::State;

pub type DebugWidget = Arc<
    dyn Fn(&mut egui::Ui) -> egui::InnerResponse<()> + Send + Sync + 'static,
>;

pub struct DebugWindow {
    title: String,

    slots: Vec<(String, DebugWidget)>,
}

impl DebugWindow {
    pub fn initialize(title: &str) -> Self {
        Self {
            title: title.to_string(),
            slots: Vec::new(),
        }
    }

    pub fn push_slot<F>(&mut self, name: &str, widget: F)
    where F: Fn(&mut egui::Ui) -> egui::InnerResponse<()> + Send + Sync + 'static,
    {
        let widget = Arc::new(widget) as DebugWidget;
        self.slots.push((name.to_string(), widget));
    }

    pub fn show(&self, ctx: &egui::Context) {
        egui::Window::new(&self.title).show(ctx, |ui| {
            ui.with_layout(egui::Layout::top_down(egui::Align::Center), |ui| {
                for (_name, widget) in self.slots.iter() {
                    let _resp = widget(ui);
                }
            })
        });
    }
}

pub struct EguiCtx {
    ctx: egui::Context,

    winit_state: egui_winit::State,
    renderer: egui_wgpu::Renderer,

    clipped_primitives: Vec<ClippedPrimitive>,
    textures_delta: TexturesDelta,

    load_op: wgpu::LoadOp<wgpu::Color>,
}

impl EguiCtx {
    pub fn init(
        event_loop: &EventLoopWindowTarget<()>,
        state: &State,
        clear_color: Option<wgpu::Color>,
    ) -> Self {
        let egui_ctx = egui::Context::default();

        let egui_state = egui_winit::State::new(event_loop);

        let output_color_format = state.surface_format;
        let msaa_samples = 1;

        let load_op = clear_color
            .map(wgpu::LoadOp::Clear)
            .unwrap_or(wgpu::LoadOp::Load);

        let renderer = egui_wgpu::Renderer::new(
            &state.device,
            output_color_format,
            None,
            msaa_samples,
        );

        let clipped_primitives = Vec::new();

        Self {
            ctx: egui_ctx,
            winit_state: egui_state,
            renderer,
            clipped_primitives,
            textures_delta: TexturesDelta::default(),
            load_op,
        }
    }

    pub fn run(
        &mut self,
        window: &Window,
        run_ui: impl FnOnce(&egui::Context),
    ) {
        let raw_input = self.winit_state.take_egui_input(&window);
        let full_output = self.ctx.run(raw_input, run_ui);

        self.winit_state.handle_platform_output(
            window,
            &self.ctx,
            full_output.platform_output,
        );

        let clipped = self.ctx.tessellate(full_output.shapes);
        self.clipped_primitives = clipped;
        self.textures_delta = full_output.textures_delta;
    }

    pub fn on_event(
        &mut self,
        event: &winit::event::WindowEvent<'_>,
    ) -> egui_winit::EventResponse {
        self.winit_state.on_event(&self.ctx, event)
    }

    pub fn render(
        &mut self,
        state: &State,
        render_target: &wgpu::TextureView,
        encoder: &mut CommandEncoder,
    ) {
        let screen_desc = ScreenDescriptor {
            size_in_pixels: [state.size.width, state.size.height],
            pixels_per_point: self.winit_state.pixels_per_point(),
        };

        self.renderer.update_buffers(
            &state.device,
            &state.queue,
            encoder,
            &self.clipped_primitives,
            &screen_desc,
        );

        for (id, image_delta) in &self.textures_delta.set {
            self.renderer.update_texture(
                &state.device,
                &state.queue,
                *id,
                image_delta,
            );
        }

        for id in &self.textures_delta.free {
            self.renderer.free_texture(id);
        }

        {
            let mut render_pass =
                encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    color_attachments: &[Some(
                        wgpu::RenderPassColorAttachment {
                            view: render_target,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: self.load_op,
                                store: true,
                            },
                        },
                    )],
                    depth_stencil_attachment: None,
                    label: Some("egui_render"),
                });

            self.renderer.render(
                &mut render_pass,
                &self.clipped_primitives,
                &screen_desc,
            );
        }
    }
}
