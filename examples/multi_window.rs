use std::collections::HashMap;

use raving_wgpu::gui::EguiCtx;
use raving_wgpu::{NewState, WindowState};
use wgpu::Maintain;
use winit::event::{ElementState, Event, VirtualKeyCode, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};

use anyhow::Result;
use winit::window::{WindowBuilder, WindowId};

struct AppWindow {
    window: winit::window::Window,
    state: WindowState,
    egui: EguiCtx,
}

impl AppWindow {
    fn init(
        title: &str,
        state: &NewState,
        event_loop: &EventLoop<()>,
    ) -> Result<Self> {
        let window =
            WindowBuilder::new().with_title(title).build(event_loop)?;

        let win_state = state.prepare_window(&window)?;

        let egui_ctx = EguiCtx::init_new(
            &state,
            win_state.surface_format,
            &event_loop,
            Some(wgpu::Color::BLACK),
        );

        Ok(Self {
            window,
            state: win_state,
            egui: egui_ctx,
        })
    }

    fn resize(&mut self, state: &NewState) {
        let size = self.window.inner_size();
        self.state.resize(&state.device, size);
    }

    fn on_event<'a>(&mut self, event: &WindowEvent<'a>) {
        let resp = self.egui.on_event(event);
    }
}

async fn run() -> Result<()> {
    let state = NewState::new().await?;

    let event_loop = EventLoop::new();

    let mut windows: HashMap<WindowId, AppWindow> = HashMap::default();

    let (wid0, wid1) = {
        let window0 = AppWindow::init("first window", &state, &event_loop)?;
        let window1 = AppWindow::init("second window", &state, &event_loop)?;

        let wid0 = window0.window.id();
        let wid1 = window1.window.id();
        windows.insert(wid0, window0);
        windows.insert(wid1, window1);

        (wid0, wid1)
    };

    // let mut appwin = AppWindow::init("first window", &state, &event_loop)?;

    let mut is_ready = false;
    let mut prev_frame_t = std::time::Instant::now();

    event_loop.run(move |event, _, control_flow| {
        if let Event::Resumed = event {
            if !is_ready {
                is_ready = true;
            }
        } else if let Event::WindowEvent { window_id, event } = event {
            let appwin = windows.get_mut(&window_id).unwrap();
            appwin.on_event(&event);
            //

            if let WindowEvent::KeyboardInput { input, .. } = event {
                use winit::event::VirtualKeyCode as Key;
                if input.state == ElementState::Pressed
                    && input.virtual_keycode == Some(Key::Escape)
                {
                    *control_flow = ControlFlow::Exit;
                }
            } else if let WindowEvent::CloseRequested = event {
                *control_flow = ControlFlow::Exit;
            } else if let WindowEvent::Resized(physical_size) = event {
                if is_ready {
                    appwin.resize(&state);
                }
            }
        } else if let Event::RedrawRequested(window_id) = event {
            state.device.poll(Maintain::Wait);

            let appwin = windows.get_mut(&window_id).unwrap();

            if let Ok(output) = appwin.state.surface.get_current_texture() {
                let output_view = output
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default());

                let mut encoder = state.device.create_command_encoder(
                    &wgpu::CommandEncoderDescriptor {
                        label: Some("egui render"),
                    },
                );

                appwin.egui.render_new(
                    &state,
                    &appwin.state,
                    &output_view,
                    &mut encoder,
                );

                state.queue.submit(Some(encoder.finish()));
                output.present();
            } else {
                appwin.resize(&state);
            }
        } else if let Event::MainEventsCleared = event {
            let _dt = prev_frame_t.elapsed().as_secs_f32();
            prev_frame_t = std::time::Instant::now();

            {
                let appwin = windows.get_mut(&wid0).unwrap();
                appwin.egui.run(&appwin.window, |ctx| {
                    egui::Window::new("hello world window").show(ctx, |ui| {
                        ui.label("hello!!!");
                        if ui.button("a button!!!").clicked() {
                            println!("clicked!!");
                        }
                    });
                });
                appwin.window.request_redraw();
            }

            {
                let appwin = windows.get_mut(&wid1).unwrap();
                appwin.egui.run(&appwin.window, |ctx| {
                    egui::Window::new("hello world window").show(ctx, |ui| {
                        ui.label("a bunch of buttons");
                        for _ix in 0..10 {
                            if ui.button("a button!!!").clicked() {
                                println!("clicked!!");
                            }
                        }
                    });
                });
                appwin.window.request_redraw();
            }
        }
    })
}

pub fn main() -> Result<()> {
    env_logger::builder()
        .filter_level(log::LevelFilter::Warn)
        .init();

    if let Err(e) = pollster::block_on(run()) {
        log::error!("{:?}", e);
    }

    Ok(())
}
