use raving_wgpu::gui::EguiCtx;
use wgpu::Maintain;
use winit::event::{ElementState, Event, VirtualKeyCode, WindowEvent};
use winit::event_loop::ControlFlow;

use anyhow::Result;

async fn run() -> Result<()> {
    let (event_loop, window, mut state) = raving_wgpu::initialize().await?;

    let mut egui_ctx =
        EguiCtx::init(&event_loop, &state, Some(wgpu::Color::BLACK));

    let mut first_resize = true;

    let mut prev_frame_t = std::time::Instant::now();

    event_loop.run(move |event, _, control_flow| {
        match event {
            Event::DeviceEvent { device_id, event } => {
                // println!("DeviceEvent: {event:#?}");
            }

            Event::WindowEvent {
                ref event,
                window_id,
            } => {
                let resp = egui_ctx.on_event(event);

                if !resp.consumed {
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
                        // for some reason i get a validation error if i actually attempt
                        // to execute the first resize
                        if first_resize {
                            first_resize = false;
                        } else {
                            state.resize(*physical_size);
                        }
                    }
                }
            }
            Event::RedrawRequested(window_id) if window_id == window.id() => {
                // let w_size = window.inner_size();
                // let size = [w_size.width, w_size.height];
                // gol.render(&mut state, size).unwrap();

                state.device.poll(Maintain::Wait);

                if let Ok(output) = state.surface.get_current_texture() {
                    let output_view = output
                        .texture
                        .create_view(&wgpu::TextureViewDescriptor::default());

                    let mut encoder = state.device.create_command_encoder(
                        &wgpu::CommandEncoderDescriptor {
                            label: Some("egui render"),
                        },
                    );

                    egui_ctx.render(&state, &output_view, &mut encoder);

                    state.queue.submit(Some(encoder.finish()));
                    output.present();
                } else {
                    state.resize(state.size);
                }

                // renderer.update_texture(device, queue, id, image_delta)
            }
            Event::MainEventsCleared => {
                let dt = prev_frame_t.elapsed().as_secs_f32();
                prev_frame_t = std::time::Instant::now();

                egui_ctx.run(&window, |ctx| {
                    egui::CentralPanel::default().show(&ctx, |ui| {
                        ui.label("hello world");
                    });
                });

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

    if let Err(e) = pollster::block_on(run()) {
        log::error!("{:?}", e);
    }

    Ok(())
}
