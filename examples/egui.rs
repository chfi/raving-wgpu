use winit::event::{ElementState, Event, VirtualKeyCode, WindowEvent};
use winit::event_loop::ControlFlow;

use anyhow::Result;

async fn run() -> Result<()> {
    let (event_loop, window, mut state) = raving_wgpu::initialize().await?;
    // let size = window.inner_size();

    let mut egui_ctx = egui::Context::default();

    let mut egui_state = egui_winit::State::new(&event_loop);

    let output_color_format = state.surface_format;
    let output_depth_format = wgpu::TextureFormat::Depth32Float;
    let msaa_samples = 1;

    let renderer = egui_wgpu::Renderer::new(
        &state.device,
        output_color_format,
        // None,
        Some(output_depth_format),
        msaa_samples,
    );

    // let rpass = egui_wgpu::

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
                let resp = egui_state.on_event(&egui_ctx, event);

                if !resp.consumed {
                    if let WindowEvent::KeyboardInput { input, .. } = event {
                        use winit::event::VirtualKeyCode as Key;
                        if input.state == ElementState::Pressed
                            && input.virtual_keycode == Some(Key::Escape)
                        {
                            *control_flow = ControlFlow::Exit;
                        }
                    }
                }

                /*
                        if window_id == window.id() => match event {
                    WindowEvent::AxisMotion {
                        device_id,
                        axis,
                        value,
                    } => {
                        // println!("AxisMotion: {device_id:?}\t{axis:?}\t{value:?}");
                    }
                    WindowEvent::Touch(touch) => {
                        // println!("Touch: {touch:?}");
                    }
                    WindowEvent::TouchpadPressure {
                        device_id,
                        pressure,
                        stage,
                    } => {
                        // println!("TouchPadPressure pressure: {pressure:?}\tstage: {stage:?}");
                    }
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
                }
                */
            }
            Event::RedrawRequested(window_id) if window_id == window.id() => {
                // let w_size = window.inner_size();
                // let size = [w_size.width, w_size.height];
                // gol.render(&mut state, size).unwrap();


                // renderer.update_texture(device, queue, id, image_delta)
            }
            Event::MainEventsCleared => {
                let dt = prev_frame_t.elapsed().as_secs_f32();
                prev_frame_t = std::time::Instant::now();

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
