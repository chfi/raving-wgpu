
use ultraviolet::Vec2;
use winit::event::{
    ElementState, Event, VirtualKeyCode, WindowEvent,
};
use winit::event_loop::ControlFlow;



pub struct GraphLayout {
    positions: Vec<Vec2>,
}



async fn run() -> anyhow::Result<()> {
    let (event_loop, window, mut state) = raving_wgpu::initialize().await?;

    let size = window.inner_size();

    let dims = [size.width, size.height];
    // let mut gol = GameOfLife::new(&state)?;

    let mut first_resize = true;

    let mut prev_frame_t = std::time::Instant::now();

    event_loop.run(move |event, _, control_flow| {
        match event {
            Event::WindowEvent {
                ref event,
                window_id,
            } if window_id == window.id() => match event {
                WindowEvent::Touch(touch) => {
                    //
                    todo!();
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

                        /* 
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
                        */
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
                // let w_size = window.inner_size();
                // let size = [w_size.width, w_size.height];
                // gol.render(&mut state, size).unwrap();
            }
            Event::MainEventsCleared => {
                // RedrawRequested will only trigger once, unless we manually
                // request it.

                let dt = prev_frame_t.elapsed().as_secs_f32();
                prev_frame_t = std::time::Instant::now();

                /*
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
                */

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
