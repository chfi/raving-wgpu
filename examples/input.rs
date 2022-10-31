use stick::Controller;
use winit::event::{ElementState, Event, VirtualKeyCode, WindowEvent};
use winit::event_loop::ControlFlow;

use gilrs::{Button, Event as GEvent, Gilrs};

use raving_wgpu::graph::dfrog::{Graph, InputResource};

async fn run() -> anyhow::Result<()> {
    let (event_loop, window, mut state) = raving_wgpu::initialize().await?;
    let size = window.inner_size();

    let dims = [size.width, size.height];

    let mut first_resize = true;

    let mut prev_frame_t = std::time::Instant::now();

    event_loop.run(move |event, _, control_flow| {
        match event {
            Event::DeviceEvent { device_id, event } => {
                println!("DeviceEvent: {event:#?}");
            }
            Event::WindowEvent {
                ref event,
                window_id,
            } if window_id == window.id() => match event {
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
            },
            Event::RedrawRequested(window_id) if window_id == window.id() => {
                // let w_size = window.inner_size();
                // let size = [w_size.width, w_size.height];
                // gol.render(&mut state, size).unwrap();
            }
            Event::MainEventsCleared => {
                let dt = prev_frame_t.elapsed().as_secs_f32();
                prev_frame_t = std::time::Instant::now();

                // gol.update(dt);

                // {
                //     let w_size = window.inner_size();
                //     let size = [w_size.width, w_size.height];

                //     gol.cfg.viewport_size = size;

                //     state.queue.write_buffer(
                //         &gol.cfg_buffer,
                //         0,
                //         bytemuck::cast_slice(&[gol.cfg]),
                //     );
                // }

                window.request_redraw();
            }

            _ => {}
        }
    })
}

pub fn test_gilrs() -> anyhow::Result<()> {
    let mut gilrs = Gilrs::new().unwrap();

    for (_id, gamepad) in gilrs.gamepads() {
        println!("{} is {:?}", gamepad.name(), gamepad.power_info());
    }

    let mut active_gamepad = None;
    loop {
        // Examine new events
        while let Some(GEvent { id, event, time }) = gilrs.next_event() {
            println!("{:?} New event from {}: {:?}", time, id, event);
            active_gamepad = Some(id);
        }

        // You can also use cached gamepad state
        if let Some(gamepad) = active_gamepad.map(|id| gilrs.gamepad(id)) {
            if gamepad.is_pressed(Button::South) {
                println!("Button South is pressed (XBox - A, PS - X)");
            }
        }
    }

    Ok(())
}

async fn handle_controller(mut cont: Controller) -> anyhow::Result<()> {
    println!("controller: {}", cont.name());

    loop {
        println!("awaiting event");
        let event = (&mut cont).await;
        println!("received {event:?}");
    }

    Ok(())
}

async fn test_stick() -> anyhow::Result<()> {
    use stick::*;
    let mut listener = Listener::default();

    loop {
        let controller = (&mut listener).await;
        println!("found controller");
        handle_controller(controller).await?;
        dbg!();
    }

    Ok(())
}

fn test_fishstick() -> anyhow::Result<()> {
    use fishsticks::*;
    let mut ctx = GamepadContext::init().map_err(|s| anyhow::anyhow!(s))?;

    for (id_, gamepad) in ctx.gamepads() {
        println!(
            "{id_:?}, analog: {:?}, digital {:?}", 
            gamepad.analog_inputs,
            gamepad.digital_inputs,
        );
    }

    Ok(())
}

pub fn main() -> anyhow::Result<()> {
    use stick::*;
    let mut listener = Listener::default();

    // let builder = gilrs::GilrsBuilder::default()

    // test_gilrs?;
    // test_stick()?;
    test_fishstick()?;

    env_logger::builder()
        .filter_level(log::LevelFilter::Warn)
        .init();

    if let Err(e) = pollster::block_on(run()) {
        // if let Err(e) = pollster::block_on(test_stick()) {
        log::error!("{:?}", e);
    }

    Ok(())
}
