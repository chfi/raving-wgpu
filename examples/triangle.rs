use raving_wgpu::State;
use winit::{event::{Event, WindowEvent, ElementState, VirtualKeyCode, KeyboardInput}, event_loop::ControlFlow};

pub async fn run() -> anyhow::Result<()> {
    env_logger::init();
    let (event_loop, window, mut state) = raving_wgpu::initialize().await?;

    
    event_loop.run(move |event, _, control_flow| {
        match event {
            Event::WindowEvent {
                ref event,
                window_id,
            } if window_id == window.id() => {
                if !state.input(event) {
                    // UPDATED!
                    match event {
                        WindowEvent::CloseRequested
                        | WindowEvent::KeyboardInput {
                            input:
                                KeyboardInput {
                                    state: ElementState::Pressed,
                                    virtual_keycode: Some(VirtualKeyCode::Escape),
                                    ..
                                },
                            ..
                        } => *control_flow = ControlFlow::Exit,
                        WindowEvent::Resized(physical_size) => {
                            state.resize(*physical_size);
                        }
                        WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                            state.resize(**new_inner_size);
                        }
                        _ => {}
                    }
                }
            }
            _ => {}
        }
    });

    Ok(())
}

pub fn main() {
    if let Err(e) = pollster::block_on(run()) {
        log::error!("{:?}", e);
    }
}
