use raving_wgpu::State;
use winit::{
    event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent},
    event_loop::ControlFlow,
};

pub fn main() -> anyhow::Result<()> {
    env_logger::init();
    
    let shader_src = include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/shaders/shader.comp.spv"
    ));

    let module = naga::front::spv::parse_u8_slice(
        shader_src,
        &naga::front::spv::Options {
            adjust_coordinate_space: true,
            strict_capabilities: false,
            block_ctx_dump_prefix: None,
        },
    )?;

    let mut globals: Vec<(&naga::GlobalVariable, &naga::Type)> = Vec::new();

    for (handle, var) in module.global_variables.iter() {
        let ty = &module.types[var.ty];
        println!(" - {:?} -", var.name);
        println!(" - - - {:#?}", ty);
        println!();

        globals.push((var, ty));
    }

    Ok(())
}
