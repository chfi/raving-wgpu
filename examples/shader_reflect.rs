use raving_wgpu::{
    shader::{interface::PushConstants, render::VertexShader},
    State,
};
use winit::{
    event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent},
    event_loop::ControlFlow,
};

fn load_shader(shader_src: &[u8]) -> anyhow::Result<()> {
    let module = naga::front::spv::parse_u8_slice(
        shader_src,
        &naga::front::spv::Options {
            adjust_coordinate_space: true,
            strict_capabilities: false,
            block_ctx_dump_prefix: None,
        },
    )?;

    let mut pc: Option<PushConstants> = None;

    println!(" --- entry points ---");

    for entry in module.entry_points.iter() {
        println!("{:#?}", entry);
    }

    println!(" --- constants ---");

    for (handle, cnst) in module.constants.iter() {
        println!("{:?} - {:#?}", handle, cnst);
    }

    println!(" --- functions ---");

    for (handle, cnst) in module.functions.iter() {
        println!("{:?} - {:#?}", handle, cnst);
    }

    println!(" --- globals ---");

    for (handle, var) in module.global_variables.iter() {
        let ty = &module.types[var.ty];
        println!(" - {:?} -", var.name);
        println!(" - - - {:#?}", var);
        println!("------------------------");
        println!(" - - - {:#?}", ty);
        println!(" ty scalar kind: {:?}", ty.inner.scalar_kind());
        println!();

        if let naga::AddressSpace::PushConstant = var.space {
            println!("in push constants");
            println!("ty.inner: {:#?}", ty.inner);
            let push = PushConstants::from_naga_struct(
                &module,
                &ty.inner,
                naga::ShaderStage::Compute,
            )?;
            pc = Some(push);
        }

        if let naga::TypeInner::Struct { members, .. } = &ty.inner {
            for mem in members {
                let mem_ty = &module.types[mem.ty];
                println!(
                    "member: {:?} - {:#?}\n{:?} - {:?}",
                    mem,
                    mem_ty,
                    mem_ty.inner.scalar_kind(),
                    mem_ty.inner.size(&module.constants),
                );
            }
        }
    }

    println!("{:#?}", module);

    Ok(())
}

fn compute() -> anyhow::Result<()> {
    let shader_src = include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/shaders/shader.comp.spv"
    ));

    load_shader(shader_src)
}

fn vertex() -> anyhow::Result<()> {
    let shader_src = include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/shaders/shader.vert.spv"
    ));

    load_shader(shader_src)
}

fn fragment() -> anyhow::Result<()> {
    let shader_src = include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/shaders/shader.frag.spv"
    ));

    load_shader(shader_src);

    
    let module = naga::front::spv::parse_u8_slice(
        shader_src,
        &naga::front::spv::Options {
            adjust_coordinate_space: true,
            strict_capabilities: false,
            block_ctx_dump_prefix: None,
        },
    )?;

    // let frag_locs =
    //     raving_wgpu::shader::render::find_fragment_var_location_map(&module);
    // log::warn!("fragment output locations: {:#?}", frag_locs);

    Ok(())
}

pub fn main() -> anyhow::Result<()> {
    env_logger::builder()
        .filter_level(log::LevelFilter::Warn)
        .init();

    // compute();
    // vertex();
    fragment();

    Ok(())
}
