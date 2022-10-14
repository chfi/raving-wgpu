use raving_wgpu::{shader::interface::PushConstants, State};
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

    let mut pc: Option<PushConstants> = None;

    println!(" --- entry points ---");

    for entry in module.entry_points.iter() {
        println!("{:#?}", entry);
    }

    /*
    println!(" --- constants ---");

    for (handle, cnst) in module.constants.iter() {
        println!("{:?} - {:#?}", handle, cnst);
    }
    */

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
                /*
                println!(
                    "member: {:?} - {:#?}\n{:?} - {:?}",
                    mem,
                    mem_ty,
                    mem_ty.inner.scalar_kind(),
                    mem_ty.inner.size(&module.constants),
                );
                */
            }
        }

        // println!(" ")

        globals.push((var, ty));
    }

    // println!("{:#?}", pc);

    // println!("---------\n\n");

    // println!("{:?}", module);

    Ok(())
}
