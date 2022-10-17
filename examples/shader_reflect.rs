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

    let frag_locs =
        raving_wgpu::shader::render::find_fragment_var_location_map(&module);
    log::warn!("fragment output locations: {:#?}", frag_locs);

    Ok(())
}

pub fn froggy() -> anyhow::Result<()> {
    use datafrog::*;

    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
    struct Expr(usize);

    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
    struct Global(usize);

    let mut iteration = Iteration::new();

    let globals = iteration.variable::<(Global, String)>("globals");

    let expr_load = iteration.variable::<(Expr, Expr)>("expr_load");
    let expr_struct_load =
        iteration.variable::<(Expr, (usize, Expr))>("expr_struct_load");
    let expr_global = iteration.variable::<(Expr, Global)>("expr_global");

    let result_ix_location =
        iteration.variable::<(usize, u32)>("result_ix_location");

    globals.extend(
        ["f_color", "u_id", "data"]
            .into_iter()
            .enumerate()
            .map(|(i, s)| (Global(i), s.to_string())),
    );

    use Expr as E;
    use Global as G;

    /*
        expressions: {
        [1]: N/a
        [2]: Global(1),
        [3]: Global(0),
        [4]: Global(2),
        [5]: Load(3),
        [6]: Load(4),
        [7]: Compose {
            ty: [8],
            components: [
                [5],
                [6],
            ],
        },
    },
         */

    expr_global.extend([(E(2), G(1)), (E(3), G(0)), (E(4), G(2))]);
    expr_load.extend([(E(5), E(3)), (E(6), E(4))]);

    let struct_loads = [(E(7), (0, E(5))), (E(7), (1, E(6)))];
    expr_struct_load.extend(struct_loads.clone());

    let init_struct_loads = Relation::from_iter(struct_loads);

    result_ix_location.extend([(0, 1), (1, 0)]);

    let return_expr = Relation::from_iter([E(7)]);

    let result = iteration.variable::<(Global, (String, u32))>("result");

    println!("init struct_loads: {:#?}", &init_struct_loads.elements);

    let util_struct_load =
        iteration.variable::<(Expr, (usize, Expr))>("util_struct_load");

    let global_return = iteration.variable("global_return");

    let global_loc = iteration.variable("global_loc");

    println!("running ~~~");

    while iteration.changed() {
        /* in a nutshell:

            result(global_id, name, location)
              :- return_expr(return_id),
                 expr_struct_load(return_id, (ix, ptr)),
                 expr_global(ptr, global_id),
                 global(global_id, name),
                 result_ix_location(ix, location).
        */

        // helper
        // util_struct_load(ptr, (ix, return_id))
        //    :- expr_struct_load(return_id, (ix, ptr))
        util_struct_load
            .from_map(&expr_struct_load, |(return_id, (ix, ptr))| {
                (*ptr, (*ix, *return_id))
            });

        /*  util_struct_load(dst, (ix, return_id))
              :- util_struct_load(ptr, (ix, return_id)),
                 expr_load(ptr, dst).
        */
        util_struct_load.from_join(
            &util_struct_load,
            &expr_load,
            |ptr, (ix, return_id), dst| (*dst, (*ix, *return_id)),
        );

        /*
           global_return(ix, global_id)
             :- util_struct_load(ptr, (ix, return_id)),
                expr_global(ptr, global_id).
        */
        global_return.from_join(
            &util_struct_load,
            &expr_global,
            |ptr, (ix, return_id), global_id| (*ix, *global_id),
        );

        /*
          global_loc(global_id, location)
           :- global_return(ix, global_id),
              result_ix_location(ix, location).
        */

        global_loc.from_join(
            &global_return,
            &result_ix_location,
            |ix, g_id, loc| (*g_id, *loc),
        );

        result.from_join(&globals, &global_loc, |g_id, name, loc| {
            (*g_id, (name.to_string(), *loc))
        });
    }
    println!("complete!");
    let rel_struct_load = expr_struct_load.complete();

    println!("rel_struct_load: {:#?}", &rel_struct_load.elements);

    let rel = util_struct_load.complete();
    println!("inverted: {:#?}", &rel.elements);

    let val = global_return.complete();
    println!("global_return: {:#?}", &val.elements);

    let result = result.complete();

    let result = result
        .elements
        .into_iter()
        .map(|(_, s)| s)
        .collect::<Vec<_>>();

    println!("results: {:#?}", result);

    Ok(())
}

pub fn main() -> anyhow::Result<()> {
    env_logger::builder()
        .filter_level(log::LevelFilter::Warn)
        .init();

    // compute();
    // vertex();
    fragment()?;

    // froggy()?;

    Ok(())
}
