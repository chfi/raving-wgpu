/*

read shader source into naga module and wgpu shader module

*/

use anyhow::{Context, Result};
use wgpu::ShaderLocation;

pub struct NodeDesc {
    shader: wgpu::ShaderModule,
    //
}

struct VertexInputs {
    // location -> name
    attribute_names: Vec<String>,
    // location -> type
    attribute_types: Vec<naga::TypeInner>,
}

fn binding_location(binding: &naga::Binding) -> Option<ShaderLocation> {
    match binding {
        naga::Binding::BuiltIn(_) => None,
        naga::Binding::Location { location, .. } => Some(*location),
    }
}

fn valid_vertex_attribute_type(ty: &naga::TypeInner) -> bool {
    match ty {
        naga::TypeInner::Scalar { .. } => true,
        naga::TypeInner::Vector { .. } => true,
        naga::TypeInner::Matrix { .. } => true,
        _ => false,
    }
}

fn module_vertex_inputs(
    module: &naga::Module,
    entry_point: &str,
) -> Result<VertexInputs> {
    let entry_point = module_entry_point(&module, entry_point)?;

    let mut args: Vec<(u32, String, naga::TypeInner)> = Vec::new();

    for arg in entry_point.function.arguments.iter() {
        let location = arg.binding.as_ref().and_then(binding_location);
        let name = arg.name.as_ref();

        let ty = module.types.get_handle(arg.ty)?;

        if let Some((location, name)) = location.zip(name) {
            if !valid_vertex_attribute_type(&ty.inner) {
                anyhow::bail!(
                    "Unsupported vertex attribute type: {:?}",
                    ty.inner
                );
            }

            args.push((location, name.to_string(), ty.inner.clone()));
        }
    }

    args.sort_by_key(|(loc, _, _)| *loc);

    let (names, types) =
        args.into_iter().map(|(_loc, name, ty)| (name, ty)).unzip();

    Ok(VertexInputs {
        attribute_names: names,
        attribute_types: types,
    })
}

pub struct NodeInterface {
    vert_inputs: Vec<()>,
    frag_outputs: Vec<()>,

    bindings: Vec<()>,
}

pub fn graphics_node_interface(
    module: &naga::Module,
    vert_entry: &str,
    frag_entry: &str,
) -> Result<NodeInterface> {
    let vert_entry_point = module_entry_point(&module, vert_entry)?;

    // let vert_inputs: Vec<()> = vert_entry_point
    // let vert_inputs = vert_entry_point
    //     .function
    //     .arguments
    //     .iter()
    //     .filter_map(|arg| Some(arg.ty))
    //     .collect::<Vec<_>>();

    // for h in &vert_inputs {

    let args = vert_entry_point.function.arguments.iter();

    for arg in args {
        println!("  -- {arg:#?}");
        if let Some(binding) = arg.binding.as_ref() {
            match binding {
                naga::Binding::BuiltIn(b) => print!(" [builtin {b:?}"),
                naga::Binding::Location {
                    location,
                    interpolation,
                    sampling,
                } => print!(" [loc({location})]"),
            }
        }
        let ty_h = arg.ty;
        let ty = module.types.get_handle(ty_h)?;
        let ty = ty.inner.clone();

        println!("  {ty_h:?} -> {ty:?}");
    }

    // println!("vert inputs: {vert_inputs:#?}");

    let frag_entry_point = module_entry_point(&module, frag_entry)?;

    let frag_outputs = {
        let result = frag_entry_point.function.result.as_ref();
        println!("result: {:#?}", result);
        let output_type =
            result.and_then(|r| module.types.get_handle(r.ty).ok());

        println!("output type: {output_type:#?}");
    };

    todo!();
}

fn module_entry_point<'a>(
    module: &'a naga::Module,
    entry_point: &str,
) -> Result<&'a naga::EntryPoint> {
    module
        .entry_points
        .iter()
        .find(|ep| ep.name == entry_point)
        .with_context(|| format!("Entry point `{entry_point}` not found"))
}

pub mod graphics {

    pub struct VertexShader {
        pub shader_module: wgpu::ShaderModule,
        pub entry_point: naga::EntryPoint,
    }
    //
}

pub mod compute {
    //
}

#[cfg(test)]
mod tests {
    use super::*;

    use anyhow::Result;

    #[test]
    fn test_wgsl() -> Result<()> {
        //
        // let shader_src = include_str!("../shaders/test.wgsl");
        let shader_src = include_str!("../shaders/test2.wgsl");
        let naga_mod = naga::front::wgsl::parse_str(shader_src)?;

        println!("{:#?}", naga_mod);

        println!("//////////////////////");

        let interface =
            graphics_node_interface(&naga_mod, "vs_main", "fs_main")?;

        // todo!();
        Ok(())
    }
}
