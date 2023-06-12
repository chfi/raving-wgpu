/*

read shader source into naga module and wgpu shader module

*/

use std::collections::{HashMap, HashSet};

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

struct FragmentOutputs {
    attch_names: Vec<String>,
    attch_types: Vec<naga::TypeInner>,
}

impl VertexInputs {
    fn vertex_buffer_layouts<'a>(
        &self,
        // buffer_attributes: &'a [
        buffer_attributes: impl IntoIterator<Item = &'a [&'a str]>,
        // attributes: &HashMap<&str, &'a wgpu::Buffer>,
    ) -> Result<Vec<Vec<wgpu::VertexAttribute>>> {
        // ) -> Result<Vec<wgpu::VertexBufferLayout<'a>>> {

        let mut remaining_names = self
            .attribute_names
            .iter()
            .enumerate()
            .map(|(l, n)| (n.as_str(), l as u32))
            .collect::<HashMap<_, _>>();

        let mut attribute_lists = Vec::new();

        for buffer_attrs in buffer_attributes {
            let mut list = Vec::new();

            let mut offset = 0u64;

            for &attr_name in buffer_attrs {
                let location = remaining_names.remove(attr_name)
                    .with_context(|| format!("Vertex attribute `{attr_name}` not found in shader"))?;

                let ty = &self.attribute_types[location as usize];

                let (kind, width, size) = match ty {
                    naga::TypeInner::Scalar { kind, width } => {
                        (*kind, *width, None)
                    }
                    naga::TypeInner::Vector { size, kind, width } => {
                        (*kind, *width, Some(*size))
                    }
                    _ => unreachable!(),
                };

                let format = naga_type_vertex_format(kind, width, size)
                    .with_context(|| {
                        format!("Incompatible vertex attribute type: `{ty:?}`")
                    })?;

                list.push(wgpu::VertexAttribute {
                    format,
                    offset,
                    shader_location: location,
                });

                offset += format.size();
            }
            attribute_lists.push(list);
        }

        if !remaining_names.is_empty() {
            // TODO bail with error message containing remaining variable names
        }

        Ok(attribute_lists)
    }
}

fn naga_type_vertex_format(
    scalar_kind: naga::ScalarKind,
    width: u8,
    size: Option<naga::VectorSize>,
) -> Option<wgpu::VertexFormat> {
    use naga::VectorSize as Size;
    use wgpu::VertexFormat as Format;

    match scalar_kind {
        naga::ScalarKind::Sint => {
            if width == 4 {
                match size {
                    None => Some(Format::Sint32),
                    Some(Size::Bi) => Some(Format::Sint32x2),
                    Some(Size::Tri) => Some(Format::Sint32x3),
                    Some(Size::Quad) => Some(Format::Sint32x4),
                }
            } else {
                None
            }
        }
        naga::ScalarKind::Uint => {
            if width == 4 {
                match size {
                    None => Some(Format::Uint32),
                    Some(Size::Bi) => Some(Format::Uint32x2),
                    Some(Size::Tri) => Some(Format::Uint32x3),
                    Some(Size::Quad) => Some(Format::Uint32x4),
                }
            } else {
                None
            }
        }
        naga::ScalarKind::Float => {
            if width == 4 {
                match size {
                    None => Some(Format::Float32),
                    Some(Size::Bi) => Some(Format::Float32x2),
                    Some(Size::Tri) => Some(Format::Float32x3),
                    Some(Size::Quad) => Some(Format::Float32x4),
                }
            } else {
                None
            }
        }
        naga::ScalarKind::Bool => None,
    }
}

fn naga_type_texture_format_valid(
    scalar_kind: naga::ScalarKind,
    width: u8,
    size: Option<naga::VectorSize>,
    format: wgpu::TextureFormat,
) -> bool {
    use naga::ScalarKind as Kind;
    use naga::VectorSize as Size;
    use wgpu::TextureFormat as Format;

    let size = size.map(vector_size_u64).unwrap_or(1);

    match format {
        Format::R8Unorm | Format::R8Snorm => {
            matches!(scalar_kind, Kind::Float) && size == 1
        }
        Format::R8Uint => matches!(scalar_kind, Kind::Uint) && size == 1,
        Format::R8Sint => matches!(scalar_kind, Kind::Sint) && size == 1,
        Format::R16Uint => matches!(scalar_kind, Kind::Uint) && size == 1,
        Format::R16Sint => matches!(scalar_kind, Kind::Sint) && size == 1,
        Format::R16Unorm | Format::R16Snorm | Format::R16Float => {
            matches!(scalar_kind, Kind::Float) && size == 1
        }
        Format::Rg8Unorm | Format::Rg8Snorm => {
            matches!(scalar_kind, Kind::Float) && size == 2
        }
        Format::Rg8Uint | Format::Rg16Uint => {
            matches!(scalar_kind, Kind::Uint) && size == 2
        }
        Format::Rg8Sint | Format::Rg16Sint => {
            matches!(scalar_kind, Kind::Sint) && size == 2
        }
        Format::R32Uint => matches!(scalar_kind, Kind::Uint) && size == 1,
        Format::R32Sint => matches!(scalar_kind, Kind::Sint) && size == 1,
        Format::R32Float => matches!(scalar_kind, Kind::Float) && size == 1,
        Format::Rg16Float
        | Format::Rg16Unorm
        | Format::Rg16Snorm
        | Format::Rg32Float => {
            //
            matches!(scalar_kind, Kind::Float) && size == 2
        }
        Format::Rgba8Unorm
        | Format::Rgba8UnormSrgb
        | Format::Rgba8Snorm
        | Format::Bgra8Unorm
        | Format::Bgra8UnormSrgb
        | Format::Rgba16Unorm
        | Format::Rgba16Snorm
        | Format::Rgba16Float
        | Format::Rgba32Float => {
            //
            matches!(scalar_kind, Kind::Float) && size == 4
        }
        Format::Rgba8Uint | Format::Rgba16Uint | Format::Rgba32Uint => {
            matches!(scalar_kind, Kind::Uint) && size == 4
        }
        Format::Rgba8Sint | Format::Rgba16Sint | Format::Rgba32Sint => {
            matches!(scalar_kind, Kind::Sint) && size == 4
        }
        Format::Rg32Uint => matches!(scalar_kind, Kind::Uint) && size == 2,
        Format::Rg32Sint => matches!(scalar_kind, Kind::Sint) && size == 2,
        _ => false,
    }
}

fn binding_location(binding: &naga::Binding) -> Option<ShaderLocation> {
    match binding {
        naga::Binding::BuiltIn(_) => None,
        naga::Binding::Location { location, .. } => Some(*location),
    }
}

fn vector_size_u64(size: naga::VectorSize) -> u64 {
    match size {
        naga::VectorSize::Bi => 2,
        naga::VectorSize::Tri => 3,
        naga::VectorSize::Quad => 4,
    }
}

fn valid_shader_io_type(ty: &naga::TypeInner) -> bool {
    match ty {
        naga::TypeInner::Scalar { kind, .. } => kind.is_numeric(),
        naga::TypeInner::Vector { kind, .. } => kind.is_numeric(),
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
            if !valid_shader_io_type(&ty.inner) {
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

fn module_fragment_outputs(
    module: &naga::Module,
    entry_point: &str,
) -> Result<FragmentOutputs> {
    let entry_point = module_entry_point(&module, entry_point)?;

    let mut args: Vec<(u32, String, naga::TypeInner)> = Vec::new();

    for arg in entry_point.function.arguments.iter() {
        let location = arg.binding.as_ref().and_then(binding_location);
        let name = arg.name.as_ref();

        let ty = module.types.get_handle(arg.ty)?;

        if let Some((location, name)) = location.zip(name) {
            if !valid_shader_io_type(&ty.inner) {
                anyhow::bail!(
                    "Unsupported fragment attachment type: {:?}",
                    ty.inner
                );
            }

            args.push((location, name.to_string(), ty.inner.clone()));
        }
    }

    args.sort_by_key(|(loc, _, _)| *loc);

    let (names, types) =
        args.into_iter().map(|(_loc, name, ty)| (name, ty)).unzip();

    Ok(FragmentOutputs {
        attch_names: names,
        attch_types: types,
    })
}

pub struct NodeInterface {
    vert_inputs: VertexInputs,
    frag_outputs: FragmentOutputs,

    bind_groups: Vec<()>,
}

pub fn graphics_node<'a>(
    shader_src: &str,
    vert_entry: &str,
    frag_entry: &str,

    primitive: wgpu::PrimitiveState,
    depth_stencil: Option<wgpu::DepthStencilState>,
    multisample: wgpu::MultisampleState,

    vertex_buffer_attrs: impl IntoIterator<Item = &'a [&'a str]>,
    fragment_attchs: impl IntoIterator<
        Item = &'a [(&'a str, wgpu::ColorTargetState)],
    >,
) -> Result<()> {
    todo!();
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
