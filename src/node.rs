/*

read shader source into naga module and wgpu shader module

*/

use std::collections::{BTreeMap, HashMap, HashSet};

use anyhow::{Context, Result};
use wgpu::ShaderLocation;

#[derive(Debug)]
pub struct NodeDesc {
    shader: wgpu::ShaderModule,
    //
}

#[derive(Debug)]
struct VertexInputs {
    // location -> name
    attribute_names: Vec<String>,
    // location -> type
    attribute_types: Vec<naga::TypeInner>,
}

#[derive(Debug)]
struct FragmentOutputs {
    attch_names: Vec<String>,
    attch_types: Vec<naga::TypeInner>,
}

#[derive(Debug)]
struct BindGroups {
    // string -> (group id, bind group entry)
    bindings: BTreeMap<String, (u32, wgpu::BindGroupLayoutEntry)>,

    layouts: Vec<wgpu::BindGroupLayout>,
}

impl BindGroups {
    pub fn create_bind_groups(
        &self,
        device: &wgpu::Device,
        resources: &HashMap<String, wgpu::BindingResource<'_>>,
    ) -> Result<Vec<wgpu::BindGroup>> {
        let mut group_entries: BTreeMap<u32, Vec<wgpu::BindGroupEntry>> =
            BTreeMap::new();
        // iter through resources, map to group, sort by binding

        for (name, (group, layout_entry)) in &self.bindings {
            let resource = resources
                .get(name)
                .with_context(|| format!("Missing resource `{name}`"))?
                .clone();

            let entry = wgpu::BindGroupEntry {
                binding: layout_entry.binding,
                resource,
            };

            group_entries.entry(*group).or_default().push(entry);
        }

        group_entries
            .values_mut()
            .for_each(|es| es.sort_by_key(|e| e.binding));

        let mut groups = Vec::new();

        for (group, entries) in group_entries {
            let layout = &self.layouts[group as usize];

            let bind_group =
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout,
                    entries: entries.as_slice(),
                });

            groups.push(bind_group);
        }

        Ok(groups)
    }
}

impl VertexInputs {
    fn array_stride(&self, attrs: &[&str]) -> u64 {
        let mut stride = 0;

        for name in attrs {
            let (ix, _) = self
                .attribute_names
                .iter()
                .enumerate()
                .find(|(_, n)| n == name)
                .unwrap();

            let ty = &self.attribute_types[ix];

            let size = match ty {
                naga::TypeInner::Scalar { width, .. } => *width as u64,
                naga::TypeInner::Vector { width, size, .. } => {
                    *size as u64 * *width as u64
                }
                _ => panic!("unsupported vertex attribute type"),
            };

            stride += size;
        }

        stride
    }

    fn vertex_buffer_layouts<'a>(
        &self,
        // buffer_attributes: &'a [
        buffer_attributes: impl IntoIterator<
            Item = (&'a [&'a str], wgpu::VertexStepMode),
        >,
        // attributes: &HashMap<&str, &'a wgpu::Buffer>,
    ) -> Result<Vec<(u64, Vec<wgpu::VertexAttribute>, wgpu::VertexStepMode)>>
    {
        // ) -> Result<Vec<wgpu::VertexBufferLayout<'a>>> {

        let mut remaining_names = self
            .attribute_names
            .iter()
            .enumerate()
            .map(|(l, n)| (n.as_str(), l as u32))
            .collect::<HashMap<_, _>>();

        let mut attribute_lists = Vec::new();

        for (buffer_attrs, step_mode) in buffer_attributes {
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

            let stride = offset;
            attribute_lists.push((stride, list, step_mode));
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

    let size = size.map(|s| s as u32).unwrap_or(1);

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

    if let Some(result) = &entry_point.function.result {
        let ty = module.types.get_handle(result.ty)?;

        match &ty.inner {
            naga::TypeInner::Struct { members, .. } => {
                for member in members.iter() {
                    let name = member
                        .name
                        .as_ref()
                        .map(|s| s.as_str())
                        .unwrap_or("<MISSING>");

                    let inner_ty =
                        module.types.get_handle(member.ty)?.inner.clone();

                    if let Some(naga::Binding::Location { location, .. }) =
                        &member.binding
                    {
                        args.push((*location, name.to_string(), inner_ty));
                    } else {
                        anyhow::bail!(
                            "Incompatible fragment output binding: `{name}` {:?}",
                            &member.binding
                        );
                    }
                }
            }
            other => {
                anyhow::bail!("Incompatible fragment output type: {other:?}");
            }
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

fn naga_global_bind_group_entry(
    module: &naga::Module,
    var: &naga::GlobalVariable,
    has_dynamic_offset: bool,
) -> Result<Option<(String, u32, wgpu::BindGroupLayoutEntry)>> {
    // let binding = var.binding.as_ref();
    // let name = var.name.as_ref();

    let space = match var.space {
        naga::AddressSpace::Uniform
        | naga::AddressSpace::Storage { .. }
        | naga::AddressSpace::Handle => var.space,
        // naga::AddressSpace::Storage { access } => todo!(),
        // naga::AddressSpace::Handle => todo!(),
        _ => return Ok(None),
    };

    let binding = var.binding.as_ref();
    let name = var.name.as_ref();

    let Some((binding, name)) = binding.zip(name)
    else {
        return Ok(None);
    };

    let var_ty = module.types.get_handle(var.ty)?;

    // let (binding_ty, binding_count) = match &var_ty.inner {
    let binding_ty = match &var_ty.inner {
        naga::TypeInner::Scalar { .. }
        | naga::TypeInner::Vector { .. }
        | naga::TypeInner::Matrix { .. }
        | naga::TypeInner::Struct { .. } => {
            let buffer_ty = match space {
                naga::AddressSpace::Uniform => wgpu::BufferBindingType::Uniform,
                naga::AddressSpace::Storage { access } => {
                    let read_only =
                        !access.contains(naga::StorageAccess::STORE);
                    wgpu::BufferBindingType::Storage { read_only }
                }
                _ => anyhow::bail!("Incompatible address space for binding"),
            };

            // TODO: actually take the sizes into account (need
            // separate match arms probably) and set the
            // min_binding_size field
            let ty = wgpu::BindingType::Buffer {
                ty: buffer_ty,
                has_dynamic_offset,
                min_binding_size: None,
            };

            ty
        }
        // naga::TypeInner::Atomic { kind, width } => {
        //     todo!()
        // }
        naga::TypeInner::Array { base, size, stride } => {
            let buffer_ty = match space {
                naga::AddressSpace::Uniform => wgpu::BufferBindingType::Uniform,
                naga::AddressSpace::Storage { access } => {
                    let read_only =
                        !access.contains(naga::StorageAccess::STORE);
                    wgpu::BufferBindingType::Storage { read_only }
                }
                _ => anyhow::bail!("Incompatible address space for binding"),
            };

            let ty = wgpu::BindingType::Buffer {
                ty: buffer_ty,
                has_dynamic_offset,
                min_binding_size: None,
            };

            ty
        }
        naga::TypeInner::Image {
            dim,
            arrayed,
            class,
        } => {
            use naga::ImageDimension as ImgDim;
            use wgpu::TextureViewDimension as TvDim;

            let view_dimension = match (*arrayed, dim) {
                (false, ImgDim::D1) => TvDim::D1,
                (false, ImgDim::D2) => TvDim::D2,
                (false, ImgDim::D3) => TvDim::D3,
                (false, ImgDim::Cube) => TvDim::Cube,
                // not supporting arrayed textures yet
                // (true, ImgDim::D2) => TvDim::D2Array,
                // (true, ImgDim::Cube) => TvDim::CubeArray,
                _ => panic!("Unsupported image array/dimension combination"),
            };

            let ty = match class {
                naga::ImageClass::Sampled { kind, multi } => {
                    let sample_type = match kind {
                        naga::ScalarKind::Sint => wgpu::TextureSampleType::Sint,
                        naga::ScalarKind::Uint => wgpu::TextureSampleType::Uint,
                        naga::ScalarKind::Float => {
                            wgpu::TextureSampleType::Float { filterable: true }
                        }
                        _ => unimplemented!(),
                    };

                    wgpu::BindingType::Texture {
                        sample_type,
                        view_dimension,
                        multisampled: *multi,
                    }
                }
                naga::ImageClass::Depth { multi } => {
                    let sample_type = wgpu::TextureSampleType::Depth;

                    wgpu::BindingType::Texture {
                        sample_type,
                        view_dimension,
                        multisampled: *multi,
                    }
                }
                naga::ImageClass::Storage { format, access } => {
                    let read = access.contains(naga::StorageAccess::LOAD);
                    let write = access.contains(naga::StorageAccess::STORE);

                    let input_format = format;
                    let format =
                        crate::util::format_naga_to_wgpu(format.clone());

                    use wgpu::StorageTextureAccess as Access;

                    let access = match (read, write) {
                        (false, false) => unreachable!(),
                        (true, false) => Access::ReadOnly,
                        (false, true) => Access::WriteOnly,
                        (true, true) => Access::ReadWrite,
                    };

                    wgpu::BindingType::StorageTexture {
                        access,
                        format,
                        view_dimension,
                    }
                }
            };

            ty
        }
        naga::TypeInner::Sampler { comparison } => {
            let binding_type = if *comparison {
                wgpu::SamplerBindingType::Comparison
            } else {
                wgpu::SamplerBindingType::Filtering
            };

            wgpu::BindingType::Sampler(binding_type)
        }
        naga::TypeInner::BindingArray { base, size } => todo!(),
        _ => anyhow::bail!("Unsupported binding type `{:?}`", var_ty.inner),
    };

    let group = binding.group;

    //
    let mut entry = wgpu::BindGroupLayoutEntry {
        binding: binding.binding,
        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
        ty: binding_ty,
        count: None,
    };

    Ok(Some((name.to_string(), group, entry)))
}

fn module_bind_groups(
    device: &wgpu::Device,
    module: &naga::Module,
    dynamic_bindings: HashSet<&'_ str>,
) -> Result<BindGroups> {
    let mut bindings: BTreeMap<String, (u32, wgpu::BindGroupLayoutEntry)> =
        BTreeMap::default();

    for (_h, var) in module.global_variables.iter() {
        // get binding (must be Some)
        let binding = var.binding.as_ref();
        let name = var.name.as_ref();

        let has_dynamic_offset = name
            .map(|n| dynamic_bindings.contains(n.as_str()))
            .unwrap_or(false);

        match naga_global_bind_group_entry(module, var, has_dynamic_offset)? {
            None => continue,
            Some((name, group, entry)) => {
                bindings.insert(name, (group, entry));
            }
        }
    }

    let mut sorted_bindings = bindings.values().collect::<Vec<_>>();

    sorted_bindings.sort_by_key(|(group, _)| *group);

    let mut bind_group_layouts: Vec<wgpu::BindGroupLayout> = Vec::new();

    let mut prev_group = None;

    let mut entries = Vec::new();

    for (group, entry) in sorted_bindings {
        if prev_group.is_some_and(|p| p != group) {
            let desc = wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &entries,
            };

            let layout = device.create_bind_group_layout(&desc);
            bind_group_layouts.push(layout);

            entries.clear();
        }

        entries.push(entry.clone());

        prev_group = Some(group);
    }

    let desc = wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &entries,
    };

    let layout = device.create_bind_group_layout(&desc);
    bind_group_layouts.push(layout);

    Ok(BindGroups {
        bindings,
        layouts: bind_group_layouts,
    })
}

pub struct NodeInterface {
    vert_inputs: VertexInputs,
    frag_outputs: FragmentOutputs,

    bind_groups: BindGroups,
}

impl NodeInterface {
    fn graphics<'a>(
        device: &wgpu::Device,
        module: &naga::Module,
        vert_entry: &str,
        frag_entry: &str,
        dynamic_binding_vars: impl IntoIterator<Item = &'a str>,
    ) -> Result<Self> {
        let vert_inputs = module_vertex_inputs(module, vert_entry)?;
        let frag_outputs = module_fragment_outputs(module, frag_entry)?;

        println!("frag_outputs: {frag_outputs:#?}");

        let dyn_bindings = dynamic_binding_vars.into_iter().collect();

        let mut bind_groups = module_bind_groups(device, module, dyn_bindings)?;

        bind_groups.bindings.values_mut().for_each(|(_, entry)| {
            entry.visibility = wgpu::ShaderStages::VERTEX_FRAGMENT
        });

        println!("bind groups: {bind_groups:#?}");

        Ok(Self {
            vert_inputs,
            frag_outputs,
            bind_groups,
        })
    }

    pub fn render_pass<'pass>(
        &self,
        attachments: &[(
            &'pass wgpu::TextureView,
            wgpu::Operations<wgpu::Color>,
        )],
        encoder: &'pass mut wgpu::CommandEncoder,
    ) -> Result<wgpu::RenderPass<'pass>> {
        let attchs = attachments
            .into_iter()
            .map(|(view, ops)| {
                //
                Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    ops: *ops,
                })
            })
            .collect::<Vec<_>>();

        // TODO make sure the attachments actually match; use a map with names

        let pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: attchs.as_slice(),
            depth_stencil_attachment: None,
        });

        Ok(pass)
    }

    pub fn create_bind_groups(
        &self,
        device: &wgpu::Device,
        resources: &HashMap<String, wgpu::BindingResource<'_>>,
    ) -> Result<Vec<wgpu::BindGroup>> {
        self.bind_groups.create_bind_groups(device, resources)
    }
}

pub struct GraphicsNode {
    pub interface: NodeInterface,
    pub pipeline: wgpu::RenderPipeline,
}

pub fn graphics_node<'a, 'b>(
    device: &wgpu::Device,

    shader_src: &str,
    vert_entry: &str,
    frag_entry: &str,

    dynamic_binding_vars: impl IntoIterator<Item = &'b str>,

    primitive: wgpu::PrimitiveState,
    depth_stencil: Option<wgpu::DepthStencilState>,
    multisample: wgpu::MultisampleState,

    vertex_buffer_attrs: impl IntoIterator<
        Item = (&'a [&'a str], wgpu::VertexStepMode),
    >,
    fragment_attchs: impl IntoIterator<Item = (&'a str, wgpu::ColorTargetState)>,
) -> Result<GraphicsNode> {
    let naga_module = naga::front::wgsl::parse_str(shader_src)?;

    let interface = NodeInterface::graphics(
        device,
        &naga_module,
        vert_entry,
        frag_entry,
        dynamic_binding_vars,
    )?;

    let pipeline_layout = {
        let layout_refs =
            interface.bind_groups.layouts.iter().collect::<Vec<_>>();

        if layout_refs.is_empty() {
            None
        } else {
            let desc = wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: layout_refs.as_slice(),
                push_constant_ranges: &[],
            };

            println!("pipeline layout: {desc:#?}");

            Some(device.create_pipeline_layout(&desc))
        }
    };

    // TODO need to pass some more metadata (shader source path/file name) for the labels
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(shader_src.into()),
    });

    // let vertex_buffers: Vec<wgpu::VertexBufferLayout> = Vec::new();

    let vertex_attributes = interface
        .vert_inputs
        .vertex_buffer_layouts(vertex_buffer_attrs)?;

    let vertex_buffers = {
        let mut bufs = Vec::new();

        for (stride, attributes, step_mode) in &vertex_attributes {
            // let stride = interface.vert_inputs.array_stride(attrs);

            let layout = wgpu::VertexBufferLayout {
                array_stride: *stride,
                step_mode: *step_mode,
                attributes,
            };

            bufs.push(layout);
        }

        bufs
    };

    let vertex_state = wgpu::VertexState {
        module: &module,
        entry_point: vert_entry,
        buffers: &vertex_buffers,
    };

    let color_targets = {
        let mut tgts: Vec<(u32, wgpu::ColorTargetState)> = fragment_attchs
            .into_iter()
            .map(|(name, tgt)| {
                let (loc, _) = interface
                    .frag_outputs
                    .attch_names
                    .iter()
                    .enumerate()
                    .find(|(_, n)| *n == name)
                    .unwrap();

                (loc as u32, tgt)
            })
            .collect::<Vec<_>>();

        tgts.sort_by_key(|(loc, _)| *loc);

        tgts.into_iter()
            .map(|(_, tgt)| Some(tgt))
            .collect::<Vec<_>>()
    };

    let fragment_state = wgpu::FragmentState {
        module: &module,
        entry_point: frag_entry,
        targets: &color_targets,
    };

    let pipeline =
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: pipeline_layout.as_ref(),
            vertex: vertex_state,
            primitive,
            depth_stencil,
            multisample,
            fragment: Some(fragment_state),
            multiview: None,
        });

    Ok(GraphicsNode {
        interface,
        pipeline,
    })
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

/*
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
            NodeInterface::graphics(&naga_mod, "vs_main", "fs_main")?;

        // todo!();
        Ok(())
    }
}
*/
