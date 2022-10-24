use std::collections::{BTreeMap, HashMap};

use anyhow::Result;
use wgpu::{
    BindingResource, BindingType, PushConstantRange, StorageTextureAccess,
};

use super::BindingDef;

// uniforms...
// bind groups

#[derive(Debug)]
pub struct GroupBindings {
    pub group_ix: u32,

    pub entries: Vec<wgpu::BindGroupLayoutEntry>,
    pub(crate) bindings: Vec<super::BindingDef>,
}

impl GroupBindings {
    pub fn create_bind_groups(
        group_bindings: &[Self],
        state: &crate::State,
        bind_group_layouts: &[wgpu::BindGroupLayout],
        resources: &[crate::graph::Resource],
        resource_map: &HashMap<String, crate::graph::ResourceId>,
    ) -> Result<Vec<wgpu::BindGroup>> {
        let mut bind_groups = Vec::new();

        for bindings in group_bindings.iter() {
            let ix = bindings.group_ix as usize;
            let layout = &bind_group_layouts[ix];

            let bind_group = bindings.create_bind_group(
                state,
                layout,
                resources,
                resource_map,
            )?;
            bind_groups.push(bind_group);
        }

        Ok(bind_groups)
    }

    pub fn create_bind_group_new(
        &self,
        state: &crate::State,
        layout: &wgpu::BindGroupLayout,
    ) {
        todo!();
    }

    pub fn create_bind_group(
        &self,
        state: &crate::State,
        layout: &wgpu::BindGroupLayout,
        resources: &[crate::graph::Resource],
        resource_map: &HashMap<String, crate::graph::ResourceId>,
    ) -> Result<wgpu::BindGroup> {
        let mut entries = Vec::new();

        for (binding_ix, entry) in self.entries.iter().enumerate() {
            let def = &self.bindings[binding_ix];

            let res_id =
                resource_map.get(def.global_var_name.as_str()).unwrap();

            let resource = &resources[res_id.0];

            let binding_resource = match resource {
                crate::Resource::Buffer { buffer, .. } => match entry.ty {
                    BindingType::Buffer { .. } => BindingResource::Buffer(
                        buffer.as_ref().unwrap().as_entire_buffer_binding(),
                    ),
                    _ => {
                        panic!("TODO: Binding type mismatch!");
                    }
                },
                crate::Resource::Texture { texture, .. } => match entry.ty {
                    BindingType::Sampler(_) => BindingResource::Sampler(
                        &texture.as_ref().unwrap().sampler,
                    ),
                    BindingType::Texture { .. }
                    | BindingType::StorageTexture { .. } => {
                        BindingResource::TextureView(
                            &texture.as_ref().unwrap().view,
                        )
                    }
                    BindingType::Buffer { .. } => {
                        panic!("TODO: Binding type mismatch!");
                    }
                },
            };

            let entry = wgpu::BindGroupEntry {
                binding: binding_ix as u32,
                resource: binding_resource,
            };

            entries.push(entry);
        }

        let bind_group =
            state.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout,
                entries: entries.as_slice(),
            });

        Ok(bind_group)
    }

    /// intended for use when creating all the bind group layouts
    /// for a single shader at once
    pub fn create_bind_group_layouts_checked(
        group_bindings: &[Self],
        state: &crate::State,
    ) -> Result<Vec<wgpu::BindGroupLayout>> {
        let mut bind_group_layouts = Vec::new();
        let mut expected_group = 0;

        dbg!(group_bindings);

        for bindings in group_bindings.iter() {
            let group_ix = bindings.group_ix;

            // Group indices both have to be compact and sorted
            if expected_group != group_ix {
                anyhow::bail!(
                    "Missing group index: Expected {}, but saw {}",
                    expected_group,
                    group_ix
                );
            }

            let bind_group_layout = bindings.create_bind_group_layout(state);
            bind_group_layouts.push(bind_group_layout);

            expected_group += 1;
        }
        Ok(bind_group_layouts)
    }

    pub fn create_bind_group_layout(
        &self,
        state: &crate::State,
    ) -> wgpu::BindGroupLayout {
        state.device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                // label: Some("test bind group layout"),
                label: None,
                entries: self.entries.as_slice(),
            },
        )
    }

    pub fn from_spirv(
        module: &naga::Module,
        stages: wgpu::ShaderStages,
    ) -> Result<Vec<Self>> {
        let mut result = Vec::new();

        let mut shader_bindings: BTreeMap<u32, Vec<(BindingDef, _)>> =
            BTreeMap::default();

        for (handle, var) in module.global_variables.iter() {
            log::warn!("{var:#?}");
            if (matches!(var.space, naga::AddressSpace::Storage { .. })
                || var.space == naga::AddressSpace::Uniform
                || var.space == naga::AddressSpace::Handle)
                && var.binding.is_some()
            {
                let binding = var.binding.as_ref().unwrap();
                let ty = &module.types[var.ty];

                let binding_def = super::BindingDef {
                    global_var_name: var.name.as_ref().unwrap().into(),
                    binding: binding.clone(),
                    ty: ty.inner.clone(),
                };

                shader_bindings
                    .entry(binding.group.clone())
                    .or_default()
                    .push((binding_def, var.space));
            }
        }

        let mut final_bindings = Vec::new();

        let mut expected_group = 0;
        for (group_ix, mut defs) in shader_bindings {
            // Group and binding indices both have to be compact,
            // so while `shader_bindings` is a BTreeMap, its keys
            // have to be in some `0..n` range, and iterated in order
            if expected_group != group_ix {
                anyhow::bail!(
                    "Missing group index: Expected {}, but saw {}",
                    expected_group,
                    group_ix
                );
            }

            defs.sort_by_key(|(def, space)| def.binding.binding);
            final_bindings.push(defs.clone());

            let mut entries: Vec<wgpu::BindGroupLayoutEntry> = Vec::new();

            for (binding_ix, (def, space)) in defs.iter().enumerate() {
                if binding_ix as u32 != def.binding.binding
                    || expected_group != def.binding.group
                {
                    anyhow::bail!(
                        "Binding index mismatch: Was (group {}, binding {}),\
                        but expected (group {}, binding {})",
                        def.binding.group,
                        def.binding.binding,
                        expected_group,
                        binding_ix
                    );
                }

                let (ty, count) = match &def.ty {
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
                            (true, ImgDim::D2) => TvDim::D2Array,
                            (true, ImgDim::Cube) => TvDim::CubeArray,
                            _ => panic!(
                                "Unsupported image array/dimension combination"
                            ),
                            // (true, ImgDim::D1) => TvDim::D1,
                            // (true, ImgDim::D3) => TvDim::D3,
                        };

                        match class {
                            naga::ImageClass::Depth { multi } => {
                                let sample_type =
                                    wgpu::TextureSampleType::Depth;

                                let ty = wgpu::BindingType::Texture {
                                    sample_type,
                                    view_dimension,
                                    multisampled: *multi,
                                };

                                (ty, None)
                            }
                            naga::ImageClass::Sampled { kind, multi } => {
                                let sample_type = match kind {
                                    naga::ScalarKind::Sint => {
                                        wgpu::TextureSampleType::Sint
                                    }
                                    naga::ScalarKind::Uint => {
                                        wgpu::TextureSampleType::Uint
                                    }
                                    naga::ScalarKind::Float => {
                                        wgpu::TextureSampleType::Float {
                                            filterable: true,
                                        }
                                    }
                                    _ => unimplemented!(),
                                };

                                let ty = wgpu::BindingType::Texture {
                                    sample_type,
                                    view_dimension,
                                    multisampled: *multi,
                                };

                                (ty, None)
                            }
                            naga::ImageClass::Storage { format, access } => {
                                let read =
                                    access.contains(naga::StorageAccess::LOAD);
                                let write =
                                    access.contains(naga::StorageAccess::STORE);

                                let input_format = format;
                                let format =
                                    super::format_naga_to_wgpu(format.clone());

                                log::error!(
                                    "{:?} -> {:?}\t{read}, {write}",
                                    input_format,
                                    format
                                );

                                use StorageTextureAccess as STAcc;

                                let access = match (read, write) {
                                    (false, false) => unreachable!(),
                                    (true, false) => STAcc::ReadOnly,
                                    (false, true) => STAcc::WriteOnly,
                                    (true, true) => STAcc::ReadWrite,
                                };

                                let ty = wgpu::BindingType::StorageTexture {
                                    access,
                                    format,
                                    view_dimension,
                                };

                                (ty, None)
                            }
                        }
                    }
                    naga::TypeInner::Struct { members, span } => {
                        use naga::AddressSpace as Space;

                        let binding_ty =
                            if let Space::Storage { access } = space {
                                let read_only = !access
                                    .contains(naga::StorageAccess::STORE);
                                wgpu::BufferBindingType::Storage { read_only }
                            } else if let Space::Uniform = space {
                                wgpu::BufferBindingType::Uniform
                            } else {
                                unreachable!();
                            };

                        let ty = wgpu::BindingType::Buffer {
                            ty: binding_ty,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        };
                        (ty, None)
                    }
                    naga::TypeInner::Sampler { comparison } => {
                        // binding type would probably depend on format,
                        // so idk where that information will come from
                        let binding_type = if *comparison {
                            wgpu::SamplerBindingType::Comparison
                        } else {
                            wgpu::SamplerBindingType::Filtering
                        };
                        let ty = wgpu::BindingType::Sampler(binding_type);
                        (ty, None)
                    }
                    e => {
                        panic!("unimplemented: {:?}", e);
                    }
                };

                let entry = wgpu::BindGroupLayoutEntry {
                    binding: def.binding.binding,
                    visibility: stages,
                    ty,
                    count,
                };

                entries.push(entry);
            }

            log::error!("group {} - {:?}", group_ix, entries);

            let bindings = defs.into_iter().map(|(g,_)| g).collect();

            result.push(GroupBindings {
                group_ix,
                // layout: bind_group_layout,
                entries,
                bindings,
            });

            expected_group += 1;
        }

        Ok(result)
    }
}

// push constants

#[derive(Debug, Clone)]
pub struct PushConstantEntry {
    pub kind: naga::ScalarKind,
    pub len: usize,
    pub range: std::ops::Range<u32>,
}

impl PushConstantEntry {
    pub fn size(&self) -> u32 {
        self.range.end - self.range.start
    }

    pub fn index_range(&self) -> std::ops::Range<usize> {
        (self.range.start as usize)..(self.range.end as usize)
    }
}

#[derive(Debug, Clone)]
pub struct PushConstants {
    buffer: Vec<u8>,
    pub stages: naga::ShaderStage,
    fields: Vec<(String, PushConstantEntry)>,
}

impl PushConstants {
    pub fn data(&self) -> &[u8] {
        self.buffer.as_slice()
    }

    pub fn to_range(
        &self,
        offset: u32,
        stages: wgpu::ShaderStages,
    ) -> PushConstantRange {
        let start = offset;
        let end = offset + self.buffer.len() as u32;
        dbg!((stages, start..end));

        PushConstantRange {
            stages,
            range: start..end,
        }
    }
    // pub fn write_field_float(
    //     &mut self,
    //     field_name: &str,
    //     data: &[f32],
    // ) -> Option<()> {
    //     todo!();
    // }

    pub fn write_field_bytes(
        &mut self,
        field_name: &str,
        data: &[u8],
    ) -> Option<()> {
        let field_ix = self.fields.iter().position(|(n, _)| n == field_name)?;
        let (_, entry) = &self.fields[field_ix];

        let size = entry.size();

        assert!(data.len() == size as usize);

        self.buffer[entry.index_range()].copy_from_slice(data);

        Some(())
    }

    pub fn from_naga_struct(
        module: &naga::Module,
        s: &naga::TypeInner,
        stages: naga::ShaderStage,
    ) -> Result<Self> {
        if let naga::TypeInner::Struct { members, span } = s {
            let mut buffer = Vec::new();

            let mut fields = Vec::new();

            for mem in members {
                if let Some(name) = &mem.name {
                    let offset = mem.offset;

                    let ty = &module.types[mem.ty];

                    let size = ty.inner.size(&module.constants);

                    let kind = ty.inner.scalar_kind().unwrap();
                    // only supporting 32 bit values for now
                    let len = size as usize / 4;

                    let range = offset..(offset + size);

                    for _ in range.clone() {
                        buffer.push(0u8);
                    }

                    fields.push((
                        name.clone(),
                        PushConstantEntry { kind, len, range },
                    ));
                }
            }

            Ok(PushConstants {
                buffer,
                stages,
                fields,
            })
        } else {
            anyhow::bail!("Expected `TypeInner::Struct`, was: {:?}", s);
        }
    }
}
