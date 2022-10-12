use std::collections::BTreeMap;

use anyhow::Result;
use wgpu::{PushConstantRange, StorageTextureAccess};

use super::BindingDef;

// uniforms...

// bind groups

#[derive(Debug)]
pub struct GroupBindings {
    pub group_ix: u32,

    // when creating a pipeline layout, the bind groups need
    // to be provided as a slice, which isn't possible with this approach!
    // since you'd have to destroy the GroupBindings to get the layout out...
    // annoying
    //
    // i guess i can store the layouts in a hashmap keyed by layout entries
    // it *would* be good to have this type work more as a description
    // that can be serialized
    pub layout: wgpu::BindGroupLayout,
    pub entries: Vec<wgpu::BindGroupLayoutEntry>,
    pub(super) bindings: Vec<super::BindingDef>,
}

impl GroupBindings {

    /*
    pub fn create_bind_groups(&self,
        state: &super::State,
        resources: &[super::graph::Resource],
        resource_map: &HashMap<String, ResourceId>,
    ) -> Result<Vec<wgpu::BindGroup>> {
        todo!();
    }
    */

    pub fn from_spirv(
        state: &crate::State,
        module: &naga::Module,
    ) -> Result<Vec<Self>> {
        let mut result = Vec::new();

        let mut shader_bindings: BTreeMap<u32, Vec<BindingDef>> =
            BTreeMap::default();

        for (handle, var) in module.global_variables.iter() {
            // if let naga::AddressSpace::Storage { access } = var.space {
            //     if var.binding_is.some() {
            //     }
            // }

            if (matches!(var.space, naga::AddressSpace::Storage { .. })
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
                    .push(binding_def);
            }

            // log::error!("var.space: {:?}", var.space);
            // match var.space {
            // }
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

            defs.sort_by_key(|def| def.binding.binding);
            final_bindings.push(defs.clone());

            let mut entries: Vec<wgpu::BindGroupLayoutEntry> = Vec::new();

            for (binding_ix, def) in defs.iter().enumerate() {
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
                    } => match class {
                        naga::ImageClass::Depth { multi } => {
                            panic!("unimplemented!");
                        }
                        naga::ImageClass::Sampled { kind, multi } => {
                            panic!("unimplemented!");
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

                            let access = match (read, write) {
                                (false, false) => unreachable!(),
                                (true, false) => StorageTextureAccess::ReadOnly,
                                (false, true) => {
                                    StorageTextureAccess::WriteOnly
                                }
                                (true, true) => StorageTextureAccess::ReadWrite,
                            };

                            // let mut access = StorageTextureAccess::

                            let view_dimension = match dim {
                                naga::ImageDimension::D1 => {
                                    wgpu::TextureViewDimension::D1
                                }
                                naga::ImageDimension::D2 => {
                                    wgpu::TextureViewDimension::D2
                                }
                                naga::ImageDimension::D3 => {
                                    wgpu::TextureViewDimension::D3
                                }
                                naga::ImageDimension::Cube => {
                                    wgpu::TextureViewDimension::Cube
                                }
                            };

                            let ty = wgpu::BindingType::StorageTexture {
                                access,
                                format,
                                view_dimension,
                            };

                            (ty, None)
                        }
                    },
                    naga::TypeInner::Struct { members, span } => {
                        let ty = wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage {
                                read_only: false,
                            },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        };
                        (ty, None)
                    }
                    // naga::TypeInner::
                    e => {
                        panic!("unimplemented: {:?}", e);
                    }
                };

                let entry = wgpu::BindGroupLayoutEntry {
                    binding: def.binding.binding,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty,
                    count,
                };

                entries.push(entry);
            }

            // create bind group layout and store definition

            let bind_group_layout = state.device.create_bind_group_layout(
                &wgpu::BindGroupLayoutDescriptor {
                    // label: Some("test bind group layout"),
                    label: None,
                    entries: entries.as_slice(),
                },
            );

            log::error!("group {} - {:?}", group_ix, entries);

            result.push(GroupBindings {
                group_ix,
                layout: bind_group_layout,
                entries,
                bindings: defs,
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
    pub fn to_range(&self, stages: wgpu::ShaderStages) -> PushConstantRange {
        let size = self.buffer.len();

        PushConstantRange {
            stages: wgpu::ShaderStages::COMPUTE,
            range: 0..size as u32,
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

        None
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
