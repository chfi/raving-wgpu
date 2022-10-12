use rustc_hash::{FxHashMap, FxHashSet};
use wgpu::*;

use anyhow::{anyhow, bail, Result};
use std::{
    collections::{BTreeMap, HashMap, VecDeque},
    sync::Arc,
};

use crate::{shader::interface::GroupBindings, ResourceId};

pub mod interface;
// pub mod pushconst;
// pub mod group_bindings;

/*
struct ShaderBindings {
    groups: Vec<Vec<(GVarName, )
}
*/

/*
impl VertexShader {

    pub fn from_spirv(
        state: &super::State,
        shader_src: &[u8],
        entry_point: &str
    ) -> Result<Self> {
        let shader_desc = ShaderModuleDescriptor {
            label: None,
            source: wgpu::util::make_spirv(shader_src),
        };

        let shader_module = state.device.create_shader_module(shader_desc);

        let module = naga::front::spv::parse_u8_slice(
            shader_src,
            &naga::front::spv::Options {
                adjust_coordinate_space: true,
                strict_capabilities: false,
                block_ctx_dump_prefix: None,
            },
        )?;
    }
}
*/

#[derive(Debug)]
pub struct ComputeShader {
    pub pipeline: wgpu::ComputePipeline,

    group_bindings: Vec<GroupBindings>,
    bind_group_layouts: Vec<wgpu::BindGroupLayout>,

    // bind_group_entries: Vec<Vec<BindGroupLayoutEntry>>,
    // shader_bindings: Vec<Vec<BindingDef>>,
}

impl ComputeShader {
    pub fn create_bind_groups_impl(
        &self,
        state: &super::State,
        resources: &[super::graph::Resource],
        // map binding variable names to resource IDs
        resource_map: &HashMap<String, ResourceId>,
    ) -> Result<Vec<wgpu::BindGroup>> {
        let mut bind_groups = Vec::new();

        for bindings in self.group_bindings.iter() {
            let mut entries = Vec::new();

            for (binding_ix, entry) in bindings.entries.iter().enumerate() {

                let def = &bindings.bindings[binding_ix];
                
                let res_id =
                    resource_map.get(def.global_var_name.as_str()).unwrap();

                let resource = &resources[res_id.0];

                match resource {
                    crate::Resource::Buffer {
                        buffer,
                        size,
                        usage,
                    } => {
                        let bind_res = match entry.ty
                        {
                            BindingType::Buffer { .. } => {
                                BindingResource::Buffer(
                                    buffer
                                        .as_ref()
                                        .unwrap()
                                        .as_entire_buffer_binding(),
                                )
                            }
                            _ => {
                                panic!("TODO: Binding type mismatch!");
                            }
                        };

                        let entry = BindGroupEntry {
                            binding: binding_ix as u32,
                            resource: bind_res,
                        };

                        entries.push(entry);
                    }
                    crate::Resource::Texture {
                        texture,
                        size,
                        format,
                        usage,
                    } => {
                        let bind_res = match entry.ty
                        {
                            BindingType::Buffer { .. } => {
                                panic!("TODO: Binding type mismatch!");
                            }
                            BindingType::Sampler(_) => {
                                BindingResource::Sampler(
                                    &texture.as_ref().unwrap().sampler,
                                )
                            }
                            BindingType::Texture { .. } => {
                                BindingResource::TextureView(
                                    &texture.as_ref().unwrap().view,
                                )
                            }
                            BindingType::StorageTexture { .. } => {
                                BindingResource::TextureView(
                                    &texture.as_ref().unwrap().view,
                                )
                            }
                        };

                        let entry = BindGroupEntry {
                            binding: binding_ix as u32,
                            resource: bind_res,
                        };

                        entries.push(entry);
                    }
                }

            }

            let group_ix = bindings.group_ix as usize;
            
            let layout = &self.bind_group_layouts[group_ix];

            let bind_group =
                state.device.create_bind_group(&BindGroupDescriptor {
                    label: None,
                    layout,
                    entries: entries.as_slice(),
                });

            log::error!("group {} - {:?}", group_ix, entries);

            bind_groups.push(bind_group);

        }


        Ok(bind_groups)
    }

    pub fn from_spirv(
        state: &super::State,
        shader_src: &[u8],
        entry_point: &str,
    ) -> Result<Self> {
        let shader_desc = ShaderModuleDescriptor {
            label: None,
            source: wgpu::util::make_spirv(shader_src),
        };

        let shader_module = state.device.create_shader_module(shader_desc);

        let module = naga::front::spv::parse_u8_slice(
            shader_src,
            &naga::front::spv::Options {
                adjust_coordinate_space: true,
                strict_capabilities: false,
                block_ctx_dump_prefix: None,
            },
        )?;

        log::error!("parsing group bindings");
        let group_bindings = GroupBindings::from_spirv(&module)?;
        log::error!("group bindings: {:#?}", group_bindings);

        let mut bind_group_layouts = Vec::new();

        let mut push_constants: Option<interface::PushConstants> = None;

        for (handle, var) in module.global_variables.iter() {
            if var.space == naga::AddressSpace::PushConstant {
                let ty = &module.types[var.ty];
                let push_const = interface::PushConstants::from_naga_struct(
                    &module,
                    &ty.inner,
                    naga::ShaderStage::Compute,
                )?;
                push_constants = Some(push_const);
            }
        }

        let mut expected_group = 0;

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


        let layout_refs = bind_group_layouts.iter().collect::<Vec<_>>();

        let push_constant_ranges = if let Some(p) = push_constants {
            dbg!();
            vec![p.to_range(wgpu::ShaderStages::COMPUTE)]
        } else {
            dbg!();
            vec![]
        };

        let pipeline_layout_desc = PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: layout_refs.as_slice(),
            push_constant_ranges: push_constant_ranges.as_slice(),
        };

        let pipeline_layout =
            state.device.create_pipeline_layout(&pipeline_layout_desc);

        let compute_desc = ComputePipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point,
        };

        let compute_pipeline =
            state.device.create_compute_pipeline(&compute_desc);

        let compute_shader = ComputeShader {
            pipeline: compute_pipeline,

            group_bindings,
            bind_group_layouts,
        };

        Ok(compute_shader)
    }
}

#[derive(Debug, Clone)]
struct BindingDef {
    global_var_name: rhai::ImmutableString,
    binding: naga::ResourceBinding,
    ty: naga::TypeInner,
}

pub fn format_naga_to_wgpu(format: naga::StorageFormat) -> wgpu::TextureFormat {
    match format {
        naga::StorageFormat::R8Unorm => wgpu::TextureFormat::R8Unorm,
        naga::StorageFormat::R8Snorm => wgpu::TextureFormat::R8Snorm,
        naga::StorageFormat::R8Uint => wgpu::TextureFormat::R8Uint,
        naga::StorageFormat::R8Sint => wgpu::TextureFormat::R8Sint,

        naga::StorageFormat::R16Uint => wgpu::TextureFormat::R16Uint,
        naga::StorageFormat::R16Sint => wgpu::TextureFormat::R16Sint,
        naga::StorageFormat::R16Float => wgpu::TextureFormat::R16Float,

        naga::StorageFormat::Rg8Unorm => wgpu::TextureFormat::Rg8Unorm,
        naga::StorageFormat::Rg8Snorm => wgpu::TextureFormat::Rg8Snorm,

        naga::StorageFormat::Rg8Uint => wgpu::TextureFormat::Rg8Uint,
        naga::StorageFormat::Rg8Sint => wgpu::TextureFormat::Rg8Sint,

        naga::StorageFormat::R32Uint => wgpu::TextureFormat::R32Uint,
        naga::StorageFormat::R32Sint => wgpu::TextureFormat::R32Sint,
        naga::StorageFormat::R32Float => wgpu::TextureFormat::R32Float,

        naga::StorageFormat::Rg16Uint => wgpu::TextureFormat::Rg16Uint,
        naga::StorageFormat::Rg16Sint => wgpu::TextureFormat::Rg16Sint,
        naga::StorageFormat::Rg16Float => wgpu::TextureFormat::Rg16Float,

        naga::StorageFormat::Rgba8Unorm => wgpu::TextureFormat::Rgba8Unorm,
        naga::StorageFormat::Rgba8Snorm => wgpu::TextureFormat::Rgba8Snorm,
        naga::StorageFormat::Rgba8Uint => wgpu::TextureFormat::Rgba8Uint,
        naga::StorageFormat::Rgba8Sint => wgpu::TextureFormat::Rgba8Sint,

        naga::StorageFormat::Rgb10a2Unorm => wgpu::TextureFormat::Rgb10a2Unorm,
        naga::StorageFormat::Rg11b10Float => wgpu::TextureFormat::Rg11b10Float,

        naga::StorageFormat::Rg32Uint => wgpu::TextureFormat::Rg32Uint,
        naga::StorageFormat::Rg32Sint => wgpu::TextureFormat::Rg32Sint,
        naga::StorageFormat::Rg32Float => wgpu::TextureFormat::Rg32Float,

        naga::StorageFormat::Rgba16Uint => wgpu::TextureFormat::Rgba16Uint,
        naga::StorageFormat::Rgba16Sint => wgpu::TextureFormat::Rgba16Sint,
        naga::StorageFormat::Rgba16Float => wgpu::TextureFormat::Rgba16Float,

        naga::StorageFormat::Rgba32Uint => wgpu::TextureFormat::Rgba32Uint,
        naga::StorageFormat::Rgba32Sint => wgpu::TextureFormat::Rgba32Sint,
        naga::StorageFormat::Rgba32Float => wgpu::TextureFormat::Rgba32Float,
    }
}
