use rustc_hash::{FxHashMap, FxHashSet};
use wgpu::*;

use anyhow::{anyhow, bail, Result};
use std::{
    collections::{BTreeMap, HashMap, VecDeque},
    sync::Arc,
};

use crate::{shader::interface::GroupBindings, ResourceId};

pub mod interface;
pub mod render;

#[derive(Debug)]
pub struct ComputeShader {
    pub pipeline: wgpu::ComputePipeline,

    pub group_bindings: Vec<GroupBindings>,
    pub bind_group_layouts: Vec<wgpu::BindGroupLayout>,

    pub workgroup_size: [u32; 3],

    push_constants: Option<interface::PushConstants>,
}

impl ComputeShader {
    pub fn clone_push_constants(&self) -> Option<interface::PushConstants> {
        self.push_constants.clone()
    }

    pub fn create_bind_groups_impl(
        &self,
        state: &super::State,
        resources: &[super::graph::Resource],
        // map binding variable names to resource IDs
        resource_map: &HashMap<String, ResourceId>,
    ) -> Result<Vec<wgpu::BindGroup>> {
        let mut bind_groups = Vec::new();

        for bindings in self.group_bindings.iter() {
            let ix = bindings.group_ix as usize;
            let layout = &self.bind_group_layouts[ix];

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

    pub fn from_wgsl(
        state: &super::State,
        shader_src: &str,
        entry_point: &str,
    ) -> Result<Self> {
        let label = format!("Compute shader");
        let shader_desc = ShaderModuleDescriptor {
            label: Some(label.as_str()),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        };

        let naga_mod = naga::front::wgsl::parse_str(shader_src)?;

        let shader_module = state.device.create_shader_module(shader_desc);

        log::error!("parsing group bindings");
        let group_bindings =
            GroupBindings::from_spirv(&naga_mod, wgpu::ShaderStages::COMPUTE)?;
        log::error!("group bindings: {:#?}", group_bindings);

        let (workgroup_size, stage) = {
            let entry_point = naga_mod
                .entry_points
                .iter()
                .next()
                .ok_or(anyhow!("shader entry point missing"))?;

            (entry_point.workgroup_size, entry_point.stage)
        };

        let push_constants = naga_mod
            .global_variables
            .iter()
            .find_map(|(_handle, var)| {
                (var.space == naga::AddressSpace::PushConstant).then(|| {
                    interface::PushConstants::from_naga_struct(
                        &naga_mod,
                        &naga_mod.types[var.ty].inner,
                        stage,
                    )
                })
            })
            .transpose()?;

        let bind_group_layouts =
            GroupBindings::create_bind_group_layouts_checked(
                &group_bindings,
                state,
            )?;

        let layout_refs = bind_group_layouts.iter().collect::<Vec<_>>();

        let push_constant_ranges = if let Some(p) = &push_constants {
            dbg!();
            vec![p.to_range(0, wgpu::ShaderStages::COMPUTE)]
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
            workgroup_size,

            push_constants,
        };

        Ok(compute_shader)
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
        let group_bindings =
            GroupBindings::from_spirv(&module, wgpu::ShaderStages::COMPUTE)?;
        log::error!("group bindings: {:#?}", group_bindings);

        let (workgroup_size, stage) = {
            let entry_point = module
                .entry_points
                .iter()
                .next()
                .ok_or(anyhow!("shader entry point missing"))?;

            (entry_point.workgroup_size, entry_point.stage)
        };

        let push_constants = module
            .global_variables
            .iter()
            .find_map(|(_handle, var)| {
                (var.space == naga::AddressSpace::PushConstant).then(|| {
                    interface::PushConstants::from_naga_struct(
                        &module,
                        &module.types[var.ty].inner,
                        stage,
                    )
                })
            })
            .transpose()?;

        let bind_group_layouts =
            GroupBindings::create_bind_group_layouts_checked(
                &group_bindings,
                state,
            )?;

        let layout_refs = bind_group_layouts.iter().collect::<Vec<_>>();

        let push_constant_ranges = if let Some(p) = &push_constants {
            dbg!();
            vec![p.to_range(0, wgpu::ShaderStages::COMPUTE)]
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
            workgroup_size,

            push_constants,
        };

        Ok(compute_shader)
    }
}

#[derive(Debug, Clone)]
pub(crate) struct BindingDef {
    pub(crate) global_var_name: rhai::ImmutableString,
    pub(crate) binding: naga::ResourceBinding,
    pub(crate) ty: naga::TypeInner,
}
