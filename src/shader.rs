use rustc_hash::{FxHashMap, FxHashSet};
use wgpu::*;

use anyhow::{anyhow, bail, Result};
use std::{
    collections::{BTreeMap, HashMap, VecDeque},
    sync::Arc,
};

/*
struct ShaderBindings {
    groups: Vec<Vec<(GVarName, )
}
*/

pub struct ComputeShader {
    pipeline: wgpu::ComputePipeline,

    bind_group_layouts: Vec<wgpu::BindGroupLayout>,

    shader_bindings: Vec<Vec<BindingDef>>,
}

impl ComputeShader {
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

        let mut bind_group_layouts = Vec::new();
        let mut push_constant_ranges = todo!();

        let mut shader_bindings: BTreeMap<u32, Vec<BindingDef>> =
            BTreeMap::default();

        for (handle, var) in module.global_variables.iter() {
            if var.space == naga::AddressSpace::Handle && var.binding.is_some()
            {
                let binding = var.binding.unwrap();
                let ty = &module.types[var.ty];

                let binding_def = BindingDef {
                    global_var_name: var.name.unwrap().into(),
                    binding: binding,
                    ty: ty.inner,
                };

                shader_bindings
                    .entry(binding.group)
                    .or_default()
                    .push(binding_def);
            }
        }

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
            expected_group += 1;
            // let mut group_bindings = Vec::new();
        }

        let pipeline_layout_desc = PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: bind_group_layouts.as_slice(),
            push_constant_ranges: push_constant_ranges,
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

        todo!();
    }
}

struct BindingDef {
    global_var_name: rhai::ImmutableString,
    binding: naga::ResourceBinding,
    ty: naga::TypeInner,
}
