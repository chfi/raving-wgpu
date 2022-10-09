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

            defs.sort_by_key(|def| def.binding.binding);

            let mut entries: Vec<BindGroupLayoutEntry> = Vec::new();

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

                let (ty, count) = match def.ty {
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

                            let read = access.contains(naga::StorageAccess::LOAD);
                            let write = access.contains(naga::StorageAccess::STORE);

                            let format = format_naga_to_wgpu(format);

                            let access = match (read, write) {
                                (false, false) => unreachable!(),
                                (true, false) => StorageTextureAccess::ReadOnly,
                                (false, true) => StorageTextureAccess::WriteOnly,
                                (true, true) => StorageTextureAccess::ReadWrite,
                            };

                            // let mut access = StorageTextureAccess::

                            let view_dimension = match dim {
                                naga::ImageDimension::D1 => wgpu::TextureViewDimension::D1,
                                naga::ImageDimension::D2 => wgpu::TextureViewDimension::D2,
                                naga::ImageDimension::D3 => wgpu::TextureViewDimension::D3,
                                naga::ImageDimension::Cube => wgpu::TextureViewDimension::Cube,
                            };

                            let ty = wgpu::BindingType::StorageTexture {
                                access,
                                format,
                                view_dimension,
                            };

                            (ty, None)
                        }
                    },
                    e => {
                        panic!("unimplemented: {:?}", e);
                    }
                };

                let entry = BindGroupLayoutEntry {
                    binding: def.binding.binding,
                    visibility: ShaderStages::COMPUTE,
                    ty,
                    count,
                };

                entries.push(entry);
                //
            }

            // create bind group layout and store definition

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
