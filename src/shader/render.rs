use rustc_hash::{FxHashMap, FxHashSet};
use wgpu::*;

use anyhow::{anyhow, bail, Result};
use std::{
    collections::{BTreeMap, HashMap, VecDeque},
    sync::Arc,
};

use crate::{shader::interface::GroupBindings, ResourceId};

use super::*;

pub struct VertexShader {
    pub group_bindings: Vec<GroupBindings>,
    pub bind_group_layouts: Vec<wgpu::BindGroupLayout>,

    push_constants: Option<interface::PushConstants>,
}

impl VertexShader {
    pub fn from_spirv(
        state: &crate::State,
        shader_src: &[u8],
        entry_point: &str,
    ) -> Result<Self> {
        let shader_desc = ShaderModuleDescriptor {
            label: None,
            source: wgpu::util::make_spirv(shader_src),
        };

        let shader_module = state.device.create_shader_module(shader_desc);

        let naga_mod = naga::front::spv::parse_u8_slice(
            shader_src,
            &naga::front::spv::Options {
                adjust_coordinate_space: true,
                strict_capabilities: false,
                block_ctx_dump_prefix: None,
            },
        )?;

        let entry_point = naga_mod
            .entry_points
            .iter()
            .find(|ep| ep.name == entry_point)
            .ok_or_else(|| {
                anyhow::anyhow!("Entry point not found: `{}`", entry_point)
            })?;

        let group_bindings = GroupBindings::from_spirv(&naga_mod)?;

        let push_constants = naga_mod
            .global_variables
            .iter()
            .find_map(|(_handle, var)| {
                (var.space == naga::AddressSpace::PushConstant).then(|| {
                    interface::PushConstants::from_naga_struct(
                        &naga_mod,
                        &naga_mod.types[var.ty].inner,
                        entry_point.stage,
                    )
                })
            })
            .transpose()?;

        for arg in entry_point.function.arguments.iter() {
            if let Some(binding) = arg.binding.as_ref() {
                match binding {
                    naga::Binding::BuiltIn(_) => {
                        todo!();
                        //
                    }
                    naga::Binding::Location {
                        location,
                        interpolation,
                        sampling,
                    } => {
                        todo!();

                        // build the vector input interface here
                    }
                }
            }

            //
        }

        todo!();
    }
}
