use rustc_hash::{FxHashMap, FxHashSet};
use wgpu::*;

use anyhow::{anyhow, bail, Result};
use std::{
    collections::{BTreeMap, HashMap, VecDeque},
    sync::Arc,
};

use crate::{shader::interface::GroupBindings, ResourceId};

use super::*;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct VertexInput {
    pub name: String,
    pub location: u32,
    pub format: VertexFormat,
}

impl VertexInput {
    pub fn size(&self) -> u64 {
        self.format.size()
    }

    pub fn attribute(&self, offset: u64) -> wgpu::VertexAttribute {
        wgpu::VertexAttribute {
            format: self.format,
            offset,
            shader_location: self.location
        }
    }
}

#[derive(Debug)]
pub struct VertexShader {
    pub shader_module: wgpu::ShaderModule,
    pub entry_point: naga::EntryPoint,

    pub vertex_inputs: Vec<VertexInput>,

    pub group_bindings: Vec<GroupBindings>,
    pub bind_group_layouts: Vec<wgpu::BindGroupLayout>,

    pub push_constants: Option<interface::PushConstants>,
}

impl VertexShader {
    /* 
    /// Creates a `wgpu::VertexState` that assumes all the vertex
    /// attributes are stored tightly packed in a single buffer,
    /// in `location` order.
    pub fn into_vertex_state<'a>(
        &'a self,
        step_mode: wgpu::VertexStepMode,
        //
    ) -> Result<wgpu::VertexState<'a>> {

        let mut attributes = Vec::new();
        let mut array_stride = 0;

        for input in self.vertex_inputs.iter() {
            let attr = wgpu::VertexAttribute {
                format: input.format,
                offset: array_stride,
                shader_location: input.location,
            };
            attributes.push(attr);
            array_stride += input.format.size();
        }

        let layout = wgpu::VertexBufferLayout {
            array_stride,
            step_mode,
            attributes: attributes.as_slice(),
        };

        let vx_state = wgpu::VertexState {
            module: &self.shader_module,
            entry_point: &self.entry_point.name,
            buffers: &[layout],
        };

        Ok(vx_state)
    }
    */

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

        // add vertex inputs

        let mut vertex_inputs = Vec::new();

        for arg in entry_point.function.arguments.iter() {
            if let Some(binding) = arg.binding.as_ref() {
                let location =
                    if let naga::Binding::Location { location, .. } = binding {
                        *location
                    } else {
                        continue;
                    };

                let (size, kind, width) = match &naga_mod.types[arg.ty].inner {
                    naga::TypeInner::Scalar { kind, width } => {
                        (None, *kind, *width)
                    }
                    naga::TypeInner::Vector { size, kind, width } => {
                        (Some(*size), *kind, *width)
                    }
                    other => {
                        panic!("unsupported vertex type: {:?}", other);
                    }
                };

                if let Some(format) =
                    naga_to_wgpu_vertex_format(size, kind, width)
                {
                    let input = VertexInput {
                        name: arg.name.clone().unwrap_or_default(),
                        location,
                        format,
                    };
                    vertex_inputs.push(input);
                }
            }
        }

        vertex_inputs.sort_by_key(|v| v.location);

        log::warn!("vertex inputs: {:#?}", vertex_inputs);

        let bind_group_layouts =
            GroupBindings::create_bind_group_layouts_checked(
                &group_bindings,
                state,
            )?;

        Ok(VertexShader {
            shader_module,
            entry_point: entry_point.clone(),
            vertex_inputs,
            group_bindings,
            bind_group_layouts,
            push_constants,
        })
    }
}

fn naga_to_wgpu_vertex_format(
    vec_size: Option<naga::VectorSize>,
    kind: naga::ScalarKind,
    width: u8,
) -> Option<wgpu::VertexFormat> {
    use naga::ScalarKind as Kind;
    use naga::VectorSize as Size;
    use wgpu::VertexFormat as Vx;
    match (vec_size, kind, width) {
        (None, Kind::Float, 4) => Some(Vx::Float32),
        (Some(Size::Bi), Kind::Float, 4) => Some(Vx::Float32x2),
        (Some(Size::Tri), Kind::Float, 4) => Some(Vx::Float32x3),
        (Some(Size::Quad), Kind::Float, 4) => Some(Vx::Float32x4),
        (None, Kind::Sint, 4) => Some(Vx::Sint32),
        (Some(Size::Bi), Kind::Sint, 4) => Some(Vx::Sint32x2),
        (Some(Size::Tri), Kind::Sint, 4) => Some(Vx::Sint32x3),
        (Some(Size::Quad), Kind::Sint, 4) => Some(Vx::Sint32x4),
        (None, Kind::Uint, 4) => Some(Vx::Uint32),
        (Some(Size::Bi), Kind::Uint, 4) => Some(Vx::Uint32x2),
        (Some(Size::Tri), Kind::Uint, 4) => Some(Vx::Uint32x3),
        (Some(Size::Quad), Kind::Uint, 4) => Some(Vx::Uint32x4),
        _ => None,
    }
}
