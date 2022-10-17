use naga::{Handle, Expression};
use rustc_hash::{FxHashMap, FxHashSet};
use wgpu::*;

use anyhow::{anyhow, bail, Result};
use std::{
    collections::{BTreeMap, HashMap, VecDeque},
    sync::Arc,
};

use crate::{shader::interface::GroupBindings, ResourceId};

use super::{interface::PushConstants, *};

#[derive(Debug)]
pub struct GraphicsPipeline {
    vertex: Arc<VertexShaderInstance>,
    // fragmentshader: Arc<FragmentShaderInstance>,
    pipeline: wgpu::RenderPipeline,
    // fragment_push: Option<PushConstants>,
}

#[derive(Debug)]
pub struct VertexShaderInstance {
    shader: Arc<VertexShader>,
    push_constants: Option<PushConstants>,

    vertex_step_modes: Vec<wgpu::VertexStepMode>,
    vertex_buffer_strides: Vec<u64>,
    vertex_buffer_layouts: Vec<Vec<wgpu::VertexAttribute>>,
}

impl VertexShaderInstance {
    pub fn from_shader_single_buffer(
        shader: &Arc<VertexShader>,
        step_mode: wgpu::VertexStepMode,
    ) -> Self {
        let shader = shader.clone();
        let push_constants = shader.push_constants.clone();

        let mut attributes = Vec::new();
        let mut array_stride = 0;

        for input in shader.vertex_inputs.iter() {
            let attr = input.attribute(array_stride);
            attributes.push(attr);
            array_stride += input.format.size();
        }

        let vertex_step_modes = vec![step_mode];
        let vertex_buffer_strides = vec![array_stride];
        let vertex_buffer_layouts: Vec<Vec<wgpu::VertexAttribute>> =
            vec![attributes];

        VertexShaderInstance {
            shader,
            push_constants,
            vertex_step_modes,
            vertex_buffer_strides,
            vertex_buffer_layouts,
        }
    }
}

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
            shader_location: self.location,
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

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FragmentOutput {
    pub name: String,
    pub location: u32,
    pub format: TextureFormat,
}

#[derive(Debug)]
pub struct FragmentShader {
    pub shader_module: wgpu::ShaderModule,
    pub entry_point: naga::EntryPoint,

    pub attachments: Vec<FragmentOutput>,
    // pub vertex_inputs: Vec<VertexInput>,
    pub group_bindings: Vec<GroupBindings>,
    pub bind_group_layouts: Vec<wgpu::BindGroupLayout>,

    pub push_constants: Option<interface::PushConstants>,
}

impl FragmentShader {
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

        // add fragment outputs

        // let mut fragment_outputs = Vec::new();

        if let Some(result) = entry_point.function.result.as_ref() {
            if let Some(binding) = result.binding.as_ref() {
                let location =
                    if let naga::Binding::Location { location, .. } = binding {
                        *location
                    } else {
                        anyhow::bail!("No output found on fragment shader");
                    };

                let (size, kind, width) = match &naga_mod.types[result.ty].inner
                {
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

                log::warn!(
                    "fragment output `{:?}` - {:?}\t{:?}width {}",
                    "",
                    // result.ty.,
                    size,
                    kind,
                    width
                );

                // if let Some(format) =
                //     naga_to_wgpu_vertex_format(size, kind, width)
                // {
                //     let input = VertexInput {
                //         name: arg.name.clone().unwrap_or_default(),
                //         location,
                //         format,
                //     };
                //     vertex_inputs.push(input);
                // }
            }
        }

        /*
          for arg in entry_point.function.arguments.iter() {
              if let Some(binding) = arg.binding.as_ref() {
                  let location =
                      if let naga::Binding::Location { location, .. } = binding {
                          *location
                      } else {
                          continue;
                      };

                  todo!();
                  let (size, kind, width) = match &naga_mod.types[arg.ty].inner {
                      naga::TypeInner::Scalar { kind, width } => {
                          (None, *kind, *width)
                      }
                      naga::TypeInner::Vector { size, kind, width } => {
                          (Some(*size), *kind, *width)
                      }
                      other => {
                          log::warn!("{:?} - TextureFormat: {:?}", arg.name, )
                          todo!();
                          // panic!("unsupported vertex type: {:?}", other);
                      }
                  };

                  // if let Some(format) =
                  //     naga_to_wgpu_vertex_format(size, kind, width)
                  // {
                  //     let input = VertexInput {
                  //         name: arg.name.clone().unwrap_or_default(),
                  //         location,
                  //         format,
                  //     };
                  //     vertex_inputs.push(input);
                  // }
              }
          }
        */

        // vertex_inputs.sort_by_key(|v| v.location);

        // log::warn!("vertex inputs: {:#?}", vertex_inputs);

        let bind_group_layouts =
            GroupBindings::create_bind_group_layouts_checked(
                &group_bindings,
                state,
            )?;

        todo!();

        // Ok(FragmentShader {
        //     shader_module: (),
        //     entry_point: (),
        //     attachments: (),
        //     group_bindings: (),
        //     bind_group_layouts: (),
        //     push_constants: (),
        // })

        // Ok(VertexShader {
        //     shader_module,
        //     entry_point: entry_point.clone(),
        //     vertex_inputs,
        //     group_bindings,
        //     bind_group_layouts,
        //     push_constants,
        // })
    }
}

// the naga API doesn't provide a direct map between "global variable name"
// and shader location, unlike what it does with vertex inputs, push constants,
// and bind groups, so this is a little datafrog program to deduce that map
//
// returns pairs of global variable names and output shader location indices
pub fn find_fragment_var_location_map(
    module: &naga::Module,
) -> Vec<(String, u32)> {
    use datafrog::*;

    let mut iteration = Iteration::new();

    /*
    globals(global_id, var_name).

    struct_ix_location(mem_ix, location).

    expr_load(expr_id, expr_id).
    expr_global(expr_id, global_id).
    expr_out_ix(mem_ix, expr_id).

    

    example:
    globals(2, "f_color").
    globals(3, "f_norm").
    globals(6, "id_out").

    struct_ix_location(0, 2).
    struct_ix_location(1, 0).
    struct_ix_location(2, 1).

    expr_load(6, 3).
    expr_load(7, 4).
    expr_load(8, 5).

    expr_global(3, 2).
    expr_global(4, 3).
    expr_global(5, 6).

    expr_out_ix(6, 0).
    expr_out_ix(7, 1).
    expr_out_ix(8, 2).

    */

    let globals = iteration.variable::<(usize, String)>("globals");

    let struct_ix_loc = iteration.variable::<(u32, u32)>("struct_ix_loc");

    type Expr = Handle<Expression>;

    let expr_load = iteration.variable::<(Expr, Expr)>("expr_load");
    // let expr_global = iteration.variable::<(u32, u32)>("expr_global");
    let expr_out_ix = iteration.variable::<(u32, Expr)>("out_component");
    let expr_global = iteration.variable::<(Expr, usize)>("expr_global");

    let entry_point = &module.entry_points[0].function;

    // add global variables: global(global_id, var_name)
    for (handle, var) in module.global_variables.iter() {
        if var.space != naga::AddressSpace::Private {
            continue;
        }

        let global_id = handle.index();
        if let Some(var_name) = var.name.as_ref() {
            globals.extend(Some((global_id, var_name.to_string())));
        }
    }

    // the fragment output is a "struct", which is an ordered set of
    // struct members, each corresponding to a type and a shader location.
    // here, we add the relations for the shader location and the index
    // into this ordered set.
    //
    // struct_ix_loc(mem_ix, location).
    if let Some(result) = entry_point.result.as_ref() {
        if let Some(naga::Binding::Location {
            location,
            interpolation: _,
            sampling: _,
        }) = result.binding.as_ref()
        {
            log::warn!("extending with {location}");
            struct_ix_loc.extend(Some((0, *location)));
        } else if let naga::TypeInner::Struct { members, span } =
            &module.types[result.ty].inner
        {
            dbg!(members);
            todo!();
        }
    } else {
        panic!("Fragment shader has no output");
    }

    let return_value_expr_id = entry_point
        .body
        .iter()
        .find_map(|stmt| {
            if let naga::Statement::Return { value } = stmt {
                value.as_ref()
            } else {
                None
            }
        })
        .unwrap();

    // there are some pointer-like expressions in the "outer" entry
    // point function we need
    // basically, we need to map some expressions into new edges
    // NB: i have no idea if this structure is something i can actually rely on,
    // but it should be fine
    for (h, expr) in entry_point.expressions.iter() {
        let expr_id = h;
        log::warn!("Expression: {:?} - {:#?}", h, expr);

        match expr {
            // expr_global(expr_id, global_id)
            naga::Expression::GlobalVariable(handle) => {
                let global_id = handle.index();
                expr_global.extend(Some((expr_id, global_id)));
            }
            // expr_load(expr_id, expr_id)
            naga::Expression::Load { pointer } => {
                expr_load.extend(Some((expr_id, *pointer)));
            }
            // expr_out_ix(mem_ix, expr_id)
            naga::Expression::Compose { ty, components } => {
                for (mem_ix, ptr) in components.iter().enumerate() {
                    expr_out_ix.extend(Some((mem_ix as u32, *ptr)));
                }
            }
            _ => (),
        }
    }

    todo!();
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

fn naga_to_wgpu_texture_format(
    vec_size: Option<naga::VectorSize>,
    kind: naga::ScalarKind,
    width: u8,
) -> Option<wgpu::TextureFormat> {
    use naga::ScalarKind as Kind;
    use naga::VectorSize as Size;
    use wgpu::TextureFormat as Tx;
    match (vec_size, kind, width) {
        // (None, Kind::Float, 4) => Some(Tx::Rgba),
        // (Some(Size::Quad), Kind::Float, 4) => Some(Tx::Float32x2),
        /*
        (None, Kind::Float, 4) => Some(Tx::Rgba),
        (Some(Size::Bi), Kind::Float, 4) => Some(Tx::Float32x2),
        (Some(Size::Tri), Kind::Float, 4) => Some(Tx::Float32x3),
        (Some(Size::Quad), Kind::Float, 4) => Some(Tx::Float32x4),
        (None, Kind::Sint, 4) => Some(Tx::Sint32),
        (Some(Size::Bi), Kind::Sint, 4) => Some(Tx::Sint32x2),
        (Some(Size::Tri), Kind::Sint, 4) => Some(Tx::Sint32x3),
        (Some(Size::Quad), Kind::Sint, 4) => Some(Tx::Sint32x4),
        (None, Kind::Uint, 4) => Some(Tx::Uint32),
        (Some(Size::Bi), Kind::Uint, 4) => Some(Tx::Uint32x2),
        (Some(Size::Tri), Kind::Uint, 4) => Some(Tx::Uint32x3),
        (Some(Size::Quad), Kind::Uint, 4) => Some(Tx::Uint32x4),
        */
        _ => None,
    }
}
