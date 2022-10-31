use naga::{Expression, Handle, ScalarKind, VectorSize};
use rustc_hash::{FxHashMap, FxHashSet};
use wgpu::*;

use anyhow::{anyhow, bail, Result};
use std::{
    collections::{BTreeMap, HashMap, VecDeque},
    iter,
    sync::Arc,
};

use crate::{shader::interface::GroupBindings, ResourceId};

use super::{interface::PushConstants, *};

#[derive(Debug)]
pub struct GraphicsPipeline {
    pub vertex: VertexShaderInstance,
    pub fragment: FragmentShaderInstance,

    pub pipeline_layout: wgpu::PipelineLayout,
    pub pipeline: wgpu::RenderPipeline,
    // fragment_push: Option<PushConstants>,
}

impl GraphicsPipeline {
    // NB: only supports disjoint ranges for vertex and fragment for now
    /*
    pub fn push_constant_ranges(&self) -> Vec<wgpu::PushConstantRange> {
        let vx = self.vertex.push_constants.as_ref();
        let fg = self.fragment.push_constants.as_ref();

        use wgpu::ShaderStages as S;

        [(S::VERTEX, vx), (S::FRAGMENT, fg)]
            .into_iter()
            .filter_map(|(stage, pc)| Some(pc?.to_range(stage)))
            .collect()
    }
    */

    pub fn create_bind_groups(&self, state: &crate::State) {
        todo!();
    }

    pub fn new(
        state: &crate::State,
        vertex: VertexShaderInstance,
        fragment: FragmentShaderInstance,
    ) -> Result<Self> {
        let vertex_buffers = vertex.create_buffer_layouts();
        let vertex_state = vertex.create_vertex_state(&vertex_buffers);

        let mut color_targets = Vec::new();
        let fragment_state = {
            // TODO blend and write_mask should be configurable

            for format in fragment.attachment_formats.iter() {
                let state = wgpu::ColorTargetState {
                    format: *format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                };
                color_targets.push(Some(state));
            }

            let state = wgpu::FragmentState {
                module: &fragment.shader.shader_module,
                entry_point: fragment.shader.entry_point.name.as_str(),
                targets: color_targets.as_slice(),
            };

            state
        };

        let pipeline_layout = {
            let stages = [
                (&vertex.shader.push_constants, wgpu::ShaderStages::VERTEX),
                (
                    &fragment.shader.push_constants,
                    wgpu::ShaderStages::FRAGMENT,
                ),
            ];

            let mut offset = 0;
            let mut ranges = Vec::new();

            for (consts, stage) in stages {
                if let Some(consts) = consts.as_ref() {
                    let range = consts.to_range(offset, stage);
                    dbg!(stage, offset, &range.range);
                    offset += range.range.end - range.range.start;
                    ranges.push(range);
                }
            }

            let stages = [
                vertex.shader.bind_group_layouts.as_slice(),
                fragment.shader.bind_group_layouts.as_slice(),
            ];

            let layouts = stages
                .into_iter()
                .filter(|s| !s.is_empty())
                .flatten()
                .collect::<Vec<_>>();

            let desc = wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: layouts.as_slice(),
                push_constant_ranges: ranges.as_slice(),
            };

            state.device.create_pipeline_layout(&desc)
        };

        let primitive = wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            // topology: wgpu::PrimitiveTopology::LineList,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: None,
            polygon_mode: wgpu::PolygonMode::Fill,

            strip_index_format: None,
            unclipped_depth: false,
            conservative: false,
        };

        let multisample = wgpu::MultisampleState {
            count: 1,
            mask: !0,
            alpha_to_coverage_enabled: false,
        };

        let pipeline_desc = wgpu::RenderPipelineDescriptor {
            label: Some("render pipeline"),

            layout: Some(&pipeline_layout),
            vertex: vertex_state,
            fragment: Some(fragment_state),

            primitive,
            depth_stencil: None,

            multisample,
            multiview: None,
        };

        let pipeline = state.device.create_render_pipeline(&pipeline_desc);

        let result = GraphicsPipeline {
            vertex,
            fragment,

            pipeline_layout,
            pipeline,
        };

        Ok(result)
    }
}

#[derive(Debug)]
pub struct VertexShaderInstance {
    pub(crate) shader: Arc<VertexShader>,
    // pub push_constants: Option<PushConstants>,
    pub vertex_step_modes: Vec<wgpu::VertexStepMode>,
    pub vertex_buffer_strides: Vec<u64>,
    pub vertex_buffer_layouts: Vec<Vec<wgpu::VertexAttribute>>,
}

impl VertexShaderInstance {
    // must use the buffer layouts produced by calling `create_buffer_layout`
    // on this instance
    pub fn create_vertex_state<'a>(
        &'a self,
        buffer_layouts: &'a [wgpu::VertexBufferLayout<'a>],
    ) -> wgpu::VertexState<'a> {
        wgpu::VertexState {
            module: &self.shader.shader_module,
            entry_point: self.shader.entry_point.name.as_str(),
            buffers: buffer_layouts,
        }
    }

    pub fn create_buffer_layouts(&self) -> Vec<wgpu::VertexBufferLayout<'_>> {
        let mut vertex_buffers = Vec::new();
        for (ix, attrs) in self.vertex_buffer_layouts.iter().enumerate() {
            let array_stride = self.vertex_buffer_strides[ix];
            let step_mode = self.vertex_step_modes[ix];

            let buffer = wgpu::VertexBufferLayout {
                array_stride,
                step_mode,
                attributes: attrs.as_slice(),
            };
            vertex_buffers.push(buffer);
        }
        vertex_buffers
    }

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
            // push_constants,
            vertex_step_modes,
            vertex_buffer_strides,
            vertex_buffer_layouts,
        }
    }
}

#[derive(Debug)]
pub struct FragmentShaderInstance {
    pub shader: Arc<FragmentShader>,

    attachment_formats: Vec<wgpu::TextureFormat>,
    depth_format: Option<wgpu::TextureFormat>,
    // pub push_constants: Option<interface::PushConstants>,
}

impl FragmentShaderInstance {
    // attch_formats must be in location order and
    // match the fragment shader outputs
    pub fn from_shader(
        shader: &Arc<FragmentShader>,
        attch_formats: &[wgpu::TextureFormat],
    ) -> Result<Self> {
        // let mut attchs = Vec::new();

        let attchs = attch_formats.iter().copied().collect::<Vec<_>>();

        /*
        for output in shader.fragment_outputs.iter() {
            let ix = output.location as usize;
            let format = attch_formats[ix];

            let sizes = output.sizes;
        }

        let attachment_formats: Vec<_> = attch_formats
            .iter()
            .map(|fmt| {
                todo!();
            })
            .collect();
        */

        let result = Self {
            shader: shader.clone(),

            attachment_formats: attchs,
            depth_format: None,
            // push_constants: shader.push_constants.clone(),
        };

        Ok(result)
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

        let group_bindings =
            GroupBindings::from_spirv(&naga_mod, wgpu::ShaderStages::VERTEX)?;

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
    // pub format: TextureFormat,
    pub sizes: (Option<VectorSize>, ScalarKind, u8),
}

#[derive(Debug)]
pub struct FragmentShader {
    pub shader_module: wgpu::ShaderModule,
    pub entry_point: naga::EntryPoint,

    pub fragment_outputs: Vec<FragmentOutput>,
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

        let group_bindings =
            GroupBindings::from_spirv(&naga_mod, wgpu::ShaderStages::FRAGMENT)?;

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

        let mut fragment_outputs = Vec::new();

        let var_name_locations = find_fragment_var_location_map(&naga_mod);

        let find_name_by_loc = |loc: u32| -> Option<&str> {
            var_name_locations
                .iter()
                .find_map(|(name, l)| (*l == loc).then_some(name.as_str()))
        };

        if let Some(result) = entry_point.function.result.as_ref() {
            dbg!();
            if let Some(binding) = result.binding.as_ref() {
                dbg!();
                let location =
                    if let naga::Binding::Location { location, .. } = binding {
                        *location
                    } else {
                        anyhow::bail!("No output found on fragment shader");
                    };

                let sizes =
                    naga_type_inner_sizes(&naga_mod.types[result.ty].inner)
                        .expect("unsupported fragment output type");

                let name = find_name_by_loc(location).unwrap();
                let output = FragmentOutput {
                    name: name.into(),
                    location,
                    sizes,
                };
                fragment_outputs.push(output);
            } else {
                let result_type = &naga_mod.types[result.ty].inner;

                if let naga::TypeInner::Struct { members, span } = &result_type
                {
                    for (ix, mem) in members.iter().enumerate() {
                        let location = mem
                            .binding
                            .as_ref()
                            .and_then(|binding| {
                                if let naga::Binding::Location {
                                    location,
                                    ..
                                } = binding
                                {
                                    Some(*location)
                                } else {
                                    None
                                }
                            })
                            .expect(
                                "location missing from fragment shader output",
                            );

                        let sizes = naga_type_inner_sizes(
                            &naga_mod.types[mem.ty].inner,
                        )
                        .expect("unsupported fragment output type");

                        let name = find_name_by_loc(location).unwrap();
                        let output = FragmentOutput {
                            name: name.into(),
                            location,
                            sizes,
                        };
                        fragment_outputs.push(output);
                    }
                }
                log::warn!("result_type: {:#?}", result_type);
            }
        }

        let bind_group_layouts =
            GroupBindings::create_bind_group_layouts_checked(
                &group_bindings,
                state,
            )?;

        log::warn!("fragment_outputs: {:#?}", fragment_outputs);

        Ok(FragmentShader {
            shader_module,
            entry_point: entry_point.clone(),
            fragment_outputs,
            group_bindings,
            bind_group_layouts,
            push_constants,
        })
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

    type Expr = Handle<Expression>;
    type Global = usize;

    let globals = iteration.variable::<(Global, String)>("globals");

    let expr_load = iteration.variable::<(Expr, Expr)>("expr_load");
    let expr_struct_load =
        iteration.variable::<(Expr, (usize, Expr))>("expr_struct_load");
    let expr_global = iteration.variable::<(Expr, Global)>("expr_global");

    let result_ix_location =
        iteration.variable::<(usize, u32)>("result_ix_location");

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
    // result_ix_location(mem_ix, location).
    if let Some(result) = entry_point.result.as_ref() {
        if let Some(naga::Binding::Location {
            location,
            interpolation: _,
            sampling: _,
        }) = result.binding.as_ref()
        {
            log::warn!("extending with {location}");
            result_ix_location.extend(Some((0, *location)));
        } else if let naga::TypeInner::Struct { members, span } =
            &module.types[result.ty].inner
        {
            result_ix_location.extend(members.iter().enumerate().map(
                |(mem_ix, member)| {
                    if let Some(naga::Binding::Location {
                        location,
                        interpolation,
                        sampling,
                    }) = member.binding.as_ref()
                    {
                        (mem_ix, *location)
                    } else {
                        unreachable!();
                    }
                },
            ));
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
        .expect("Fragment shader returns nothing");

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
                    expr_struct_load.extend(Some((expr_id, (mem_ix, *ptr))));
                }
            }
            _ => (),
        }
    }

    // the expression ID that is returned by the shader function
    let return_expr_rel =
        Relation::from_iter(Some(return_value_expr_id.clone()));
    let return_expr = iteration.variable("return_expr");
    return_expr.extend(return_expr_rel.elements.iter().map(|i| (*i, ())));

    let util_struct_load =
        iteration.variable::<(Expr, (usize, Expr))>("util_struct_load");

    let global_return =
        iteration.variable::<(Expr, (usize, Global))>("global_return");
    let global_loc = iteration.variable::<(Global, u32)>("global_loc");

    let global_result_ix =
        iteration.variable::<(usize, Global)>("global_result_ix");

    let result = iteration.variable::<(Global, (String, u32))>("result");

    while iteration.changed() {
        /* in a nutshell:

            result(global_id, name, location)
              :- return_expr(return_id),
                 expr_struct_load(return_id, (ix, ptr)),
                 expr_global(ptr, global_id),
                 global(global_id, name),
                 result_ix_location(ix, location).
        */

        /* util_struct_load(dst, (0, return_id))
            :- return_expr(return_id),  expr_load(return_id, dst).
        */
        util_struct_load.from_join(
            &return_expr,
            &expr_load,
            |return_id, _, dst| (*dst, (0, *return_id)),
        );

        /* util_struct_load(ptr, (ix, return_id))
          :- expr_struct_load(return_id, (ix, ptr))
        */
        util_struct_load
            .from_map(&expr_struct_load, |(return_id, (ix, ptr))| {
                (*ptr, (*ix, *return_id))
            });

        /*  util_struct_load(dst, (ix, return_id))
              :- util_struct_load(ptr, (ix, return_id)),
                 expr_load(ptr, dst).
        */
        util_struct_load.from_join(
            &util_struct_load,
            &expr_load,
            |ptr, (ix, return_id), dst| (*dst, (*ix, *return_id)),
        );

        /*
           global_return(return_id, (ix, global_id))
             :- util_struct_load(ptr, (ix, return_id)),
                expr_global(ptr, global_id).
        */
        global_return.from_join(
            &util_struct_load,
            &expr_global,
            |ptr, (ix, return_id), global_id| (*return_id, (*ix, *global_id)),
        );

        /*  global_result_ix(ix, global_id)
              :- global_return(return_id, (ix, global_id)),
                 return_expr(return_id).
        */
        global_result_ix.from_join(
            &global_return,
            &return_expr,
            |return_id, (ix, global_id), _| (*ix, *global_id),
        );

        /*  global_loc(global_id, location)
              :- global_result_ix(ix, global_id),
                 result_ix_location
        */

        global_loc.from_join(
            &global_result_ix,
            &result_ix_location,
            |ix, g_id, loc| (*g_id, *loc),
        );

        result.from_join(&globals, &global_loc, |g_id, name, loc| {
            (*g_id, (name.to_string(), *loc))
        });
    }

    let result = result.complete();

    let result = result
        .elements
        .into_iter()
        .map(|(_, s)| s)
        .collect::<Vec<_>>();

    dbg!();

    result
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

fn naga_type_inner_sizes(
    inner: &naga::TypeInner,
) -> Option<(Option<VectorSize>, naga::ScalarKind, u8)> {
    match inner {
        naga::TypeInner::Scalar { kind, width } => Some((None, *kind, *width)),
        naga::TypeInner::Vector { size, kind, width } => {
            Some((Some(*size), *kind, *width))
        }
        other => None,
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
        (None, Kind::Uint, 4) => Some(Tx::R32Uint),
        (Some(Size::Bi), Kind::Uint, 4) => Some(Tx::Rg32Uint),
        (Some(Size::Quad), Kind::Uint, 4) => Some(Tx::Rgba32Uint),
        (None, Kind::Sint, 4) => Some(Tx::R32Sint),
        (Some(Size::Bi), Kind::Sint, 4) => Some(Tx::Rg32Sint),
        (Some(Size::Quad), Kind::Sint, 4) => Some(Tx::Rgba32Sint),
        // (None, Kind::Float, 4) => Some(Tx::R32Float),
        // (Some(Size::Bi), Kind::Float, 4) => Some(Tx::Rg32Float),
        // (Some(Size::Quad), Kind::Float, 4) => Some(Tx::Rgba32Float),
        (None, Kind::Float, 4) => Some(Tx::R8Unorm),
        (Some(Size::Bi), Kind::Float, 4) => Some(Tx::Rg8Unorm),
        // (Some(Size::Quad), Kind::Float, 4) => Some(Tx::Rgba8Unorm),
        (Some(Size::Quad), Kind::Float, 4) => Some(Tx::Bgra8UnormSrgb),
        _ => None,
    }
}
