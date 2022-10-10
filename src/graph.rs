use rustc_hash::{FxHashMap, FxHashSet};
use wgpu::*;

use anyhow::{anyhow, bail, Result};
use std::{
    collections::{HashMap, VecDeque},
    sync::Arc,
};

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct NodeId(usize);

pub type SocketIx = usize;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DataType {
    Buffer,
    Image,
    // Scalar,
}

pub enum Resource {
    Buffer {
        buffer: Option<wgpu::Buffer>,
        size: Option<usize>,
        usage: BufferUsages,
    },
    Texture {
        texture: Option<crate::texture::Texture>,
        size: Option<[u32; 2]>,
        format: Option<TextureFormat>,
        usage: TextureUsages,
    },
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct ResourceId(pub usize);

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ResourceHandle {
    id: ResourceId,
    time: u64,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct TextureId(usize);

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct BufferId(usize);

/*
pub struct NodeContext<'a> {
    id: NodeId,
    // inputs:
}
*/

#[derive(Clone)]
pub enum OutputSource<T> {
    InputPassthrough {
        input: InputName,
    },
    Allocate {
        allocate: Arc<dyn Fn(&Graph<T>, NodeId) -> Result<Resource>>,
    },
    Ref {
        resource: ResourceId,
    },
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LocalSocketRef {
    Input { socket_name: InputName },
    Output { socket_name: OutputName }
}

#[derive(Clone)]
pub struct OutputSocket<T> {
    // name: OutputName,
    ty: DataType,
    link: Option<(NodeId, InputName)>,

    source: OutputSource<T>,
    resource: Option<ResourceHandle>,
}

#[derive(Clone)]
pub struct InputSocket {
    // name: InputName,
    ty: DataType,
    link: Option<(NodeId, OutputName)>,

    resource: Option<ResourceHandle>,
}

impl LocalSocketRef {
    pub fn input(socket_name: &str) -> Self {
        Self::Input { socket_name: socket_name.into() }
    }
    
    pub fn output(socket_name: &str) -> Self {
        Self::Output { socket_name: socket_name.into() }
    }
}

#[derive(Clone)]
pub struct Node_<T> {
    id: NodeId,
    inputs: HashMap<InputName, InputSocket>,
    outputs: HashMap<OutputName, OutputSocket<T>>,

    is_prepared: bool,
    is_ready: bool,
    data: T,
}

trait ExecuteNode {
    fn execute<T>(
        &self,
        graph: &Graph<T>,
        cmd: &mut CommandEncoder,
    ) -> Result<()>;
}

#[derive(Default, Clone, Copy)]
struct NodeNoOp;

impl ExecuteNode for NodeNoOp {
    fn execute<T>(
        &self,
        _graph: &Graph<T>,
        _cmd: &mut CommandEncoder,
    ) -> Result<()> {
        Ok(())
    }
}

#[derive(Default, Clone)]
struct NodeComputeOp {
    compute_pipeline: usize,
    bind_group_layout: usize,

    bind_group_framework: (),

    /// Mapping from binding names in the shader (each 
    /// corresponding to a group and binding index for a bind group)
    /// to input/output sockets on the node instance
    binding_socket_map: HashMap<rhai::ImmutableString, LocalSocketRef>,
}

impl ExecuteNode for NodeComputeOp {
    fn execute<T>(
        &self,
        graph: &Graph<T>,
        cmd: &mut CommandEncoder,
    ) -> Result<()> {
        todo!();
    }
}

#[derive(Default)]
pub struct Graph<T> {
    nodes: Vec<Node_<T>>,

    resources: Vec<Resource>,

    compute_pipelines: Vec<wgpu::ComputePipeline>,
    bind_group_layouts: Vec<BindGroupDef>,
    bind_groups: Vec<wgpu::BindGroup>,

    graph_inputs: rhai::Map,
}

impl<T> Graph<T> {
    pub fn add_node(&mut self, data: T) -> NodeId {
        let id = NodeId(self.nodes.len());
        let node = Node_ {
            id,

            inputs: HashMap::default(),
            outputs: HashMap::default(),

            is_prepared: false,
            is_ready: false,
            data,
        };

        self.nodes.push(node);
        id
    }

    /// returns an ordering of the nodes the `terminus` node depends on,
    /// that can be used to initialize and execute the graph so that
    /// `terminus` can be evaluated
    pub fn resolution_order(&self, terminus: NodeId) -> Vec<NodeId> {
        let mut output = Vec::new();

        #[derive(Clone, Copy, PartialEq, Eq)]
        enum Stage {
            Visit(NodeId),
            Emit(NodeId),
        }

        let mut stack = VecDeque::new();

        // let mut tmp: FxHashSet<NodeId> = FxHashSet::default();
        let mut visited: FxHashSet<NodeId> = FxHashSet::default();

        stack.push_back(Stage::Visit(terminus));

        while let Some(stage) = stack.pop_back() {
            match stage {
                Stage::Visit(current) => {
                    if visited.contains(&current) {
                        continue;
                    }

                    visited.insert(current);

                    stack.push_back(Stage::Emit(current));

                    for (other, other_out, this_in) in
                        self.node_inputs_iter(current)
                    {
                        stack.push_back(Stage::Visit(other));
                    }
                }
                Stage::Emit(current) => {
                    output.push(current);
                }
            }
        }

        output
    }

    pub fn node_inputs_iter<'a>(
        &'a self,
        id: NodeId,
    ) -> impl Iterator<Item = (NodeId, &'a OutputName, &'a InputName)> {
        let node = &self.nodes[id.0];

        node.inputs.iter().filter_map(|(self_input, input_socket)| {
            let (from, from_output) = input_socket.link.as_ref()?;
            Some((*from, from_output, self_input))
        })
    }

    pub fn node_outputs_iter<'a>(
        &'a self,
        id: NodeId,
    ) -> impl Iterator<Item = (NodeId, &'a OutputName, &'a InputName)> {
        let node = &self.nodes[id.0];

        node.outputs
            .iter()
            .filter_map(|(self_output, output_socket)| {
                let (to, to_input) = output_socket.link.as_ref()?;
                Some((*to, self_output, to_input))
            })
    }

    pub fn link_nodes(
        &mut self,
        from: NodeId,
        from_output: &str,
        to: NodeId,
        to_input: &str,
    ) -> Result<()> {
        let output_ty = self
            .nodes
            .get(from.0)
            .and_then(|n| {
                let s = n.outputs.get(from_output)?;
                Some(s.ty)
            })
            .ok_or_else(|| {
                anyhow!(
                    "Node `{}` does not have output `{}`",
                    from.0,
                    from_output,
                )
            })?;

        let input_ty = self
            .nodes
            .get(to.0)
            .and_then(|n| {
                let s = n.outputs.get(to_input)?;
                Some(s.ty)
            })
            .ok_or_else(|| {
                anyhow!("Node `{}` does not have input `{}`", to.0, to_input)
            })?;

        if output_ty != input_ty {
            anyhow::bail!(
                "Output socket {}.{} doesn't match type of input socket {}.{}",
                from.0,
                from_output,
                to.0,
                to_input
            );
        }

        {
            let socket =
                self.nodes[from.0].outputs.get_mut(from_output).unwrap();

            if socket.link.is_some() {
                anyhow::bail!(
                    "Output socket {}.{} already in use",
                    from.0,
                    from_output
                );
            }

            socket.link = Some((to, to_input.into()));
        }

        {
            let socket = self.nodes[to.0].outputs.get_mut(to_input).unwrap();

            if socket.link.is_some() {
                anyhow::bail!(
                    "Input socket {}.{} already in use",
                    to.0,
                    to_input
                );
            }

            socket.link = Some((from, from_output.into()));
        }

        Ok(())
    }

    fn prepare_node(&mut self, node: NodeId) -> Result<bool> {
        // let node = &mut self.nodes[node.0];
        todo!();
    }

    fn is_allocated(&self, res: ResourceId) -> bool {
        if let Some(res) = self.resources.get(res.0) {
            match res {
                Resource::Buffer { buffer, .. } => buffer.is_some(),
                Resource::Texture { texture, .. } => texture.is_some(),
            }
        } else {
            false
        }
    }

    fn allocate_resource(
        state: &super::State,
        res: &mut Resource,
    ) -> Result<()> {
        match res {
            Resource::Buffer {
                buffer,
                size,
                usage,
            } => {
                if buffer.is_some() {
                    // allocation already exists
                    return Ok(());
                }
                if size.is_none() {
                    anyhow::bail!("Can't allocate buffer without size")
                }

                let new_buffer =
                    state.device.create_buffer(&BufferDescriptor {
                        label: None,
                        size: size.unwrap() as u64,
                        usage: *usage,
                        mapped_at_creation: false,
                    });

                *buffer = Some(new_buffer);
            }
            Resource::Texture {
                texture,
                size,
                format,
                usage,
            } => {
                if texture.is_some() {
                    // allocation already exists
                    return Ok(());
                }
                if size.is_none() || format.is_none() {
                    anyhow::bail!(
                        "Can't allocate image without known size and format"
                    )
                }

                let [width, height] = size.unwrap();
                let format = format.unwrap();

                let new_texture = crate::texture::Texture::new(
                    &state.device,
                    &state.queue,
                    width as usize,
                    height as usize,
                    format,
                    *usage,
                    None,
                )?;

                *texture = Some(new_texture);
            }
        }

        Ok(())
    }

    fn prepare_buffer(
        &mut self,
        size: Option<usize>,
        usage: Option<BufferUsages>,
    ) -> ResourceId {
        let res = Resource::Buffer {
            buffer: None,
            size,
            usage: usage.unwrap_or(BufferUsages::empty()),
        };

        let id = ResourceId(self.resources.len());
        self.resources.push(res);
        id
    }

    fn prepare_image(
        &mut self,
        size: Option<[u32; 2]>,
        format: Option<TextureFormat>,
        usage: Option<TextureUsages>,
    ) -> ResourceId {
        let res = Resource::Texture {
            texture: None,
            size,
            format,
            usage: usage.unwrap_or(TextureUsages::empty()),
        };

        let id = ResourceId(self.resources.len());
        self.resources.push(res);
        id
    }
}



pub fn example_graph(
    state: &mut super::State,
    dims: [u32; 2],
) -> Result<Graph<()>> {
    let mut graph = Graph::default();

    let create_image = create_image_node(
        &mut graph,
        (),
        TextureFormat::Rgba8Unorm,
        TextureUsages::COPY_DST
            | TextureUsages::TEXTURE_BINDING
            | TextureUsages::STORAGE_BINDING,
        "window_dims",
    );

    let mut dims_map = rhai::Map::default();
    dims_map.insert("x".into(), rhai::Dynamic::from_int(dims[0] as i64));
    dims_map.insert("y".into(), rhai::Dynamic::from_int(dims[1] as i64));

    graph
        .graph_inputs
        .insert("window_dims".into(), dims_map.into());

    Ok(graph)
}

pub fn create_image_node<T>(
    graph: &mut Graph<T>,
    data: T,
    format: TextureFormat,
    usage: TextureUsages,
    dims_graph_input: &str,
) -> NodeId {
    use rhai::reify;

    let node_id = graph.add_node(data);

    let dims_graph_input = dims_graph_input.to_string();
    let output_source: OutputSource<T> = OutputSource::Allocate {
        allocate: Arc::new(move |graph, id| {
            let input = dims_graph_input.as_str();

            let dims = graph.graph_inputs.get(input).and_then(|v| {
                let map = reify!(v.clone() => Option<rhai::Map>)?;
                let x = map.get("x")?.clone();
                let y = map.get("y")?.clone();

                let x = reify!(x => Option<i64>)?;
                let y = reify!(y => Option<i64>)?;

                Some([x as u32, y as u32])
            });

            let dims = if let Some(dims) = dims {
                dims
            } else {
                anyhow::bail!(
                    "Error initializing image node:\
                key `{}` not found in graph inputs, or value\
                was not a map with integer `x` and `y` fields",
                    input
                );
            };

            let resource = Resource::Texture {
                texture: None,
                size: Some(dims),
                format: Some(format),
                usage,
            };

            Ok(resource)
        }),
    };

    let output_socket = OutputSocket {
        ty: DataType::Image,
        link: None,
        source: output_source,
        resource: None,
    };

    {
        let node = &mut graph.nodes[node_id.0];
        node.outputs.insert("output".into(), output_socket);
    }

    node_id
}


#[derive(Clone, PartialEq, Eq, Hash)]
pub enum NodeOutputDescriptor {
    Texture {
        desc: wgpu::TextureDescriptor<'static>,
        data: Option<Vec<u8>>,
    },
    Buffer {
        desc: wgpu::BufferDescriptor<'static>,
        data: Option<Vec<u8>>,
    },
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum NodeOutput {
    Texture { texture: TextureId },
    Buffer { buffer: BufferId },
}

pub struct NodeOutputDef {
    name: rhai::ImmutableString,
    allocate: Option<Arc<dyn Fn(rhai::Map) -> Result<NodeOutputDescriptor>>>,
}

pub type InputName = rhai::ImmutableString;
pub type OutputName = rhai::ImmutableString;
pub type CtxInputName = rhai::ImmutableString;

pub struct Node {
    id: NodeId,

    input_defs: Vec<(InputName, DataType)>,
    inputs: HashMap<InputName, NodeOutput>,

    output_defs: Vec<NodeOutputDef>,
    outputs: HashMap<OutputName, NodeOutput>,

    // encodes the edges
    input_socket_links: HashMap<InputName, (NodeId, OutputName)>,
    output_socket_links: HashMap<OutputName, (NodeId, InputName)>,

    // map from input names to sets of affected outputs,
    // outputs identified by name in the `outputs` map
    context_inputs: HashMap<CtxInputName, Vec<OutputName>>,
    // inputs: HashMap<rhai::ImmutableString,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct ComputeNodeDef {
    pipeline: usize, // index into graph context's compute_pipeline vec
    bind_group_layout: usize, // index in graph ctx `bind_group_layouts` vec
    push_constant_size: u32,
}

// TODO: only supports a single bind group for now
pub struct ComputeNode {
    node: Node,

    compute_def: ComputeNodeDef,

    bind_group: Option<usize>, // index in graph ctx `bind_group` vec

    // map from input sockets (and their contents) to bind group binding indices
    bind_group_map: Vec<(InputName, usize)>,
}

impl ComputeNode {
    fn execute(
        &self,
        ctx: &GraphContext,
        push_constants: &[u8],
        x_groups: u32,
        y_groups: u32,
        z_groups: u32,
        cmd: &mut CommandEncoder,
    ) -> Result<()> {
        let def = self.compute_def;

        if push_constants.len() != def.push_constant_size as usize {
            anyhow::bail!(
                "Compute push constant was {} bytes, but expected {} bytes",
                push_constants.len(),
                def.push_constant_size
            );
        }

        if let Some(group_ix) = self.bind_group {
            let pipeline = &ctx.compute_pipelines[def.pipeline];
            let bind_group = &ctx.bind_groups[group_ix];

            {
                let mut pass = cmd
                    .begin_compute_pass(&ComputePassDescriptor { label: None });

                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, bind_group, &[]);
                pass.set_push_constants(0, push_constants);
                pass.dispatch_workgroups(x_groups, y_groups, z_groups);
            }
        } else {
            anyhow::bail!("Can't execute compute node without bind group")
        }

        Ok(())
    }

    fn create_bind_group(&self, ctx: &GraphContext) -> Result<wgpu::BindGroup> {
        let def = &ctx.bind_group_layouts[self.compute_def.bind_group_layout];

        let mut entries = Vec::new();

        for (entry, (input, binding)) in
            std::iter::zip(&def.entries, &self.bind_group_map)
        {
            // get the input value from the node input
            // if it doesn't exist, return an error

            match entry.ty {
                // NB: in the future the fields may be useful
                BindingType::StorageTexture { .. } => {
                    let in_val = self
                        .node
                        .inputs
                        .get(input)
                        .and_then(|output| {
                            if let NodeOutput::Texture { texture } = output {
                                Some(texture)
                            } else {
                                None
                            }
                        })
                        .ok_or(anyhow::anyhow!(
                            "node input missing! {}",
                            input
                        ))?;

                    let texture = &ctx.textures[in_val.0];

                    let resource =
                        wgpu::BindingResource::TextureView(&texture.view);

                    let entry = wgpu::BindGroupEntry {
                        binding: *binding as u32,
                        resource,
                    };

                    entries.push(entry);
                }
                _ => anyhow::bail!("Only StorageTexture is supported"), /*
                                                                        BindingType::Buffer { ty, has_dynamic_offset, min_binding_size } => todo!(),
                                                                        BindingType::Sampler(_) => todo!(),
                                                                        BindingType::Texture { sample_type, view_dimension, multisampled } => todo!(),
                                                                        */
            }
        }

        let desc = wgpu::BindGroupDescriptor {
            label: None,
            layout: &def.layout,
            entries: entries.as_slice(),
        };

        /*
        let entries = vec![
            wgpu::BindGroupEntry
        ];
        */

        /*
         */

        todo!();
    }
}

impl Node {
    pub fn node_create_image(id: NodeId) -> Result<Self> {
        // let output_name = format!("create_image_output_{}", id.0);

        let alloc = move |ctx_input: rhai::Map| {
            let dims = ctx_input.get("dims").expect("context input missing");
            let dims = dims.clone_cast::<rhai::Map>();
            let width = dims.get("width").unwrap().as_int().unwrap();
            let height = dims.get("height").unwrap().as_int().unwrap();

            let size = wgpu::Extent3d {
                width: width as u32,
                height: height as u32,
                depth_or_array_layers: 1,
            };

            use wgpu::TextureUsages as Usages;

            let texture_desc = wgpu::TextureDescriptor {
                label: None, // Some(&name),
                size: size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: Usages::TEXTURE_BINDING | Usages::COPY_SRC,
            };

            let desc = NodeOutputDescriptor::Texture {
                desc: texture_desc,
                data: None,
            };

            Ok(desc)
        };

        let allocate = Arc::new(alloc)
            as Arc<dyn Fn(rhai::Map) -> Result<NodeOutputDescriptor>>;

        let output_def = NodeOutputDef {
            name: "output".into(),
            allocate: Some(allocate),
        };

        let mut result = Node {
            id,

            input_defs: Vec::new(),
            inputs: HashMap::default(),

            input_socket_links: HashMap::default(),
            output_socket_links: HashMap::default(),

            output_defs: vec![output_def],
            outputs: HashMap::default(),
            context_inputs: HashMap::default(),
        };

        result
            .context_inputs
            .insert("dims".into(), vec!["output".into()]);

        Ok(result)
    }
}

/*
Render and compute pipelines has bind group layouts (and push constant ranges)
associated with them, by ID, and each layout can track its "children",
also by ID


 */

// pub

pub struct BindGroupDef {
    layout: wgpu::BindGroupLayout,
    entries: Vec<wgpu::BindGroupLayoutEntry>,
}

pub struct GraphContext {
    resources: Vec<Resource>,

    buffers: Vec<wgpu::Buffer>,

    textures: Vec<crate::texture::Texture>,
    // textures: Vec<wgpu::Texture>,
    // texture_views: Vec<wgpu::TextureView>,
    render_pipelines: Vec<wgpu::RenderPipeline>,
    compute_pipelines: Vec<wgpu::ComputePipeline>,

    bind_group_layouts: Vec<BindGroupDef>,
    // bind_group_layouts: Vec<(wgpu::BindGroupLayout, wgpu::BindGroupLayoutDescriptor<'static>)>,
    bind_groups: Vec<wgpu::BindGroup>,

    compute_node_defs: Vec<ComputeNodeDef>,
}

impl std::ops::Index<ResourceId> for GraphContext {
    type Output = Resource;

    fn index(&self, index: ResourceId) -> &Self::Output {
        &self.resources[index.0]
    }
}

impl std::ops::IndexMut<ResourceId> for GraphContext {
    fn index_mut(&mut self, index: ResourceId) -> &mut Self::Output {
        &mut self.resources[index.0]
    }
}

impl std::default::Default for GraphContext {
    fn default() -> Self {
        Self {
            resources: Vec::new(),

            buffers: Vec::new(),
            textures: Vec::new(),
            // texture_views: Vec::new(),
            render_pipelines: Vec::new(),
            compute_pipelines: Vec::new(),

            bind_group_layouts: Vec::new(),
            bind_groups: Vec::new(),

            compute_node_defs: Vec::new(),
        }
    }
}

pub fn test_graph_context(state: &super::State) -> Result<GraphContext> {
    let mut ctx = GraphContext::default();

    let push_constant_size = 16 + 4 + 4;

    // load a compute shader and create a node definition for it
    let compute_node = ctx.create_compute_node_def(
        state,
        include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/shaders/shader.comp.spv"
        )),
        "main",
        &[BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::StorageTexture {
                access: StorageTextureAccess::WriteOnly,
                format: TextureFormat::Rgba8Unorm,
                view_dimension: TextureViewDimension::D2,
            },
            count: None,
        }],
        //
        PushConstantRange {
            stages: ShaderStages::COMPUTE,
            range: 0..push_constant_size,
        },
    )?;

    Ok(ctx)
}

impl GraphContext {
    pub fn dfs_postorder_input(&self, from: NodeId) -> Vec<NodeId> {
        let mut stack: VecDeque<NodeId> = VecDeque::new();

        let mut visited: FxHashSet<NodeId> = FxHashSet::default();

        while let Some(current) = stack.pop_back() {

            // let node = self.
        }

        todo!();
    }

    fn is_allocated(&self, res: ResourceId) -> bool {
        if let Some(res) = self.resources.get(res.0) {
            match res {
                Resource::Buffer { buffer, .. } => buffer.is_some(),
                Resource::Texture { texture, .. } => texture.is_some(),
            }
        } else {
            false
        }
    }

    fn allocate_resource(
        state: &super::State,
        res: &mut Resource,
    ) -> Result<()> {
        match res {
            Resource::Buffer {
                buffer,
                size,
                usage,
            } => {
                if buffer.is_some() {
                    // allocation already exists
                    return Ok(());
                }
                if size.is_none() {
                    anyhow::bail!("Can't allocate buffer without size")
                }

                let new_buffer =
                    state.device.create_buffer(&BufferDescriptor {
                        label: None,
                        size: size.unwrap() as u64,
                        usage: *usage,
                        mapped_at_creation: false,
                    });

                *buffer = Some(new_buffer);
            }
            Resource::Texture {
                texture,
                size,
                format,
                usage,
            } => {
                if texture.is_some() {
                    // allocation already exists
                    return Ok(());
                }
                if size.is_none() || format.is_none() {
                    anyhow::bail!(
                        "Can't allocate image without known size and format"
                    )
                }

                let [width, height] = size.unwrap();
                let format = format.unwrap();

                let new_texture = crate::texture::Texture::new(
                    &state.device,
                    &state.queue,
                    width as usize,
                    height as usize,
                    format,
                    *usage,
                    None,
                )?;

                *texture = Some(new_texture);
            }
        }

        Ok(())
    }

    fn prepare_buffer(
        &mut self,
        size: Option<usize>,
        usage: Option<BufferUsages>,
    ) -> ResourceId {
        let res = Resource::Buffer {
            buffer: None,
            size,
            usage: usage.unwrap_or(BufferUsages::empty()),
        };

        let id = ResourceId(self.resources.len());
        self.resources.push(res);
        id
    }

    fn prepare_image(
        &mut self,
        size: Option<[u32; 2]>,
        format: Option<TextureFormat>,
        usage: Option<TextureUsages>,
    ) -> ResourceId {
        let res = Resource::Texture {
            texture: None,
            size,
            format,
            usage: usage.unwrap_or(TextureUsages::empty()),
        };

        let id = ResourceId(self.resources.len());
        self.resources.push(res);
        id
    }

    /// Returns the index into `self.compute_node_defs` for the new def
    pub fn create_compute_node_def(
        &mut self,
        state: &super::State,
        shader_code: &[u8],
        entry_point: &str,
        bind_group_entries: &[BindGroupLayoutEntry],
        push_constant_range: PushConstantRange,
    ) -> Result<usize> {
        let bind_group_layout =
            state
                .device
                .create_bind_group_layout(&BindGroupLayoutDescriptor {
                    label: None,
                    entries: bind_group_entries,
                });

        let shader_desc = ShaderModuleDescriptor {
            label: None,
            source: wgpu::util::make_spirv(shader_code),
        };

        let shader_module = state.device.create_shader_module(shader_desc);

        let push_constant_size = push_constant_range.range.end;

        let pipeline_layout_desc = PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[push_constant_range],
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

        let bind_group_layout_ix = self.bind_group_layouts.len();
        let pipeline_ix = self.compute_pipelines.len();

        self.bind_group_layouts.push(BindGroupDef {
            layout: bind_group_layout,
            entries: bind_group_entries.to_vec(),
        });
        self.compute_pipelines.push(compute_pipeline);

        let compute_def_ix = self.compute_node_defs.len();
        self.compute_node_defs.push(ComputeNodeDef {
            pipeline: pipeline_ix,
            bind_group_layout: bind_group_layout_ix,
            push_constant_size,
        });

        Ok(compute_def_ix)
    }

    pub fn init(state: &super::State) -> Result<Self> {
        let mut result = Self::default();

        {
            let bg_layout_desc = wgpu::BindGroupLayoutDescriptor {
                label: Some("test bind group layout"),
                entries: &[BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        access: StorageTextureAccess::WriteOnly,
                        format: TextureFormat::Rgba8Unorm,
                        view_dimension: TextureViewDimension::D2,
                    },
                    count: None,
                }],
            };
            let bg_layout =
                state.device.create_bind_group_layout(&bg_layout_desc);

            let shader_desc = wgpu::include_spirv!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/shaders/",
                "shader.comp.spv"
            ));

            let shader_module = state.device.create_shader_module(shader_desc);

            let pipeline_layout_desc = PipelineLayoutDescriptor {
                label: Some("test pipeline layout"),
                bind_group_layouts: &[&bg_layout],
                push_constant_ranges: &[PushConstantRange {
                    stages: ShaderStages::COMPUTE,
                    range: 0..(4 * 4 + 4 + 4),
                }],
            };

            let pipeline_layout =
                state.device.create_pipeline_layout(&pipeline_layout_desc);

            let compute_desc = ComputePipelineDescriptor {
                label: Some("test compute pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader_module,
                entry_point: "main",
            };

            let compute_pipeline =
                state.device.create_compute_pipeline(&compute_desc);

            result.bind_group_layouts.push(BindGroupDef {
                layout: bg_layout,
                entries: bg_layout_desc.entries.to_vec(),
            });
            result.compute_pipelines.push(compute_pipeline);
        }

        Ok(result)
    }
}

/*


pub struct Node {
    // def: Arc<NodeDefinition>,
    id: NodeId,

    input_sockets: Vec<Option<(NodeId, SocketIx)>>,
    // output_sockets: Vec<Option<

    inputs: Vec<Option<(NodeId, SocketIx)>>,
    outputs: Vec<Option<(NodeId, SocketIx)>>,

}
*/

/*

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct TextureResourceSig {
    dimensions: [u32; 2],
    format: wgpu::TextureFormat,
    usage: wgpu::TextureUsages,
}

pub struct TextureResource {
    texture: wgpu::Texture,
    layout: wgpu::ImageDataLayout,
}

pub struct RenderGraph {
    images: FxHashMap<usize, ()>,
    buffers: FxHashMap<usize, ()>,

    nodes: FxHashMap<usize, ()>,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SocketDir {
    Input,
    Output,
}



#[derive(Clone)]
pub struct NodeDef {
    name: String,
    input_sockets: Vec<(String, DataType)>,
    output_sockets: Vec<(String, DataType)>,
}

impl NodeDef {
    pub fn new<'a, 'b>(
        name: &str,
        inputs: impl IntoIterator<Item = (&'a str, DataType)>,
        outputs: impl IntoIterator<Item = (&'b str, DataType)>,
    ) -> Self {
        let input_sockets = inputs
            .into_iter()
            .map(|(n, d)| (n.to_string(), d))
            .collect();

        let output_sockets = outputs
            .into_iter()
            .map(|(n, d)| (n.to_string(), d))
            .collect();

        Self {
            name: name.to_string(),
            input_sockets,
            output_sockets,
        }
    }
}
pub struct Node {
    def: Arc<NodeDef>,
    id: NodeId,

    inputs: Vec<Option<(NodeId, SocketIx)>>,
    outputs: Vec<Option<(NodeId, SocketIx)>>,
}

#[derive(Clone)]
pub struct NodeDefIO<I, O> {
    inputs: Vec<I>,
    outputs: Vec<O>,
}

#[derive(Clone)]
pub struct NodeInstanceIO<I, O> {
    inputs: Vec<Option<I>>,
    outputs: Vec<Option<O>>,
}

pub struct ImageResource {
    image: wgpu::Texture,
    view: wgpu::TextureView,
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
    usage: wgpu::TextureUsages,
    layout: wgpu::ImageDataLayout,
}


pub enum Resource {
    Free {
        name: String,
        data: rhai::Map,
        source: Option<NodeId>,
    },
    Resource {
        name: String,
        data: rhai::Dynamic,
    },
    /*
    Buffer {
        name: String,
        desc: wgpu::BufferDescriptor<'static>,
    },
    Image {
        name: String,
        desc: wgpu::TextureDescriptor<'static>,
    },
    */
}

pub struct Graph {
    defs: HashMap<String, Arc<NodeDef>>,
    nodes: Vec<Node>,

    outputs: FxHashSet<NodeId>,
}


#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct ResourceId(usize);

pub struct GraphContext {
    graph: Graph,

    /*
    first initialize all nodes by creating their corresponding
     */

    // resources: FxHashMap<NodeId, Resource>,
    // resources: FxHashMap<ResourceId, Resource>,
    // next_id: usize,
}

impl Graph {
    pub fn cache_outputs(&mut self) {
        self.outputs.clear();

        self.outputs
            .extend(self.nodes.iter().enumerate().filter_map(|(ix, node)| {
                if node.outputs.is_empty() {
                    let id = NodeId(ix);
                    Some(id)
                } else {
                    None
                }
            }));
    }

    /*
    pub fn build_schedule(&mut self) -> Result<Vec<NodeId>> {
        self.cache_outputs();

        let mut queue = VecDeque::new();

        let mut visited: FxHashSet<NodeId> = FxHashSet::default();

        let mut result = Vec::new();

        for &output in &self.outputs {
            let node = self.get_node(output)?;

            if !visited.contains(&output) {
                visited.insert(output);
                result.push(output);

                for (in_socket, &input) in node.inputs.iter().enumerate() {
                    if !visited.contains(&input) {
                        let in_node = self.get_node(input)?;

                    }

                }
            }
        }

        result.reverse();

        Ok(result)
    }
    */

    /*
    pub fn interpret<F, T>(&self, i_fn: F) -> Result<T>
    where F: FnMut(&Node, SocketIx, &Node, SocketIx) -> Result<T>,
    {
        todo!();
    }
    */

    pub fn init() -> Result<Self> {
        let mut defs = HashMap::new();

        let alloc_img_def = NodeDef::new("create_image", [], [("image", DataType::Image)]);

        defs.insert(alloc_img_def.name.clone(), Arc::new(alloc_img_def));

        let nodes = Vec::new();

        Ok(Self {
            defs,
            nodes,
            outputs: FxHashSet::default(),
        })
    }

    pub fn get_node_def(&self, def_name: &str) -> Result<&Arc<NodeDef>> {
        self.defs
            .get(def_name)
            .ok_or(anyhow!("Node definition `{}` not found", def_name))
    }

    pub fn node_degree(&self, node: NodeId) -> Result<(usize, usize)> {
        let node = self.get_node(node)?;
        let in_d = node.inputs.len();
        let out_d = node.outputs.len();
        Ok((in_d, out_d))
    }

    pub fn contains_node(&self, node: NodeId) -> bool {
        self.nodes.len() >= node.0
    }

    pub fn get_node(&self, node: NodeId) -> Result<&Node> {
        self.nodes
            .get(node.0)
            .ok_or_else(|| anyhow!("Node `{}` not found", node.0))
    }

    pub fn get_node_mut(&mut self, node: NodeId) -> Result<&mut Node> {
        self.nodes
            .get_mut(node.0)
            .ok_or_else(|| anyhow!("Node `{}` not found", node.0))
    }

    pub fn insert_node(&mut self, def_name: &str) -> Result<NodeId> {
        let def = self.get_node_def(def_name)?.clone();
        let id = NodeId(self.nodes.len());

        let inputs = vec![None; def.input_sockets.len()];
        let outputs = vec![None; def.output_sockets.len()];

        let node = Node {
            def,
            id,
            inputs,
            outputs,
        };

        self.nodes.push(node);

        Ok(id)
    }

    pub fn connect_nodes(
        &mut self,
        a: NodeId,
        a_socket: SocketIx,
        b: NodeId,
        b_socket: SocketIx,
    ) -> Result<()> {
        let (_a_in, a_out) = self.node_degree(a)?;
        let (b_in, _b_out) = self.node_degree(b)?;

        if a_socket > b_in || b_socket > a_out {
            bail!("Node socket mismatch");
        }

        {
            let a_n = &mut self.nodes[a.0];
            // let a_n = self.get_node_mut(a)?;
            a_n.outputs[a_socket] = Some((b, b_socket));
        }

        {
            let b_n = &mut self.nodes[b.0];
            // let b_n = self.get_node_mut(b)?;
            b_n.inputs[b_socket] = Some((a, a_socket));
        }

        Ok(())
    }
}



pub type ResourceAllocDef<T> = Arc<dyn Fn(rhai::Dynamic) -> Result<T>>;

pub type ImageAllocDef = ResourceAllocDef<wgpu::TextureDescriptor<'static>>;
pub type BufferAllocDef = ResourceAllocDef<wgpu::BufferDescriptor<'static>>;

pub enum ResourceAllocRule {
    Image(ImageAllocDef),
    Buffer(BufferAllocDef),
}


pub enum OutputDef {
    Passthrough { input_socket: SocketIx, data: DataType },
    Create { rule: ResourceAllocRule },
}

pub struct NodeDefinition {
    name: String,

    inputs: Vec<(String, DataType)>,
    output_defs: Vec<(String, OutputDef)>,
}
*/
