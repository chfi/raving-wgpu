use rustc_hash::{FxHashMap, FxHashSet};
use wgpu::*;

use anyhow::{anyhow, bail, Result};
use std::{
    collections::{HashMap, VecDeque},
    sync::Arc,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct NodeId(usize);

impl From<usize> for NodeId {
    #[inline]
    fn from(u: usize) -> Self {
        Self(u)
    }
}

impl From<NodeId> for usize {
    #[inline]
    fn from(u: NodeId) -> usize {
        u.0
    }
}

pub type SocketIx = usize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DataType {
    Buffer,
    Texture,
    // Scalar,
}

#[derive(Debug)]
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct ResourceId(pub usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ResourceHandle {
    id: ResourceId,
    time: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct TextureId(usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
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

impl<T> std::fmt::Debug for OutputSource<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InputPassthrough { input } => f
                .debug_struct("InputPassthrough")
                .field("input", input)
                .finish(),
            Self::Allocate { allocate } => f.debug_struct("Allocate").finish(),
            Self::Ref { resource } => {
                f.debug_struct("Ref").field("resource", resource).finish()
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LocalSocketRef {
    Input { socket_name: InputName },
    Output { socket_name: OutputName },
}

#[derive(Debug, Clone)]
pub struct OutputSocket<T> {
    // name: OutputName,
    ty: DataType,
    link: Option<(NodeId, InputName)>,
    source: OutputSource<T>,
    resource: Option<ResourceHandle>,
}

impl<T> OutputSocket<T> {
    pub fn buffer(source: OutputSource<T>) -> Self {
        OutputSocket {
            ty: DataType::Buffer,
            link: None,
            source,
            resource: None,
        }
    }

    pub fn texture(source: OutputSource<T>) -> Self {
        OutputSocket {
            ty: DataType::Texture,
            link: None,
            source,
            resource: None,
        }
    }

    pub fn passthrough(ty: DataType, socket: &str) -> Self {
        OutputSocket {
            ty,
            link: None,
            source: OutputSource::InputPassthrough {
                input: socket.into(),
            },
            resource: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct InputSocket {
    // name: InputName,
    ty: DataType,
    link: Option<(NodeId, OutputName)>,
    resource: Option<ResourceHandle>,
}

impl InputSocket {
    pub fn buffer() -> Self {
        InputSocket {
            ty: DataType::Buffer,
            link: None,
            resource: None,
        }
    }

    pub fn texture() -> Self {
        InputSocket {
            ty: DataType::Texture,
            link: None,
            resource: None,
        }
    }
}

impl LocalSocketRef {
    pub fn input(socket_name: &str) -> Self {
        Self::Input {
            socket_name: socket_name.into(),
        }
    }

    pub fn output(socket_name: &str) -> Self {
        Self::Output {
            socket_name: socket_name.into(),
        }
    }
}

pub struct Node_<T> {
    id: NodeId,
    inputs: HashMap<InputName, InputSocket>,
    outputs: HashMap<OutputName, OutputSocket<T>>,

    is_prepared: bool,
    is_ready: bool,
    data: T,

    // pub bind: Option<Box<dyn BindableNode<T>>>,
    pub execute: Option<Box<dyn ExecuteNode<T>>>,
}

impl<T: std::fmt::Debug> std::fmt::Debug for Node_<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Node_")
            .field("id", &self.id)
            .field("inputs", &self.inputs)
            .field("outputs", &self.outputs)
            .field("is_prepared", &self.is_prepared)
            .field("is_ready", &self.is_ready)
            .field("data", &self.data)
            .finish()
    }
}

pub struct ComputeShaderOp {
    shader: Arc<crate::shader::ComputeShader>,
    // mapping from compute shader global var. name to local socket
    resource_map: HashMap<String, LocalSocketRef>,

    pub bind_groups: Vec<BindGroup>,
}

pub trait ExecuteNode<T> {
    fn execute(&self, graph: &Graph<T>, cmd: &mut CommandEncoder)
        -> Result<()>;

    fn set_bind_groups(&mut self, bind_groups: Vec<BindGroup>);

    fn create_bind_groups(
        &self,
        node: &Node_<T>,
        state: &super::State,
        resources: &[Resource],
    ) -> Result<Vec<BindGroup>>;
}

impl<T> ExecuteNode<T> for ComputeShaderOp {
    fn set_bind_groups(&mut self, bind_groups: Vec<BindGroup>) {
        self.bind_groups = bind_groups;
    }

    fn create_bind_groups(
        &self,
        node: &Node_<T>,
        state: &crate::State,
        resources: &[Resource],
    ) -> Result<Vec<BindGroup>> {
        let mut binding_map = HashMap::default();

        for (g_var, socket) in &self.resource_map {
            let res_id = match socket {
                LocalSocketRef::Input { socket_name } => node
                    .inputs
                    .get(socket_name)
                    .and_then(|socket| socket.resource),
                LocalSocketRef::Output { socket_name } => node
                    .outputs
                    .get(socket_name)
                    .and_then(|socket| socket.resource),
            }
            .ok_or_else(|| anyhow!("Socket `{:?}` not found", socket))?;

            binding_map.insert(g_var.into(), res_id.id);
        }

        self.shader
            .create_bind_groups_impl(state, resources, &binding_map)
    }

    fn execute(
        &self,
        graph: &Graph<T>,
        cmd: &mut CommandEncoder,
    ) -> Result<()> {
        let mut pass =
            cmd.begin_compute_pass(&ComputePassDescriptor { label: None });

        pass.set_pipeline(&self.shader.pipeline);

        for (ix, group) in self.bind_groups.iter().enumerate() {
            pass.set_bind_group(ix as u32, group, &[]);
        }

        // TODO figure out dispatch group counts
        let x_groups = 800 / 16;
        let y_groups = 600 / 16;
        pass.dispatch_workgroups(x_groups, y_groups, 1);

        Ok(())
    }
}

#[derive(Default)]
pub struct Graph<T> {
    pub nodes: Vec<Node_<T>>,

    pub resources: Vec<Resource>,

    // compute_pipelines: Vec<wgpu::ComputePipeline>,
    // bind_group_layouts: Vec<BindGroupDef>,
    // bind_groups: Vec<wgpu::BindGroup>,
    pub graph_inputs: rhai::Map,
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

            // bind: None,
            execute: None,
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
                let s = n.inputs.get(to_input)?;
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
            let socket = self.nodes[to.0].inputs.get_mut(to_input).unwrap();

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

    pub fn prepare_node(&mut self, id: NodeId) -> Result<bool> {
        {
            // loop through all inputs and update the local `resource` fields
            let node = &self.nodes[id.0];
            let mut input_resources = Vec::new();

            for (input_name, input) in node.inputs.iter() {
                let (input_id, in_output_name) =
                    input.link.as_ref().ok_or_else(|| {
                        anyhow!("Input socket {}.{} not set", id.0, input_name,)
                    })?;

                let other = &self.nodes[input_id.0];
                let output =
                    other.outputs.get(in_output_name).ok_or_else(|| {
                        anyhow!(
                            "Output socket {}.{} not set",
                            input_id.0,
                            in_output_name,
                        )
                    })?;

                if let Some(handle) = output.resource {
                    input_resources.push((input_name.clone(), handle));
                }
            }

            let _ = node;

            for (name, handle) in input_resources {
                let input = self.nodes[id.0].inputs.get_mut(&name).unwrap();
                input.resource = Some(handle);
            }
        }

        let mut output_passthroughs: Vec<_> = Vec::new();
        let mut output_resources: Vec<_> = Vec::new();

        {
            // scope since we need a mutable reference to `node` later
            let node = &self.nodes[id.0];

            // iterate through all node outputs, preparing or linking
            // the appropriate resource descriptors
            for (output_name, output) in node.outputs.iter() {
                match &output.source {
                    OutputSource::InputPassthrough { input } => {
                        let input_socket = node.inputs.get(input).unwrap();

                        output_passthroughs.push((
                            output_name.clone(),
                            input_socket.resource.unwrap().clone(),
                        ));
                    }
                    OutputSource::Allocate { allocate } => {
                        // allocate the resource and store it for later
                        let resource = allocate(&self, id)?;
                        output_resources.push((output_name.clone(), resource));
                    }
                    OutputSource::Ref { resource } => {
                        //
                        // todo!();
                        log::error!(
                            "ref output sources unimplemented, ignored"
                        );
                    }
                }
            }
        }

        for (output_name, handle) in output_passthroughs {
            let socket =
                self.nodes[id.0].outputs.get_mut(&output_name).unwrap();
            socket.resource = Some(handle);
        }

        for (output_name, res) in output_resources {
            let res_id = ResourceId(self.resources.len());

            let handle = ResourceHandle {
                id: res_id,
                time: 0,
            };

            self.resources.push(res);

            let socket =
                self.nodes[id.0].outputs.get_mut(&output_name).unwrap();
            socket.resource = Some(handle);
        }

        self.nodes[id.0].is_prepared = true;

        Ok(true)
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

    pub fn allocate_node_resources(
        &mut self,
        state: &super::State,
    ) -> Result<()> {
        let mut resources = FxHashSet::default();
        for node in self.nodes.iter() {
            for socket in node.inputs.values() {
                if let Some(h) = socket.resource {
                    resources.insert(h.id);
                }
            }
            for socket in node.outputs.values() {
                if let Some(h) = socket.resource {
                    resources.insert(h.id);
                }
            }
        }

        for res_id in resources {
            let res = &mut self.resources[res_id.0];
            Self::allocate_resource(state, res)?;
        }

        Ok(())
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

    // let format = TextureFormat::Bgra8Unorm

    let create_image = create_image_node(
        &mut graph,
        (),
        // TextureFormat::Bgra8Unorm,
        TextureFormat::Rgba8Unorm,
        TextureUsages::COPY_DST
            | TextureUsages::COPY_SRC
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

    let compute = example_compute_node(state, &mut graph, ())?;

    let buffer_size = 512;

    let buffer = create_buffer_node(
        &mut graph,
        (),
        BufferUsages::COPY_SRC | BufferUsages::STORAGE,
        buffer_size,
    );

    dbg!();
    graph.link_nodes(create_image, "output", compute, "input")?;
    graph.link_nodes(buffer, "output", compute, "buffer_in")?;
    dbg!();

    Ok(graph)
}

pub fn example_compute_node<T>(
    state: &super::State,
    graph: &mut Graph<T>,
    data: T,
) -> Result<NodeId> {
    let node_id = graph.add_node(data);

    let output_socket = OutputSocket::passthrough(DataType::Texture, "input");
    // let output_source: OutputSource<T> = OutputSource::InputPassthrough {
    //     input: "input".into(),
    // };

    // let output_socket = OutputSocket {
    //     ty: DataType::Texture,
    //     link: None,
    //     source: output_source,
    //     resource: None,
    // };

    let input_socket = InputSocket {
        ty: DataType::Texture,
        link: None,
        resource: None,
    };

    let shader_op = {
        let shader_src = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/shaders/shader.comp.spv"
        ));

        let shader = crate::shader::ComputeShader::from_spirv(
            state, shader_src, "main",
        )?;

        let shader = Arc::new(shader);

        let mut resource_map = HashMap::default();

        resource_map.insert("image".into(), LocalSocketRef::input("input"));

        ComputeShaderOp {
            shader,
            resource_map,
            bind_groups: Vec::new(),
        }
    };

    {
        let node = &mut graph.nodes[node_id.0];
        node.inputs.insert("input".into(), input_socket);
        node.inputs.insert(
            "buffer_in".into(),
            InputSocket {
                ty: DataType::Buffer,
                link: None,
                resource: None,
            },
        );
        node.outputs.insert("output".into(), output_socket);

        node.execute = Some(Box::new(shader_op));
    }

    Ok(node_id)
}

pub fn create_buffer_node<T>(
    graph: &mut Graph<T>,
    data: T,
    usage: BufferUsages,
    size: usize,
) -> NodeId {
    let node_id = graph.add_node(data);

    let output_source: OutputSource<T> = OutputSource::Allocate {
        allocate: Arc::new(move |graph, id| {
            Ok(Resource::Buffer {
                buffer: None,
                size: Some(size),
                usage,
            })
        }),
    };

    let output_socket = OutputSocket::buffer(output_source);

    {
        let node = &mut graph.nodes[node_id.0];
        node.outputs.insert("output".into(), output_socket);
    }

    node_id
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
            log::error!("dims_graph_input: {}", dims_graph_input);

            let dims = graph.graph_inputs.get(input).and_then(|v| {
                dbg!();
                dbg!(v.type_name());
                let map = v.clone_cast::<rhai::Map>();
                let x = map.get("x")?.clone();
                let y = map.get("y")?.clone();

                let x = x.as_int().ok()?;
                let y = y.as_int().ok()?;

                Some([x as u32, y as u32])
            });

            log::error!("dims is: {:?}", dims);
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
        ty: DataType::Texture,
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

pub type InputName = rhai::ImmutableString;
pub type OutputName = rhai::ImmutableString;
pub type CtxInputName = rhai::ImmutableString;
