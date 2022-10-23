use std::{
    collections::{BTreeMap, BTreeSet, HashMap, HashSet},
    path::PathBuf,
};

use datafrog::{Iteration, Relation, RelationLeaper, Variable};
use rustc_hash::FxHashMap;
use wgpu::{SubmissionIndex, TextureUsages};

use std::sync::Arc;

use crate::{
    shader::{
        render::{
            FragmentShader, FragmentShaderInstance, GraphicsPipeline,
            VertexShader, VertexShaderInstance,
        },
        ComputeShader,
    },
    texture::Texture,
    DataType, State,
};

use anyhow::Result;

use super::NodeId;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct NodeSchemaId(pub usize);

#[derive(Clone)]
pub struct NodeSchema {
    node_type: NodeType,
    schema_id: NodeSchemaId,

    socket_names: Vec<rhai::ImmutableString>,

    source_sockets: Vec<(LocalSocketIx, DataType)>,
    source_rules_sockets: Vec<(LocalSocketIx, SocketMetadataSource)>,
    source_rules_scalars: Vec<(LocalSocketIx, rhai::ImmutableString)>,
    // source_rules_scalars: Vec<(LocalSocketIx, Vec<rhai::ImmutableString>)>,
    default_sources: FxHashMap<LocalSocketIx, ResourceMeta>,

    create_source_metadata: FxHashMap<
        LocalSocketIx,
        Arc<dyn Fn(&rhai::Map, Option<ResourceMeta>) -> Result<ResourceMeta>>,
    >,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ResourceMetadataEntry {
    BufferSize,
    BufferUsage,

    TextureSize,
    TextureFormat,
    TextureUsage,
}

impl ResourceMetadataEntry {
    fn is_buffer(&self) -> bool {
        match self {
            Self::BufferSize | Self::BufferUsage => true,
            _ => false,
        }
    }

    fn is_texture(&self) -> bool {
        match self {
            Self::TextureSize | Self::TextureFormat | Self::TextureUsage => {
                true
            }
            _ => false,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SocketMetadataSource {
    other_socket_ix: usize,
    entry: ResourceMetadataEntry,
}

impl SocketMetadataSource {
    pub fn buffer_size(other_socket_ix: LocalSocketIx) -> Self {
        SocketMetadataSource {
            other_socket_ix,
            entry: ResourceMetadataEntry::BufferSize,
        }
    }

    pub fn buffer_usage(other_socket_ix: LocalSocketIx) -> Self {
        SocketMetadataSource {
            other_socket_ix,
            entry: ResourceMetadataEntry::BufferUsage,
        }
    }

    pub fn texture_size(other_socket_ix: LocalSocketIx) -> Self {
        SocketMetadataSource {
            other_socket_ix,
            entry: ResourceMetadataEntry::TextureSize,
        }
    }

    pub fn texture_format(other_socket_ix: LocalSocketIx) -> Self {
        SocketMetadataSource {
            other_socket_ix,
            entry: ResourceMetadataEntry::TextureFormat,
        }
    }

    pub fn texture_usage(other_socket_ix: LocalSocketIx) -> Self {
        SocketMetadataSource {
            other_socket_ix,
            entry: ResourceMetadataEntry::TextureUsage,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum NodeType {
    Resource = 0,
    Compute = 1,
    Graphics = 2,
    // Commands = 4,
}
// pub struct NodeType(u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NodeSchemaSocketIx(usize);

type LocalSocketIx = usize;

type ResourceId = usize;
type TransientId = usize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum ResourceRef {
    Owned(ResourceId),
    Transient(TransientId),
}



pub enum SocketState {
    Uninitialized,
    Resolved,
    AwaitingTransient,
}

/*
pub enum NodeState {
    Uninitialized,
    Resolved,
    AwaitingTransient,
}
*/

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResourceMeta {
    Buffer {
        size: Option<usize>,
        usage: Option<wgpu::BufferUsages>,
        stride: Option<usize>,
    },
    Texture {
        size: Option<[u32; 2]>,
        format: Option<wgpu::TextureFormat>,
        usage: Option<wgpu::TextureUsages>,
    },
}

impl ResourceMeta {
    pub fn is_buffer(&self) -> bool {
        matches!(self, Self::Buffer { .. })
    }

    pub fn is_texture(&self) -> bool {
        matches!(self, Self::Texture { .. })
    }

    fn apply_socket_rule(
        &mut self,
        other_res: &ResourceMeta,
        rule: ResourceMetadataEntry,
    ) -> Result<()> {
        if (self.is_buffer() && rule.is_texture())
            || (self.is_texture() && rule.is_buffer())
        {
            anyhow::bail!("Rule did not match resource type");
        }

        if let ResourceMeta::Buffer { size, usage, .. } = self {
            if other_res.is_texture() {
                anyhow::bail!("Source resource did not match self");
            }

            if let ResourceMeta::Buffer {
                size: other_size,
                usage: other_usage,
                ..
            } = other_res
            {
                match rule {
                    ResourceMetadataEntry::BufferSize => {
                        *size = *other_size;
                    }
                    ResourceMetadataEntry::BufferUsage => {
                        *usage = *other_usage;
                    }

                    _ => (),
                }
            }
        } else if let ResourceMeta::Texture {
            size,
            format,
            usage,
        } = self
        {
            if other_res.is_buffer() {
                anyhow::bail!("Source resource did not match self");
            }
            if let ResourceMeta::Texture {
                size: o_size,
                format: o_format,
                usage: o_usage,
            } = other_res
            {
                match rule {
                    ResourceMetadataEntry::TextureSize => {
                        *size = *o_size;
                    }
                    ResourceMetadataEntry::TextureFormat => {
                        *format = *o_format;
                    }
                    ResourceMetadataEntry::TextureUsage => {
                        *usage = *o_usage;
                    }
                    _ => (),
                }
            }
        }

        Ok(())
    }

    pub fn allocate(&self, state: &State) -> Result<Resource> {
        match self {
            ResourceMeta::Buffer {
                size,
                usage,
                stride,
            } => {
                if size.is_none() || usage.is_none() {
                    anyhow::bail!(
                        "Can't allocate buffer without known size and usage"
                    );
                }

                let buffer =
                    state.device.create_buffer(&wgpu::BufferDescriptor {
                        label: None,
                        size: size.unwrap() as u64,
                        usage: usage.unwrap(),
                        mapped_at_creation: false,
                    });

                Ok(Resource::Buffer(buffer))
            }
            ResourceMeta::Texture {
                size,
                format,
                usage,
            } => {
                if size.is_none() || format.is_none() || usage.is_none() {
                    dbg!(size);
                    dbg!(format);
                    dbg!(usage);
                    anyhow::bail!("Can't allocate image without known size, format, and usage");
                }

                let [width, height] = size.unwrap();
                let format = format.unwrap();
                let usage = usage.unwrap();

                let texture = crate::texture::Texture::new(
                    &state.device,
                    &state.queue,
                    width as usize,
                    height as usize,
                    format,
                    usage,
                    None,
                )?;

                Ok(Resource::Texture(texture))
            }
        }
    }

    pub fn buffer_default() -> Self {
        Self::Buffer {
            size: None,
            usage: None,
            stride: None,
        }
    }

    pub fn texture_default() -> Self {
        Self::Texture {
            size: None,
            format: None,
            usage: None,
        }
    }
}

pub enum Resource {
    Buffer(wgpu::Buffer),
    Texture(Texture),
}

pub struct Node {
    schema: NodeSchemaId,

    links: Vec<(LocalSocketIx, (NodeId, LocalSocketIx))>,
}

pub struct Graph {
    schemas: Vec<NodeSchema>,

    nodes: Vec<Node>,
    // nodes: Vec<NodeId>,
    node_names: FxHashMap<NodeId, rhai::ImmutableString>,

    socket_resources: BTreeMap<(NodeId, LocalSocketIx), ResourceId>,
    socket_transients: BTreeMap<(NodeId, LocalSocketIx), TransientId>,

    // node_names: Vec<rhai::ImmutableString>,
    resource_meta: Vec<ResourceMeta>,
    resources: Vec<Option<Resource>>,

    transient_cache_id: HashMap<rhai::ImmutableString, TransientId>,
    transient_cache: BTreeMap<TransientId, ResourceMeta>,

    // transient_links: BTreeMap<TransientId, Vec<(NodeId, LocalSocketIx)>>,
    transient_links: BTreeMap<(NodeId, LocalSocketIx), TransientId>,
    // transient_links: Vec<(TransientId, (NodeId, LocalSocketIx))>,
    // transient_resource_links: HashMap<String, Vec<(NodeId, LocalSocketIx)>>,
    // transients_meta_cache: HashMap<String, ResourceMeta>,
}

pub enum InputResource<'a> {
    Texture {
        size: [u32; 2],
        format: wgpu::TextureFormat,

        texture: Option<&'a wgpu::Texture>,
        view: Option<&'a wgpu::TextureView>,
    },
    Buffer {
        size: usize,
        stride: Option<usize>,

        buffer: Option<&'a wgpu::Buffer>,
    },
}

impl<'a> InputResource<'a> {
    pub fn metadata(&self) -> ResourceMeta {
        match self {
            InputResource::Texture {
                size,
                format,
                texture,
                view,
            } => ResourceMeta::Texture {
                size: Some(*size),
                format: Some(*format),
                usage: None,
            },
            InputResource::Buffer {
                size,
                stride,
                buffer,
            } => ResourceMeta::Buffer {
                size: Some(*size),
                usage: None,
                stride: *stride,
            },
        }
    }

    pub fn is_buffer(&self) -> bool {
        matches!(self, Self::Buffer { .. })
    }

    pub fn is_texture(&self) -> bool {
        matches!(self, Self::Texture { .. })
    }
}

impl Graph {
    pub fn new() -> Self {
        Self {
            schemas: Vec::new(),
            nodes: Vec::new(),
            node_names: FxHashMap::default(),

            resource_meta: Vec::new(),
            resources: Vec::new(),

            socket_resources: BTreeMap::default(),
            socket_transients: BTreeMap::default(),

            transient_cache_id: HashMap::default(),
            transient_cache: BTreeMap::default(),
            transient_links: BTreeMap::new(),
            // transients_meta_cache: HashMap::default(),
        }
    }

    pub fn update_transient_cache<'a>(
        &mut self,
        transient_res: &'a HashMap<String, InputResource<'a>>,
    ) {
        for (key, in_res) in transient_res.iter() {
            dbg!(key);
            if let Some(id) = self.transient_cache_id.get(key.as_str()).copied()
            {
                let meta = in_res.metadata();

                let cache = self.transient_cache.get(&id).copied();

                if cache != Some(meta) {
                    self.transient_cache.insert(id, meta);
                }
            } else {
                let id = self.transient_cache.len();
                self.transient_cache_id.insert(key.into(), id);
                self.transient_cache.insert(id, in_res.metadata());
            }
        }
    }

    pub fn validate<'a>(
        &mut self,
        transient_res: &'a HashMap<String, InputResource<'a>>,
        graph_scalar_in: &'a rhai::Map,
    ) -> Result<bool> {
        /*
            validate graph:
             - create resource IDs for each resource origin/source in the graph
             - make sure sockets are only used once (each for I/O)
             - make sure all sockets end up with a valid resource ID (not unique)

            also make sure the `resource_meta` vector is up to date,
            i.e. each element should be fully filled in and be ready to be used
            to allocate a new resource

            likewise transients_meta_cache should be populated

        */

        self.update_transient_cache(transient_res);

        let mut socket_origins = Vec::new();

        for (ix, node) in self.nodes.iter().enumerate() {
            let node_id = NodeId::from(ix);
            let schema = &self.schemas[node.schema.0];

            for &(socket_ix, ty) in schema.source_sockets.iter() {
                let socket_key = (node_id, socket_ix);
                if let Some(res_id) = self.socket_resources.get(&socket_key) {
                    socket_origins.push(((node_id, socket_ix), *res_id));
                    continue;
                }

                let res_id = self.resource_meta.len();

                let meta = match ty {
                    DataType::Buffer => ResourceMeta::buffer_default(),
                    DataType::Texture => ResourceMeta::texture_default(),
                };
                self.resource_meta.push(meta);
                self.resources.push(None);

                self.socket_resources.insert(socket_key, res_id);
                socket_origins.push(((node_id, socket_ix), res_id));
            }
        }

        println!("socket_origins: {:?}", socket_origins);

        // now, build the relations for the datafrog programs

        // node(node_id, schema_id).
        let nodes = Relation::from_iter(
            self.nodes
                .iter()
                .enumerate()
                .map(|(ix, node)| (NodeId::from(ix), node.schema)),
        );

        // resource ID for each socket; empty if uninitialized
        // socket_resource(socket_ix, res_id)
        let existing_socket_resources = Relation::from_iter(
            self.socket_resources.iter().map(|(&k, &v)| (k, v)),
        );

        let links = Relation::from_iter(
            self.nodes.iter().enumerate().flat_map(|(ix, node)| {
                let node_id = NodeId::from(ix);
                node.links.iter().map(move |(out_ix, (to_node, in_ix))| {
                    ((node_id, *out_ix), (*to_node, *in_ix))
                })
            }),
        );

        let inv_links = Relation::from_map(&links, |&(from, to)| (to, from));

        // socket_origins((node_id, socket_ix), res_id).
        let socket_origins = Relation::from_iter(socket_origins);

        let mut iteration = datafrog::Iteration::new();

        /*
         goal: match each socket up with a resource ID from an origin

         socket_resources((node, socket), res_id)
           : socket_origins((node, socket), res_id).

         socket_resources((to_id, in_ix), res_id)
           :- socket_resources((from_id, out_ix), res_id),
              links((from_id, out_ix), (to_id, in_ix)).

         socket_resources((from_id, out_ix), res_id)
           :- socket_resources((to_id, in_ix), res_id),
              inv_links((to_id, in_ix), (from_id, out_ix)).

        */
        let socket_resources = iteration
            .variable::<((NodeId, LocalSocketIx), ResourceId)>(
                "socket_resources",
            );

        let socket_transients = iteration
            .variable::<((NodeId, LocalSocketIx), TransientId)>(
                "socket_transients",
            );

        socket_resources.extend(socket_origins.iter().copied());
        socket_transients
            .extend(self.transient_links.iter().map(|(&k, &v)| (k, v)));

        println!("transient links: {:?}", self.transient_links);

        while iteration.changed() {
            socket_resources.from_join(
                &socket_resources,
                &links,
                |_from, &res_id, &to| (to, res_id),
            );

            socket_transients.from_join(
                &socket_transients,
                &links,
                |_from, &res_id, &to| (to, res_id),
            );

            /*
            socket_resources.from_join(
                &socket_resources,
                &inv_links,
                |&(to, in_ix), &res_id, &(from, out_ix)| {
                    ((from, out_ix), res_id)
                },
            );
            */
        }

        let socket_resources = socket_resources.complete();
        let socket_transients = socket_transients.complete();

        // update stored socket resources
        self.socket_resources = socket_resources.elements.into_iter().collect();
        self.socket_transients =
            socket_transients.elements.into_iter().collect();

        println!("socket_resources: {:?}", self.socket_resources);
        println!("socket_transients: {:?}", self.socket_transients);

        let topo_order = self.build_topological_order()?;

        for node in topo_order {
            log::warn!("preparing node {:?}", node);
            self.prepare_node_meta(graph_scalar_in, node)?;
        }

        // return true if all sockets point to owned resources
        // or cached transient resources

        // for meta in self.resource_meta {
        // match meta {
        //
        // }
        // }

        Ok(true)
    }

    pub fn execute<'a>(
        &mut self,
        state: &State,
        transient_res: &'a HashMap<String, InputResource<'a>>,
        graph_scalar_in: &'a rhai::Map,
        // node_scalar_in: FxHashMap<NodeId, rhai::Map>,
    ) -> Result<SubmissionIndex> {
        // iterate through all resources, allocating all `None` resources
        // that have filled out `resource_meta`s -- afterward, all
        // owned resources should be ready for use

        println!("{:#?}", self.socket_resources);

        for (res_id, res_slot) in self.resources.iter_mut().enumerate() {
            if res_slot.is_some() {
                // TODO reallocate if necessary
                continue;
            }

            // todo error handling here
            let meta = self.resource_meta[res_id];

            match meta.allocate(state) {
                Ok(res) => *res_slot = Some(res),
                Err(e) => {
                    panic!("error allocating image for resource {res_id}");
                }
            }
            *res_slot = Some(meta.allocate(state)?);
        }

        // preprocess all nodes, handling bind groups, push constants,
        // workgroup counts, etc.

        // walk the graph in execution order, recording the node commands
        // in order to a command encoder

        // submit the commands to the GPU queue and return the submission index

        todo!();
    }

    pub fn add_node(&mut self, schema: NodeSchemaId) -> NodeId {
        let id = NodeId::from(self.nodes.len());

        let node = Node {
            schema,
            links: Vec::new(),
        };

        self.nodes.push(node);

        id
    }

    pub fn add_link(
        &mut self,
        src: NodeId,
        src_socket: LocalSocketIx,
        dst: NodeId,
        dst_socket: LocalSocketIx,
    ) {
        let node = &mut self.nodes[src.0];
        node.links.push((src_socket, (dst, dst_socket)));
    }

    pub fn add_link_from_transient(
        &mut self,
        transient_key: &str,
        dst: NodeId,
        dst_socket: LocalSocketIx,
    ) -> Option<()> {
        dbg!(&self.transient_cache_id);
        let id = *self.transient_cache_id.get(transient_key)?;
        dbg!();

        self.transient_links.insert((dst, dst_socket), id);

        println!("transient links: {:?}", self.transient_links);

        Some(())
    }

    pub fn build_topological_order(&self) -> Result<Vec<NodeId>> {
        let mut order = Vec::new();

        // find initial nodes, those without any incoming links

        let mut incoming_links: FxHashMap<NodeId, BTreeSet<NodeId>> =
            FxHashMap::default();

        for (ix, node) in self.nodes.iter().enumerate() {
            let from_id = NodeId::from(ix);

            for &(out_ix, (to_id, in_ix)) in node.links.iter() {
                incoming_links.entry(to_id).or_default().insert(from_id);
            }
        }

        let mut start_nodes = self
            .nodes
            .iter()
            .enumerate()
            .filter_map(|(ix, node)| {
                let id = NodeId::from(ix);
                (!incoming_links.contains_key(&id)).then_some(id)
            })
            .collect::<Vec<_>>();

        while let Some(n) = start_nodes.pop() {
            order.push(n);

            let node = &self.nodes[n.0];

            /*
            for each node m with an edge e from n to m do
              remove edge e from the graph
              if m has no other incoming edges then
                insert m into S
            */

            for (out_ix, (to_id, in_ix)) in node.links.iter() {
                let mut remove_set = false;
                if let Some(incoming) = incoming_links.get_mut(&to_id) {
                    incoming.remove(&n);
                    if incoming.is_empty() {
                        remove_set = true;
                        start_nodes.push(*to_id);
                    }
                }

                if remove_set {
                    incoming_links.remove(&to_id);
                }
            }
        }

        if !incoming_links.is_empty() {
            anyhow::bail!("Cycle detected");
        }

        Ok(order)
    }

    fn prepare_node_meta(
        &mut self,
        graph_scalar_in: &rhai::Map,
        id: NodeId,
    ) -> Result<()> {
        // iterate through socket_resources that match this node ID
        let node = &self.nodes[id.0];
        let schema = &self.schemas[node.schema.0];

        for &(socket_ix, ty) in schema.source_sockets.iter() {
            let key = (id, socket_ix);
            let res_id = self.socket_resources[&key];

            let mut meta = match ty {
                DataType::Buffer => ResourceMeta::buffer_default(),
                DataType::Texture => ResourceMeta::texture_default(),
            };

            if let Some(def) = schema.default_sources.get(&socket_ix) {
                // TODO ensure that the types are correct i guess
                meta = *def;
            }

            {
                let source_rules = schema
                    .source_rules_sockets
                    .iter()
                    .filter(|(s, _)| *s == socket_ix);

                for (_dst_socket, source) in source_rules {
                    log::warn!("applying rule");
                    let src_socket = source.other_socket_ix;
                    log::warn!(
                        "getting socket for node {}, socket {}",
                        id.0,
                        src_socket
                    );

                    let src_meta = {
                        if let Some(t_id) =
                            self.socket_transients.get(&(id, src_socket))
                        {
                            println!(
                                "getting transient resource at socket {:?}.{}",
                                id, src_socket
                            );
                            &self.transient_cache[t_id]
                        } else {
                            println!(
                                "getting resource at socket {:?}.{}",
                                id, src_socket
                            );
                            let res_id = self
                                .socket_resources
                                .get(&(id, src_socket))
                                .unwrap();
                            &self.resource_meta[*res_id]
                        }
                    };
                    meta.apply_socket_rule(src_meta, source.entry)?;
                }
            }

            // rules from transients
            {
                // let rules = schema.
            }

            // if there's an applicable constructor, give it the preliminary
            // ResourceMeta
            if let Some(create) = schema.create_source_metadata.get(&socket_ix)
            {
                meta = create(graph_scalar_in, Some(meta))?;
            }

            self.resource_meta[res_id] = meta;
        }

        Ok(())
    }

    pub fn add_schemas(&mut self) -> Result<()> {
        {
            let id = NodeSchemaId(self.schemas.len());

            let mut default_sources = FxHashMap::default();

            // TODO these are temporary
            let size = [800, 600];
            let usage = TextureUsages::all();

            default_sources.insert(
                0,
                ResourceMeta::Texture {
                    size: Some(size),
                    format: Some(wgpu::TextureFormat::Rgba8Unorm),
                    usage: Some(usage),
                },
            );

            let create_image_meta = move |scalars: &rhai::Map,
                                          meta: Option<ResourceMeta>|
                  -> Result<ResourceMeta> {
                let dims = scalars.get("dimensions").unwrap();
                let size = dims.clone_cast::<[u32; 2]>();

                match meta {
                    Some(ResourceMeta::Texture { format, usage, .. }) => {
                        //
                        Ok(ResourceMeta::Texture {
                            size: Some(size),
                            format: Some(wgpu::TextureFormat::Rgba8Unorm),
                            usage: Some(wgpu::TextureUsages::all()),
                        })
                    }
                    _ => panic!("todo handle errors"),
                }
            };

            let mut schema = NodeSchema {
                node_type: NodeType::Resource,
                schema_id: id,

                socket_names: vec!["image".into()],

                source_sockets: vec![(0, DataType::Texture)],
                source_rules_sockets: vec![],
                source_rules_scalars: vec![(0, "dimensions".into())],

                default_sources,

                create_source_metadata: FxHashMap::default(),
            };

            schema
                .create_source_metadata
                .insert(0, Arc::new(create_image_meta));

            self.schemas.push(schema);
            id
        };

        {
            let id = NodeSchemaId(self.schemas.len());

            let schema = NodeSchema {
                node_type: NodeType::Graphics,
                schema_id: id,

                socket_names: vec!["image_in".into(), "image_out".into()],

                source_sockets: vec![],
                source_rules_sockets: vec![],
                source_rules_scalars: vec![],

                default_sources: FxHashMap::default(),

                create_source_metadata: FxHashMap::default(),
            };

            self.schemas.push(schema);
        };

        {
            let id = NodeSchemaId(self.schemas.len());

            let default_source = ResourceMeta::Texture {
                size: None,
                format: None,
                usage: Some(
                    wgpu::TextureUsages::TEXTURE_BINDING
                        | wgpu::TextureUsages::RENDER_ATTACHMENT,
                ),
            };
            let mut default_sources = FxHashMap::default();

            default_sources.insert(1, default_source);

            let schema = NodeSchema {
                node_type: NodeType::Compute,
                schema_id: id,

                socket_names: vec!["image_in".into(), "image_out".into()],

                source_sockets: vec![(1, DataType::Texture)],
                source_rules_sockets: vec![
                    (1, SocketMetadataSource::texture_size(0)),
                    (1, SocketMetadataSource::texture_format(0)),
                ],
                source_rules_scalars: vec![],

                default_sources,

                create_source_metadata: FxHashMap::default(),
            };

            self.schemas.push(schema);
        }

        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum NodeOpId {
    NoOp,
    Graphics(usize),
    Compute(usize),
}

pub struct GraphOps {
    // vertex_shaders: HashMap<PathBuf, VertexShader>,
    // fragment_shaders: HashMap<PathBuf, FragmentShader>,
    // compute_shaders: HashMap<
    graphics: Vec<GraphicsPipeline>,
    compute: Vec<ComputeShader>,
}

impl GraphOps {
    pub fn create_graphics_schema(
        &mut self,
        state: &State,
        vert_src: &[u8],
        frag_src: &[u8],
        frag_out_formats: &[wgpu::TextureFormat],
        schema_id: NodeSchemaId,
    ) -> Result<(NodeSchema, NodeOpId)> {
        let vert = VertexShader::from_spirv(&state, vert_src, "main")?;
        let frag = FragmentShader::from_spirv(&state, frag_src, "main")?;

        let vert = Arc::new(vert);
        let frag = Arc::new(frag);

        let vert_inst = VertexShaderInstance::from_shader_single_buffer(
            &vert,
            wgpu::VertexStepMode::Vertex,
        );

        let frag_inst =
            FragmentShaderInstance::from_shader(&frag, frag_out_formats)?;

        let graphics = GraphicsPipeline::new(&state, vert_inst, frag_inst)?;

        let op_id = NodeOpId::Graphics(self.graphics.len());

        // build sockets
        // there's one socket per vertex buffer,
        // one per fragment attachment,
        // and one per binding resource... maybe?

        // there should be *some* additional logic, like you shouldn't have
        // to manually provide samplers in sockets, but i'm not sure how to
        // add that right now -- it requires making some additional assumptions
        // about the shader code

        // for now i think i'll just add some relatively simple way to
        // "redirect sockets" as well

        // all it'd have to do is -- and this is something that happens when
        // creating bind groups -- reuse the same resource for multiple
        // bindings
        // this can be accomplished by unifying the relevant variable names
        // in a given shader

        // but, honestly, it wouldn't be that big a deal to just add all
        // sockets for now, and add a handful more links to handle samplers
        //   -- that's the way to go

        let mut socket_names = Vec::new();

        // add vertex shader sockets
        //   vertex buffers
        // hardcoded to a single vertex buffer
        socket_names.push("vertex_in".into());

        //   bind groups
        for group in vert.group_bindings.iter() {
            for binding in group.bindings.iter() {
                socket_names.push(binding.global_var_name.clone());
            }
        }

        // add fragment shader sockets
        //   render attachments
        for output in frag.fragment_outputs.iter() {
            socket_names.push(output.name.as_str().into());
        }
        
        //   bind groups
        for group in frag.group_bindings.iter() {
            for binding in group.bindings.iter() {
                socket_names.push(binding.global_var_name.clone());
            }
        }

        let schema = NodeSchema {
            node_type: NodeType::Graphics,
            schema_id,

            socket_names,
            source_sockets: vec![],

            source_rules_sockets: vec![],
            source_rules_scalars: vec![],

            default_sources: HashMap::default(),
            create_source_metadata: HashMap::default(),
        };

        self.graphics.push(graphics);

        Ok((schema, op_id))
    }
}


struct SocketAnnotations {
    name_id_cache: HashMap<String, usize>,
    annotations: BTreeMap<(NodeId, LocalSocketIx), (usize, rhai::Map)>,
}

pub struct NodeOpState {
    bind_groups: Vec<wgpu::BindGroup>,

    node_parameters: rhai::Map,

}