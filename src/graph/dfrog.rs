use std::collections::{BTreeMap, BTreeSet, HashMap};

use datafrog::Relation;
use rustc_hash::{FxHashMap, FxHashSet};
use wgpu::{
    BindingResource, BindingType, CommandEncoder, ComputePass, RenderPass,
    SubmissionIndex, TextureUsages,
};

use std::sync::Arc;

use crate::{
    shader::{
        interface::{GroupBindings, PushConstants},
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SocketBinding {
    VertexBuffer {
        slot: u32,
    },
    FragmentAttachment {
        location: u32,
    },
    BindGroup {
        stage: naga::ShaderStage,
        group: u32,
        binding: u32,
    },
}

impl SocketBinding {
    fn compute_bind_group(group: u32, binding: u32) -> Self {
        Self::BindGroup {
            stage: naga::ShaderStage::Compute,
            group,
            binding,
        }
    }

    fn vertex_bind_group(group: u32, binding: u32) -> Self {
        Self::BindGroup {
            stage: naga::ShaderStage::Vertex,
            group,
            binding,
        }
    }

    fn fragment_bind_group(group: u32, binding: u32) -> Self {
        Self::BindGroup {
            stage: naga::ShaderStage::Fragment,
            group,
            binding,
        }
    }
}

#[derive(Clone)]
pub struct NodeSchema {
    pub node_type: NodeType,
    schema_id: NodeSchemaId,

    pub socket_names: Vec<rhai::ImmutableString>,

    pub source_sockets: Vec<(LocalSocketIx, DataType)>,
    pub source_rules_sockets: Vec<(LocalSocketIx, SocketMetadataSource)>,
    pub source_rules_scalars: Vec<(LocalSocketIx, rhai::ImmutableString)>,
    // source_rules_scalars: Vec<(LocalSocketIx, Vec<rhai::ImmutableString>)>,
    pub default_sources: FxHashMap<LocalSocketIx, ResourceMeta>,

    pub socket_bindings: Vec<(LocalSocketIx, SocketBinding)>,
    // binding_socket: Vec<((wgpu::ShaderStages, (u32, u32)), LocalSocketIx)>,
    pub create_source_metadata: FxHashMap<
        LocalSocketIx,
        Arc<dyn Fn(&rhai::Map, Option<ResourceMeta>) -> Result<ResourceMeta>>,
    >,
}

impl NodeSchema {
    pub fn new_with<'a, F>(
        node_type: NodeType,
        schema_id: NodeSchemaId,
        socket_names: impl IntoIterator<Item = &'a str>,
        f: F,
    ) -> Self
    where
        F: FnOnce(&mut NodeSchema),
    {
        let mut schema = NodeSchema {
            node_type,
            schema_id,
            socket_names: socket_names.into_iter().map(|s| s.into()).collect(),

            source_sockets: vec![],
            source_rules_sockets: vec![],
            source_rules_scalars: vec![],
            socket_bindings: vec![],

            default_sources: FxHashMap::default(),
            create_source_metadata: FxHashMap::default(),
        };

        f(&mut schema);

        schema
    }
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

impl ResourceRef {
    fn get_as_input<'a>(
        &self,
        owned_resources: &'a [Option<Resource>],
        resource_metadata: &[ResourceMeta],
        transient_resources: &'a FxHashMap<TransientId, &'a InputResource<'a>>,
    ) -> Option<InputResource<'a>> {
        match self {
            ResourceRef::Owned(r) => {
                let res = owned_resources.get(*r)?.as_ref()?;

                let meta = resource_metadata.get(*r)?;

                match (res, *meta) {
                    (
                        Resource::Buffer(buffer),
                        ResourceMeta::Buffer { size, stride, .. },
                    ) => Some(InputResource::Buffer {
                        size: size?,
                        stride,
                        buffer,
                    }),
                    (
                        Resource::Texture(texture),
                        ResourceMeta::Texture { size, format, .. },
                    ) => Some(InputResource::Texture {
                        size: size?,
                        format: format?,

                        texture: Some(&texture.texture),
                        view: Some(&texture.view),
                        sampler: Some(&texture.sampler),
                    }),
                    _ => None,
                }
            }
            ResourceRef::Transient(r) => {
                transient_resources.get(r).copied().copied()
            }
        }
    }
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
    ) -> Result<bool> {
        if (self.is_buffer() && rule.is_texture())
            || (self.is_texture() && rule.is_buffer())
        {
            anyhow::bail!("Rule did not match resource type");
        }

        let mut changed = false;

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
                        changed |= size != other_size;
                        *size = *other_size;
                    }
                    ResourceMetadataEntry::BufferUsage => {
                        changed |= usage != other_usage;
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
                size: other_size,
                format: other_format,
                usage: other_usage,
            } = other_res
            {
                match rule {
                    ResourceMetadataEntry::TextureSize => {
                        changed |= size != other_size;
                        *size = *other_size;
                    }
                    ResourceMetadataEntry::TextureFormat => {
                        changed |= format != other_format;
                        *format = *other_format;
                    }
                    ResourceMetadataEntry::TextureUsage => {
                        changed |= usage != other_usage;
                        *usage = *other_usage;
                    }
                    _ => (),
                }
            }
        }

        Ok(changed)
    }

    pub fn allocate(&self, state: &State) -> Result<Resource> {
        match self {
            ResourceMeta::Buffer { size, usage, .. } => {
                if size.is_none() || usage.is_none() {
                    anyhow::bail!(
                        "Can't allocate buffer without known size \
                         (was {size:?}) and usage (was {usage:?})"
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
                    anyhow::bail!(
                        "Can't allocate image without \
                    known size (was {size:?}), \
                    format (was {format:?}), \
                    and usage (was {usage:?})"
                    );
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
    pub node_names: FxHashMap<NodeId, rhai::ImmutableString>,

    // these are populated by the links, resource origins, and transients
    socket_resources: BTreeMap<(NodeId, LocalSocketIx), ResourceId>,
    socket_transients: BTreeMap<(NodeId, LocalSocketIx), TransientId>,

    // node_names: Vec<rhai::ImmutableString>,
    resource_meta: Vec<ResourceMeta>,
    resources: Vec<Option<Resource>>,

    // updated_transients: HashSet<rhai::ImmutableString>,
    updated_transients: FxHashSet<TransientId>,
    transient_cache_id: HashMap<rhai::ImmutableString, TransientId>,
    transient_cache: BTreeMap<TransientId, ResourceMeta>,

    transient_links: BTreeMap<(NodeId, LocalSocketIx), rhai::ImmutableString>,
    // transient_links: BTreeMap<(NodeId, LocalSocketIx), TransientId>,
    // transient_links: BTreeMap<TransientId, Vec<(NodeId, LocalSocketIx)>>,
    // transient_links: Vec<(TransientId, (NodeId, LocalSocketIx))>,
    // transient_resource_links: HashMap<String, Vec<(NodeId, LocalSocketIx)>>,
    // transients_meta_cache: HashMap<String, ResourceMeta>,
    pub ops: GraphOps,
}

#[derive(Clone, Copy)]
pub enum InputResource<'a> {
    Texture {
        size: [u32; 2],
        format: wgpu::TextureFormat,

        sampler: Option<&'a wgpu::Sampler>,
        texture: Option<&'a wgpu::Texture>,
        view: Option<&'a wgpu::TextureView>,
    },
    Buffer {
        size: usize,
        stride: Option<usize>,

        buffer: &'a wgpu::Buffer,
    },
}

impl<'a> InputResource<'a> {
    fn create_bind_group_entry(
        &self,
        binding: u32,
        entry: &wgpu::BindGroupLayoutEntry,
    ) -> Result<wgpu::BindGroupEntry<'a>> {
        let binding_resource = match self {
            InputResource::Texture { sampler, view, .. } => match entry.ty {
                BindingType::Sampler(_) => {
                    BindingResource::Sampler(&sampler.as_ref().unwrap())
                }
                BindingType::Texture { .. }
                | BindingType::StorageTexture { .. } => {
                    BindingResource::TextureView(&view.as_ref().unwrap())
                }
                BindingType::Buffer { .. } => {
                    panic!("TODO: Binding type mismatch!");
                }
            },
            InputResource::Buffer { buffer, .. } => match entry.ty {
                BindingType::Buffer { .. } => {
                    BindingResource::Buffer(buffer.as_entire_buffer_binding())
                }
                _ => {
                    panic!("TODO: Binding type mismatch!");
                }
            },
        };

        Ok(wgpu::BindGroupEntry {
            binding,
            resource: binding_resource,
        })
    }

    pub fn metadata(&self) -> ResourceMeta {
        match self {
            InputResource::Texture { size, format, .. } => {
                ResourceMeta::Texture {
                    size: Some(*size),
                    format: Some(*format),
                    usage: None,
                }
            }
            InputResource::Buffer { size, stride, .. } => {
                ResourceMeta::Buffer {
                    size: Some(*size),
                    usage: None,
                    stride: *stride,
                }
            }
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

            updated_transients: FxHashSet::default(),
            transient_cache_id: HashMap::default(),
            transient_cache: BTreeMap::default(),
            transient_links: BTreeMap::new(),
            // transients_meta_cache: HashMap::default(),
            ops: GraphOps::default(),
        }
    }

    pub fn add_graphics_schema(
        &mut self,
        state: &State,
        vert_src: &[u8],
        frag_src: &[u8],
        frag_out_formats: &[wgpu::TextureFormat],
    ) -> Result<NodeSchemaId> {
        let schema_id = NodeSchemaId(self.schemas.len());
        let schema = self.ops.create_graphics_schema(
            state,
            vert_src,
            frag_src,
            frag_out_formats,
            schema_id,
        )?;
        self.schemas.push(schema);

        Ok(schema_id)
    }

    pub fn add_compute_schema(
        &mut self,
        state: &State,
        comp_src: &[u8],
    ) -> Result<NodeSchemaId> {
        let schema_id = NodeSchemaId(self.schemas.len());
        let schema =
            self.ops.create_compute_schema(state, comp_src, schema_id)?;
        self.schemas.push(schema);

        Ok(schema_id)
    }

    pub fn add_custom_compute_schema<'a, F>(
        &mut self,
        state: &State,
        comp_src: &[u8],
        f: F,
    ) -> Result<NodeSchemaId>
    where
        F: FnOnce(&mut NodeSchema),
    {
        let schema_id = NodeSchemaId(self.schemas.len());
        let mut schema =
            self.ops.create_compute_schema(state, comp_src, schema_id)?;

        f(&mut schema);
        self.schemas.push(schema);

        Ok(schema_id)
    }

    pub fn add_custom_schema<'a, F>(
        &mut self,
        socket_names: impl IntoIterator<Item = &'a str>,
        // state: &State,
        f: F,
    ) -> NodeSchemaId
    where
        F: FnOnce(&mut NodeSchema),
    {
        let schema_id = NodeSchemaId(self.schemas.len());
        let schema = NodeSchema::new_with(
            NodeType::Resource,
            schema_id,
            socket_names,
            f,
        );
        self.schemas.push(schema);
        schema_id
    }

    pub fn update_transient_cache<'a>(
        &mut self,
        transient_res: &'a HashMap<String, InputResource<'a>>,
    ) {
        for (key, in_res) in transient_res.iter() {
            if let Some(id) = self.transient_cache_id.get(key.as_str()).copied()
            {
                let meta = in_res.metadata();
                let cache = self.transient_cache.get(&id).copied();

                if cache != Some(meta) {
                    self.updated_transients.insert(id);
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

        // now, build the relations for the datafrog programs
        let links = Relation::from_iter(
            self.nodes.iter().enumerate().flat_map(|(ix, node)| {
                let node_id = NodeId::from(ix);
                node.links.iter().map(move |(out_ix, (to_node, in_ix))| {
                    ((node_id, *out_ix), (*to_node, *in_ix))
                })
            }),
        );

        // let inv_links = Relation::from_map(&links, |&(from, to)| (to, from));

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

        socket_transients.extend(
            self.transient_links
                .iter()
                .map(|(&k, v)| (k, *self.transient_cache_id.get(v).unwrap())),
        );

        // println!("transient links: {:?}", self.transient_links);

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

        // println!("socket_resources: {:?}", self.socket_resources);
        // println!("socket_transients: {:?}", self.socket_transients);

        let topo_order = self.build_topological_order()?;

        /*
        find all resources that should change/be reallocated
        */

        for node in topo_order {
            // log::warn!("preparing node {:?}", node);
            self.prepare_node_meta(graph_scalar_in, node)?;
        }

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
        for (res_id, res_slot) in self.resources.iter_mut().enumerate() {
            if res_slot.is_some() {
                continue;
            }

            // todo error handling here
            let meta = self.resource_meta[res_id];

            match meta.allocate(state) {
                Ok(res) => {
                    *res_slot = Some(res);
                }
                Err(e) => {
                    panic!(
                        "error allocating image for resource {res_id}: {:?}",
                        e
                    );
                }
            }
            *res_slot = Some(meta.allocate(state)?);
        }

        // preprocess all nodes, handling bind groups, push constants,
        // workgroup counts, etc.

        // this maps each socket, ordered by node, to a pointer to
        // a binding that can be used for resources etc.
        let socket_res_refs = {
            let mut refs = BTreeMap::default();
            for (key, res) in self.socket_resources.iter() {
                refs.insert(*key, ResourceRef::Owned(*res));
            }

            for (key, res) in self.socket_transients.iter() {
                refs.insert(*key, ResourceRef::Transient(*res));
            }
            refs
        };

        let transient_res_ids = self.map_transient_resource_ids(transient_res);

        let resource_ctx = ResourceCtx {
            socket_resource_refs: &socket_res_refs,
            owned_resources: &self.resources,
            resource_metadata: &self.resource_meta,
            transient_resource: &transient_res_ids,
            graph_scalars: graph_scalar_in,
        };

        // walk the graph in execution order, recording the node commands
        // in order to a command encoder
        let topo_order = self.build_topological_order()?;

        for node_id in topo_order.iter() {
            // println!("preprocessing node {}", node_id.0);
            let node = &self.nodes[node_id.0];
            let schema = &self.schemas[node.schema.0];

            self.ops
                .preprocess_node(state, &resource_ctx, schema, *node_id)?;
        }

        // submit the commands to the GPU queue and return the submission index

        let mut encoder = state.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("Render graph encoder"),
            },
        );

        for node_id in topo_order.iter() {
            //
            // println!("executing node {}", node_id.0);
            let node = &self.nodes[node_id.0];
            let schema = &self.schemas[node.schema.0];

            self.ops.execute_node(
                &resource_ctx,
                schema,
                *node_id,
                &mut encoder,
            )?;
        }

        let sub_ix = state.queue.submit(Some(encoder.finish()));

        Ok(sub_ix)
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
        self.transient_links
            .insert((dst, dst_socket), transient_key.into());

        Some(())
    }

    pub fn build_topological_order(&self) -> Result<Vec<NodeId>> {
        let mut order = Vec::new();

        // find initial nodes, those without any incoming links

        let mut incoming_links: FxHashMap<NodeId, BTreeSet<NodeId>> =
            FxHashMap::default();

        for (ix, node) in self.nodes.iter().enumerate() {
            let from_id = NodeId::from(ix);

            for &(_out_ix, (to_id, _in_ix)) in node.links.iter() {
                incoming_links.entry(to_id).or_default().insert(from_id);
            }
        }

        let mut start_nodes = self
            .nodes
            .iter()
            .enumerate()
            .filter_map(|(ix, _node)| {
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

            for (_out_ix, (to_id, _in_ix)) in node.links.iter() {
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

    // assumes that the transient cache has already been updated
    // for the provided map
    fn map_transient_resource_ids<'a>(
        &self,
        transient_res: &'a HashMap<String, InputResource<'a>>,
    ) -> FxHashMap<TransientId, &'a InputResource<'a>> {
        let mut output = HashMap::default();

        for (name, res) in transient_res.iter() {
            let id = self.transient_cache_id.get(name.as_str()).unwrap();
            output.insert(*id, res);
        }

        output
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
                    // log::warn!("applying rule");
                    let src_socket = source.other_socket_ix;
                    // log::warn!(
                    //     "getting socket for node {}, socket {}",
                    //     id.0,
                    //     src_socket
                    // );

                    let src_meta = {
                        if let Some(t_id) =
                            self.socket_transients.get(&(id, src_socket))
                        {
                            // println!(
                            //     "getting transient resource at socket {:?}.{}",
                            //     id, src_socket
                            // );
                            &self.transient_cache[t_id]
                        } else {
                            // println!(
                            //     "getting resource at socket {:?}.{}",
                            //     id, src_socket
                            // );
                            let res_id = self
                                .socket_resources
                                .get(&(id, src_socket))
                                .unwrap();
                            &self.resource_meta[*res_id]
                        }
                    };
                    let changed =
                        meta.apply_socket_rule(src_meta, source.entry)?;

                    if changed {
                        self.resources[res_id] = None;
                    }
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

    //////////

    //////////

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
                    Some(ResourceMeta::Texture { .. }) => {
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

                socket_bindings: vec![],

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

                socket_bindings: vec![],

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

            let socket_bindings = vec![
                (0, SocketBinding::compute_bind_group(0, 0)),
                (1, SocketBinding::compute_bind_group(0, 1)),
            ];

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

                socket_bindings,

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

#[derive(Default)]
pub struct GraphOps {
    // vertex_shaders: HashMap<PathBuf, VertexShader>,
    // fragment_shaders: HashMap<PathBuf, FragmentShader>,
    // compute_shaders: HashMap<
    graphics: Vec<GraphicsPipeline>,
    compute: Vec<ComputeShader>,

    pub node_op_state: FxHashMap<NodeId, NodeOpState>,
    schema_ops: FxHashMap<NodeSchemaId, NodeOpId>,
}

pub struct ResourceCtx<'a> {
    socket_resource_refs: &'a BTreeMap<(NodeId, LocalSocketIx), ResourceRef>,
    owned_resources: &'a [Option<Resource>],
    resource_metadata: &'a [ResourceMeta],
    transient_resource: &'a FxHashMap<TransientId, &'a InputResource<'a>>,
    graph_scalars: &'a rhai::Map,
}

pub struct NodeCtx<'a> {
    resource: &'a ResourceCtx<'a>,
    node_id: NodeId,
}

impl<'a> NodeCtx<'a> {
    pub fn get_resource_at_socket(
        &self,
        socket: LocalSocketIx,
    ) -> Result<InputResource<'a>> {
        self.resource.get_resource_at_socket(self.node_id, socket)
    }
}

impl<'a> ResourceCtx<'a> {
    // InputResources includes metadata
    pub fn get_resource_at_socket(
        &self,
        node_id: NodeId,
        socket: LocalSocketIx,
    ) -> Result<InputResource<'a>> {
        let res_ref = &self.socket_resource_refs[&(node_id, socket)];

        let in_res = res_ref
            .get_as_input(
                &self.owned_resources,
                &self.resource_metadata,
                &self.transient_resource,
            )
            .unwrap();

        Ok(in_res)
    }
}

impl GraphOps {
    fn initialize_node_op_state(
        &mut self,
        schema: &NodeSchema,
        node_id: NodeId,
    ) -> Result<()> {
        let schema_id = schema.schema_id;

        let op_id = self.schema_ops.get(&schema_id);
        if op_id.is_none() || matches!(op_id, Some(NodeOpId::NoOp)) {
            // don't need to do anything
            return Ok(());
        }

        let op_id = op_id.unwrap();

        let op_state = self.node_op_state.entry(node_id).or_default();

        let mut add_push_const =
            |stage: naga::ShaderStage, push_const: &Option<PushConstants>| {
                if let Some(pc) = push_const.as_ref() {
                    op_state.push_constants.insert(stage, pc.clone());
                }
            };

        match op_id {
            NodeOpId::NoOp => (),
            NodeOpId::Graphics(i) => {
                let graphics = &self.graphics[*i];

                add_push_const(
                    naga::ShaderStage::Vertex,
                    &graphics.vertex.shader.push_constants,
                );
                add_push_const(
                    naga::ShaderStage::Fragment,
                    &graphics.fragment.shader.push_constants,
                );
            }
            NodeOpId::Compute(i) => {
                let compute = &self.compute[*i];
                add_push_const(
                    naga::ShaderStage::Compute,
                    &compute.clone_push_constants(),
                );
            }
        }

        Ok(())
    }

    fn preprocess_node<'a>(
        &mut self,
        state: &State,
        resource_ctx: &'a ResourceCtx<'a>,
        schema: &NodeSchema,
        node_id: NodeId,
    ) -> Result<bool> {
        if !self.node_op_state.contains_key(&node_id) {
            self.initialize_node_op_state(schema, node_id)?;
        }

        let schema_id = schema.schema_id;

        let op_id = self.schema_ops.get(&schema_id);
        if op_id.is_none() || matches!(op_id, Some(NodeOpId::NoOp)) {
            // don't need to do anything
            return Ok(false);
        }

        let op_id = op_id.unwrap();

        let mut vert_binds = BTreeMap::default();
        let mut frag_binds = BTreeMap::default();
        let mut comp_binds = BTreeMap::default();

        for (socket, binding) in schema.socket_bindings.iter() {
            match binding {
                SocketBinding::BindGroup {
                    stage,
                    group,
                    binding,
                } => {
                    let key = (*group, *binding);
                    match stage {
                        naga::ShaderStage::Vertex => {
                            vert_binds.insert(key, *socket);
                        }
                        naga::ShaderStage::Fragment => {
                            frag_binds.insert(key, *socket);
                        }
                        naga::ShaderStage::Compute => {
                            comp_binds.insert(key, *socket);
                        }
                    }
                }
                SocketBinding::VertexBuffer { slot } => {
                    let op_state =
                        self.node_op_state.entry(node_id).or_default();
                    op_state.vertex_buffers.insert(*slot, *socket);
                }
                SocketBinding::FragmentAttachment { location } => {
                    let op_state =
                        self.node_op_state.entry(node_id).or_default();
                    op_state.attachments.insert(*location, *socket);
                }
            }
        }

        let create_bind_group = |layouts: &[wgpu::BindGroupLayout],
                                 group_bindings: &[GroupBindings],
                                 bind_sockets: &BTreeMap<
            (u32, u32),
            LocalSocketIx,
        >|
         -> Result<Vec<wgpu::BindGroup>> {
            let group_entries = create_bind_group_entries(
                resource_ctx,
                bind_sockets,
                group_bindings,
                node_id,
            )?;

            let groups: Vec<_> = group_entries
                .into_iter()
                .zip(layouts)
                .map(|(entries, layout)| {
                    state.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: None,
                        layout,
                        entries: entries.as_slice(),
                    })
                })
                .collect();

            Ok(groups)
        };

        match op_id {
            NodeOpId::Graphics(gfx_id) => {
                let pipeline = &self.graphics[*gfx_id];

                let vert_groups = create_bind_group(
                    &pipeline.vertex.shader.bind_group_layouts,
                    &pipeline.vertex.shader.group_bindings,
                    &vert_binds,
                )?;

                let frag_groups = create_bind_group(
                    &pipeline.fragment.shader.bind_group_layouts,
                    &pipeline.fragment.shader.group_bindings,
                    &frag_binds,
                )?;

                let op_state = self.node_op_state.entry(node_id).or_default();
                op_state
                    .bind_groups
                    .insert(naga::ShaderStage::Vertex, vert_groups);
                op_state
                    .bind_groups
                    .insert(naga::ShaderStage::Fragment, frag_groups);
            }
            NodeOpId::Compute(comp_id) => {
                let pipeline = &self.compute[*comp_id];

                let comp_groups = create_bind_group(
                    &pipeline.bind_group_layouts,
                    &pipeline.group_bindings,
                    &comp_binds,
                )?;

                let op_state = self.node_op_state.entry(node_id).or_default();
                op_state
                    .bind_groups
                    .insert(naga::ShaderStage::Compute, comp_groups);

                // TODO how to compute workgroup count
                op_state.workgroup_count = Some([1, 1, 1]);
            }
            NodeOpId::NoOp => unreachable!(),
        }

        Ok(true)
    }

    fn execute_node<'a>(
        &self,
        resource_ctx: &'a ResourceCtx<'a>,
        schema: &NodeSchema,
        node_id: NodeId,
        encoder: &mut CommandEncoder,
    ) -> Result<bool> {
        let schema_id = schema.schema_id;

        let op_id = self.schema_ops.get(&schema_id);
        if op_id.is_none() || matches!(op_id, Some(NodeOpId::NoOp)) {
            // don't need to do anything
            return Ok(false);
        }

        let op_id = op_id.unwrap();

        let op_state = if let Some(s) = self.node_op_state.get(&node_id) {
            s
        } else {
            return Ok(false);
        };

        // println!("op_state: {:#?}", op_state);

        match op_id {
            NodeOpId::Graphics(i) => {
                // log::warn!("executing graphics node");
                let graphics = &self.graphics[*i];

                let mut vertex_buffers = op_state
                    .vertex_buffers
                    .iter()
                    .map(|(&slot, &socket)| {
                        // log::warn!(
                        //     "vertex buffer slot {slot}, socket {socket}"
                        // );
                        let res = resource_ctx
                            .get_resource_at_socket(node_id, socket)
                            .unwrap();

                        if let InputResource::Buffer {
                            size,
                            stride,
                            buffer,
                        } = res
                        {
                            (slot, (buffer, size, stride))
                        } else {
                            panic!("texture was used as vertex buffer!");
                        }
                    })
                    .collect::<Vec<_>>();

                vertex_buffers.sort_by_key(|(a, _)| *a);
                // println!("vertex_buffers: {:#?}", vertex_buffers);

                let mut attchs = op_state
                    .attachments
                    .iter()
                    .map(|(&loc, &socket)| {
                        let res = resource_ctx
                            .get_resource_at_socket(node_id, socket)
                            .unwrap();

                        let view =
                            if let InputResource::Texture { view, .. } = res {
                                view.unwrap()
                            } else {
                                panic!("buffer was used as render attachment!");
                            };

                        (
                            loc,
                            Some(wgpu::RenderPassColorAttachment {
                                view: &view,
                                resolve_target: None,
                                ops: wgpu::Operations {
                                    load: wgpu::LoadOp::Clear(wgpu::Color {
                                        r: 0.0,
                                        g: 0.0,
                                        b: 0.0,
                                        a: 1.0,
                                    }),
                                    store: true,
                                },
                            }),
                        )
                    })
                    .collect::<Vec<_>>();

                attchs.sort_by_key(|(a, _)| *a);
                let attchs =
                    attchs.into_iter().map(|(_, b)| b).collect::<Vec<_>>();

                {
                    let label = format!("Node {} - render pass", node_id.0);
                    let desc = wgpu::RenderPassDescriptor {
                        label: Some(label.as_str()),
                        color_attachments: attchs.as_slice(),
                        depth_stencil_attachment: None,
                    };

                    let mut pass = encoder.begin_render_pass(&desc);

                    pass.set_pipeline(&graphics.pipeline);

                    for (slot, (buf, _size, _stride)) in vertex_buffers.iter() {
                        pass.set_vertex_buffer(*slot, buf.slice(..));
                    }

                    // pass.set_bind_group(index, bind_group, offsets)
                    for (stage, bind_groups) in op_state.bind_groups.iter() {
                        for (ix, group) in bind_groups.iter().enumerate() {
                            pass.set_bind_group(ix as u32, group, &[]);
                        }
                    }

                    let (vertices, instances) = {
                        let (_, size, stride) = vertex_buffers[0].1;

                        // TODO: optionally set vertices/instances via NodeOpState
                        let stride =
                            stride.expect("vertex buffer needs stride to draw");

                        let count = (size / stride) as u32;
                        ((0..count), (0..1))
                    };

                    op_state.set_render_push_constants(&mut pass);

                    pass.draw(vertices, instances);
                }
            }
            NodeOpId::Compute(i) => {
                let compute = &self.compute[*i];

                let label = format!("Node {} - compute pass", node_id.0);
                let desc = wgpu::ComputePassDescriptor {
                    label: Some(label.as_str()),
                };

                let mut pass = encoder.begin_compute_pass(&desc);

                pass.set_pipeline(&compute.pipeline);

                for (stage, bind_groups) in op_state.bind_groups.iter() {
                    if *stage == naga::ShaderStage::Compute {
                        for (ix, group) in bind_groups.iter().enumerate() {
                            pass.set_bind_group(ix as u32, group, &[]);
                        }
                    }
                }

                op_state.set_compute_push_constants(&mut pass);

                let [x, y, z] = op_state.workgroup_count.unwrap();

                pass.dispatch_workgroups(x, y, z);
            }
            NodeOpId::NoOp => (),
        }

        Ok(true)
    }

    pub fn create_compute_schema(
        &mut self,
        state: &State,
        comp_src: &[u8],
        schema_id: NodeSchemaId,
    ) -> Result<NodeSchema> {
        let comp = ComputeShader::from_spirv(&state, comp_src, "main")?;

        let op_id = NodeOpId::Compute(self.compute.len());

        let mut socket_names = Vec::new();
        let mut socket_bindings = Vec::new();

        // add shader sockets; just bind groups in this case
        for group in comp.group_bindings.iter() {
            for binding in group.bindings.iter() {
                let ix = socket_names.len();
                socket_bindings.push((
                    ix,
                    SocketBinding::compute_bind_group(
                        group.group_ix,
                        binding.binding.binding,
                    ),
                ));
                socket_names.push(binding.global_var_name.clone());
            }
        }

        socket_bindings.sort();

        let schema = NodeSchema {
            node_type: NodeType::Compute,
            schema_id,

            socket_names,
            source_sockets: vec![],

            source_rules_sockets: vec![],
            source_rules_scalars: vec![],

            socket_bindings,

            default_sources: HashMap::default(),
            create_source_metadata: HashMap::default(),
        };

        self.compute.push(comp);
        self.schema_ops.insert(schema_id, op_id);

        Ok(schema)
    }

    pub fn create_graphics_schema(
        &mut self,
        state: &State,
        vert_src: &[u8],
        frag_src: &[u8],
        frag_out_formats: &[wgpu::TextureFormat],
        schema_id: NodeSchemaId,
    ) -> Result<NodeSchema> {
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
        let mut socket_bindings = Vec::new();

        // add vertex shader sockets
        //   vertex buffers
        // hardcoded to a single vertex buffer
        socket_bindings.push((
            socket_names.len(),
            SocketBinding::VertexBuffer { slot: 0 },
        ));
        socket_names.push("vertex_in".into());

        //   bind groups
        for group in vert.group_bindings.iter() {
            for binding in group.bindings.iter() {
                let ix = socket_names.len();
                socket_bindings.push((
                    ix,
                    SocketBinding::vertex_bind_group(
                        group.group_ix,
                        binding.binding.binding,
                    ),
                ));
                socket_names.push(binding.global_var_name.clone());
            }
        }

        // add fragment shader sockets
        //   render attachments
        for output in frag.fragment_outputs.iter() {
            socket_bindings.push((
                socket_names.len(),
                SocketBinding::FragmentAttachment {
                    location: output.location,
                },
            ));
            socket_names.push(output.name.as_str().into());
        }

        //   bind groups
        for group in frag.group_bindings.iter() {
            for binding in group.bindings.iter() {
                let ix = socket_names.len();
                socket_bindings.push((
                    ix,
                    SocketBinding::fragment_bind_group(
                        group.group_ix,
                        binding.binding.binding,
                    ),
                ));
                socket_names.push(binding.global_var_name.clone());
            }
        }

        socket_bindings.sort();

        let schema = NodeSchema {
            node_type: NodeType::Graphics,
            schema_id,

            socket_names,
            source_sockets: vec![],

            source_rules_sockets: vec![],
            source_rules_scalars: vec![],

            socket_bindings,

            default_sources: HashMap::default(),
            create_source_metadata: HashMap::default(),
        };

        self.graphics.push(graphics);

        self.schema_ops.insert(schema_id, op_id);

        Ok(schema)
    }
}

struct SocketAnnotations {
    name_id_cache: HashMap<String, usize>,
    annotations: BTreeMap<(NodeId, LocalSocketIx), (usize, rhai::Map)>,
}

#[derive(Debug, Default)]
pub struct NodeOpState {
    bind_groups: FxHashMap<naga::ShaderStage, Vec<wgpu::BindGroup>>,
    node_parameters: rhai::Map,

    vertex_buffers: BTreeMap<u32, LocalSocketIx>,
    attachments: BTreeMap<u32, LocalSocketIx>,

    pub push_constants: FxHashMap<naga::ShaderStage, PushConstants>,

    // only used by compute
    workgroup_count: Option<[u32; 3]>,
}

impl NodeOpState {
    fn get_push_constant_data(
        &self,
        stage: naga::ShaderStage,
    ) -> Option<&[u8]> {
        let wgpu_stage = match stage {
            naga::ShaderStage::Vertex => wgpu::ShaderStages::VERTEX,
            naga::ShaderStage::Fragment => wgpu::ShaderStages::FRAGMENT,
            naga::ShaderStage::Compute => wgpu::ShaderStages::COMPUTE,
        };

        self.push_constants.get(&stage).map(|c| c.data())
    }

    fn set_render_push_constants(&self, pass: &mut RenderPass) {
        let mut offset = 0;

        let v_data = self.get_push_constant_data(naga::ShaderStage::Vertex);
        let f_data = self.get_push_constant_data(naga::ShaderStage::Fragment);

        let vert = wgpu::ShaderStages::VERTEX;
        let frag = wgpu::ShaderStages::FRAGMENT;

        for (data, stage) in [(v_data, vert), (f_data, frag)].into_iter() {
            if let Some(data) = data {
                let end = offset + data.len() as u32;
                pass.set_push_constants(stage, offset, data);
                offset = end;
            }
        }
    }

    fn set_compute_push_constants(&self, pass: &mut ComputePass) {
        let data = self.get_push_constant_data(naga::ShaderStage::Compute);

        if let Some(data) = data {
            pass.set_push_constants(0, data);
        }
    }
}

fn create_bind_group_entries<'a>(
    // (group_ix, binding_ix) -> local socket
    resource_ctx: &'a ResourceCtx<'a>,
    // socket_res_refs: &'a BTreeMap<(NodeId, LocalSocketIx), ResourceRef>,
    // owned_res: &'a [Option<Resource>],
    // res_meta: &[ResourceMeta],
    // transient_res: &'a FxHashMap<TransientId, &'a InputResource<'a>>,
    bind_sockets: &BTreeMap<(u32, u32), LocalSocketIx>,
    group_bindings: &[GroupBindings],
    node_id: NodeId,
) -> Result<Vec<Vec<wgpu::BindGroupEntry<'a>>>> {
    let vert_group_entries = group_bindings
        .iter()
        .map(|group| {
            let mut entries = Vec::new();
            for (b_ix, l_entry) in group.entries.iter().enumerate() {
                let key = (group.group_ix, b_ix as u32);
                let socket = bind_sockets.get(&key).unwrap();

                let in_res = resource_ctx
                    .get_resource_at_socket(node_id, *socket)
                    .unwrap();

                let entry = in_res
                    .create_bind_group_entry(b_ix as u32, l_entry)
                    .unwrap();
                entries.push(entry);
            }
            entries
        })
        .collect::<Vec<_>>();

    Ok(vert_group_entries)
}
