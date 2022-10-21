use std::collections::{BTreeMap, HashMap, HashSet};

use datafrog::{Iteration, Relation, RelationLeaper, Variable};
use rustc_hash::FxHashMap;
use wgpu::{SubmissionIndex, TextureUsages};

use std::sync::Arc;

use crate::{
    texture::Texture, ComputeShaderOp, DataType, GraphicsPipelineOp, State,
};

use anyhow::Result;

use super::NodeId;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct NodeSchemaId(usize);

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

    create_source_metadata: Option<
        Arc<dyn Fn(&rhai::Map, Option<ResourceMeta>) -> Result<ResourceMeta>>,
    >,
}

/*
fn create_image_schema(schema_id: NodeSchemaId) -> NodeSchema {
    let socket_names = Relation::from_iter(["image".to_string()]);
    let source_sockets = Relation::from_iter([(0, DataType::Texture)]);

    NodeSchema {
        node_type: NodeType::Resource,
        schema_id,

        socket_names,
        source_sockets,
        source_rules_sockets: Relation::from_vec(Vec::new()),

        default_sources:
    }
}
*/

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ResourceMetadataEntry {
    BufferSize,
    BufferUsage,

    TextureSize,
    TextureFormat,
    TextureUsage,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SocketMetadataSource {
    other_socket_ix: usize,
    entry: ResourceMetadataEntry,
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

    // node_names: Vec<rhai::ImmutableString>,
    resource_meta: Vec<ResourceMeta>,
    resources: Vec<Option<Resource>>,

    transients_meta_cache: HashMap<String, ResourceMeta>,
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

impl Graph {
    pub fn new() -> Self {
        Self {
            schemas: Vec::new(),
            nodes: Vec::new(),
            node_names: FxHashMap::default(),

            resource_meta: Vec::new(),
            resources: Vec::new(),

            socket_resources: BTreeMap::default(),
            transients_meta_cache: HashMap::default(),
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

        /*
            update transients cache
        */

        let mut invalidated_transients = Vec::new();

        for (key, in_res) in transient_res.iter() {
            let meta = match in_res {
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
            };

            if let Some(cache) = self.transients_meta_cache.get_mut(key) {
                if cache != &meta {
                    *cache = meta;
                    invalidated_transients.push(key);
                }
            } else {
                self.transients_meta_cache.insert(key.to_string(), meta);
                invalidated_transients.push(key);
            }
        }

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
                    ((node_id, out_ix), (to_node, in_ix))
                })
            }),
        );

        let inv_links = Relation::from_map(&links, |&(from, to)| (to, from));

        /*
         goal: match each socket up with a `resource_origin`

         socket_resources((node_id, socket_ix), res_id)
           :-

         socket_origins((node_id, socket_ix), res_id)

        */

        /*
        let socket_origins = Relation::from_iter(
            self.nodes.iter().enumerate().flat_map(|(ix, node)| {
                let node_id = NodeId::from(ix);

                let schema = &self.schemas[node.schema.0];
                schema.source_sockets.iter().map(|(socket, _ty)| {

                });

            })
        );
        */

        let mut iteration = datafrog::Iteration::new();

        let socket_resources = iteration
            .variable::<((NodeId, LocalSocketIx), ResourceId)>(
                "socket_resources",
            );

        /*
         this is where NodeScheme's create_source_metadata comes in;

         for each resource, call the corresponding function on the schema,
         to update the entries in self.resource_meta
        */

        /*

        */

        // return true if all sockets point to owned resources
        // or cached transient resources

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

    /*
    pub fn resolve_sockets(&mut self) {

        // let all_source_sockets = Relation::from_leapjoin(source, leapers, logic)

        let all_sockets = {
            /*

            */
        };

        // let to_resolve = Relation::from_antijoin(input1, input2, logic)
    }
    */

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

            let create_source_metadata = |scalars: &rhai::Map,
                                          meta: Option<ResourceMeta>|
             -> Result<ResourceMeta> {
                todo!();
            };

            let schema = NodeSchema {
                node_type: NodeType::Resource,
                schema_id: id,

                socket_names: vec!["image".into()],

                source_sockets: vec![(0, DataType::Texture)],
                source_rules_sockets: vec![],
                source_rules_scalars: vec![(0, "dimensions".into())],

                default_sources,

                create_source_metadata: Some(Arc::new(create_source_metadata)),
            };

            self.schemas.push(schema);
            id
        };

        /*
        {
            let id = NodeSchemaId(self.schemas.len());

            let schema = NodeSchema {
                node_type: NodeType::Compute,
                schema_id: id,

                socket_names: vec![
                    "image_in".into(),
                    "image_out".into(),
                    "buffer_in".into(),
                    "buffer_out".into(),
                ],

                source_sockets: vec![
                    (1, DataType::Texture),
                    (3, DataType::Buffer),
                ],

                source_rules_sockets: vec![
                    (
                        1,
                        SocketMetadataSource {
                            other_socket_ix: 0,
                            entry: ResourceMetadataEntry::TextureFormat,
                        },
                    ),
                    (
                        1,
                        SocketMetadataSource {
                            other_socket_ix: 0,
                            entry: ResourceMetadataEntry::TextureSize,
                        },
                    ),
                ],

                source_rules_scalars: vec![],

                default_sources: FxHashMap::default(),
            };

            self.schemas.push(schema);
            id
        };
        */

        Ok(())
    }
}

pub struct GraphOps {
    graphics: Vec<GraphicsPipelineOp>,
    compute: Vec<ComputeShaderOp>,
}
