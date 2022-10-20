use std::collections::HashSet;

use datafrog::{Iteration, Relation, RelationLeaper, Variable};
use rustc_hash::FxHashMap;
use wgpu::TextureUsages;

use crate::{texture::Texture, ComputeShaderOp, DataType, GraphicsPipelineOp};

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
    source_rules: Vec<(LocalSocketIx, SocketMetadataSource)>,

    default_sources: FxHashMap<LocalSocketIx, ResourceMeta>,
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
        source_rules: Relation::from_vec(Vec::new()),

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

#[derive(Debug, Clone, Copy)]
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

pub enum Resource {
    Buffer(wgpu::Buffer),
    Texture(Texture),
}

pub struct Node {
    schema: Option<NodeSchemaId>,

    links: Vec<(LocalSocketIx, (NodeId, LocalSocketIx))>,
}

pub struct Graph {
    schemas: Vec<NodeSchema>,

    nodes: Vec<Node>,
    // nodes: Vec<NodeId>,
    node_names: FxHashMap<NodeId, rhai::ImmutableString>,
    // node_names: Vec<rhai::ImmutableString>,
    resource_defs: Vec<ResourceMeta>,
    resources: Vec<Option<Resource>>,

    // links: FxHashMap<NodeId, Vec<(usize, NodeId, usize)>>,

    // node_schema: Relation<(NodeId, NodeSchemaId)>,

    // node_schema: Vec<NodeSchema>,
    // node_schema: Vec<NodeSchemaId>,

    // links: Relation<(NodeId, (LocalSocketIx, NodeId, LocalSocketIx))>,
    // links: Relation<((NodeId, LocalSocketIx), (NodeId, LocalSocketIx))>,

    // resource_meta: Vec<ResourceMeta>,
    // resources: Vec<Option<Resource>>,
    // socket_resources: Relation<((NodeId, LocalSocketIx), ResourceId)>,
    // resources: Vec<()>,
    // socket_resources: Relation<(NodeId, (LocalSocketIx, ResourceId))>,

    // local_socket_names: Relation<
}

impl Graph {
    pub fn new() -> Self {
        Self {
            schemas: Vec::new(),
            nodes: Vec::new(),
            node_names: FxHashMap::default(),

            resource_defs: Vec::new(),
            resources: Vec::new(),
        }
    }

    pub fn add_node(&mut self, schema: Option<NodeSchemaId>) -> NodeId {
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
    

    pub fn add_schemas(&mut self) -> Result<()>  {

        let image_schema = {
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

            let schema = NodeSchema {
                node_type: NodeType::Resource,
                schema_id: id,

                socket_names: vec!["image".into()],

                source_sockets: vec![(0, DataType::Texture)],
                source_rules: vec![],

                default_sources,
            };

            self.schemas.push(schema);
            id
        };

        let compute_node_schema = {
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

                source_rules: vec![
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

                default_sources: FxHashMap::default(),
            };

            self.schemas.push(schema);
            id
        };

        Ok(())
    }
}

pub struct GraphOps {
    graphics: Vec<GraphicsPipelineOp>,
    compute: Vec<ComputeShaderOp>,
}
