use rustc_hash::FxHashMap;
use wgpu::*;

use anyhow::Result;
use std::{collections::HashMap, sync::Arc};

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
pub enum DataType {
    Buffer,
    Image,
    Scalar,
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

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct NodeId(u64);

pub type SocketIx = usize;

pub struct Node {
    def: Arc<NodeDef>,
    id: NodeId,

    inputs: Vec<Option<(NodeId, SocketIx)>>,
    outputs: Vec<Option<(NodeId, SocketIx)>>,
}

pub struct Graph {
    defs: HashMap<String, Arc<NodeDef>>,
    nodes: Vec<Node>,
}

impl Graph {
    pub fn init() -> Result<Self> {
        let mut defs = HashMap::new();

        let alloc_img_def = NodeDef::new("create_image", [], [("image", DataType::Image)]);

        defs.insert(alloc_img_def.name.clone(), Arc::new(alloc_img_def));

        let nodes = Vec::new();

        Ok(Self { defs, nodes })
    }
}
