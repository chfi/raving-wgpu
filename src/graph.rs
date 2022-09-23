
use rustc_hash::FxHashMap;
use wgpu::*;

use std::sync::Arc;


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
    input_sockets: Vec<(String, DataType)>,
    output_sockets: Vec<(String, DataType)>,
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

