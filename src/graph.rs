use rustc_hash::{FxHashMap, FxHashSet};
use wgpu::*;

use anyhow::{anyhow, bail, Result};
use std::{
    collections::{HashMap, VecDeque},
    sync::Arc,
};

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
pub struct NodeId(usize);

pub type SocketIx = usize;

pub struct Node {
    def: Arc<NodeDef>,
    id: NodeId,

    inputs: Vec<Option<(NodeId, SocketIx)>>,
    outputs: Vec<Option<(NodeId, SocketIx)>>,
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
