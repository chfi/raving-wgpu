
use wgpu::*;


#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct ImageResSignature {
    dimensions: [u32; 2],
    format: wgpu::TextureFormat,
    usage: wgpu::TextureUsages,
}

