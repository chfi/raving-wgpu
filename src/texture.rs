use std::num::NonZeroU32;

use anyhow::*;
use image::GenericImageView;

#[derive(Debug)]
pub struct Texture {
    pub dimensions: [u32; 2],
    usage: wgpu::TextureUsages,

    pub texture: wgpu::Texture,
    pub view: Option<wgpu::TextureView>,
    pub sampler: Option<wgpu::Sampler>,
}

impl Texture {
    pub fn new(
        device: &wgpu::Device,
        width: usize,
        height: usize,
        format: wgpu::TextureFormat,
        usage: wgpu::TextureUsages,
        label: Option<&str>,
    ) -> Result<Self> {
        let size = wgpu::Extent3d {
            width: width as u32,
            height: height as u32,
            depth_or_array_layers: 1,
        };
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label,
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage,
            view_formats: &[format],
        });

        let dimensions = [width as u32, height as u32];

        let view_desc = wgpu::TextureViewDescriptor {
            label,
            format: Some(format),
            dimension: Some(wgpu::TextureViewDimension::D2),
            ..wgpu::TextureViewDescriptor::default() /*
                                                     aspect: todo!(),
                                                     base_mip_level: 1,
                                                     mip_level_count: 1,
                                                     base_array_layer: todo!(),
                                                     array_layer_count: todo!(),
                                                     */
        };

        // let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        let copy_only =
            wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::COPY_SRC;

        // don't create a view or sampler if this is just for copy (TODO: refine)
        let (view, sampler) = if !usage.intersects(copy_only.complement()) {
            (None, None)
        } else {
            let view = texture.create_view(&view_desc);
            let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                address_mode_w: wgpu::AddressMode::ClampToEdge,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Nearest,
                mipmap_filter: wgpu::FilterMode::Nearest,
                ..Default::default()
            });

            (Some(view), Some(sampler))
        };

        Ok(Self {
            dimensions,
            usage,

            texture,
            view,
            sampler,
        })
    }

    pub fn from_bytes(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        bytes: &[u8],
        label: &str,
    ) -> Result<Self> {
        let img = image::load_from_memory(bytes)?;
        Self::from_image(device, queue, &img, Some(label))
    }

    pub fn from_image(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        img: &image::DynamicImage,
        label: Option<&str>,
    ) -> Result<Self> {
        let rgba = img.to_rgba8();
        let dimensions = img.dimensions();

        let size = wgpu::Extent3d {
            width: dimensions.0,
            height: dimensions.1,
            depth_or_array_layers: 1,
        };
        let usage = wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_DST;
        let format = wgpu::TextureFormat::Rgba8UnormSrgb;
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label,
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage,
            view_formats: &[format],
        });

        let dimensions = [dimensions.0, dimensions.1];

        queue.write_texture(
            wgpu::ImageCopyTexture {
                aspect: wgpu::TextureAspect::All,
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
            },
            &rgba,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(4 * dimensions[0]),
                rows_per_image: Some(dimensions[1]),
            },
            size,
        );

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        Ok(Self {
            dimensions,
            usage,

            texture,
            view: Some(view),
            sampler: Some(sampler),
        })
    }
}
