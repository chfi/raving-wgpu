pub fn format_naga_to_wgpu(format: naga::StorageFormat) -> wgpu::TextureFormat {
    match format {
        naga::StorageFormat::R8Unorm => wgpu::TextureFormat::R8Unorm,
        naga::StorageFormat::R8Snorm => wgpu::TextureFormat::R8Snorm,
        naga::StorageFormat::R8Uint => wgpu::TextureFormat::R8Uint,
        naga::StorageFormat::R8Sint => wgpu::TextureFormat::R8Sint,

        naga::StorageFormat::R16Uint => wgpu::TextureFormat::R16Uint,
        naga::StorageFormat::R16Sint => wgpu::TextureFormat::R16Sint,
        naga::StorageFormat::R16Unorm => wgpu::TextureFormat::R16Unorm,
        naga::StorageFormat::R16Snorm => wgpu::TextureFormat::R16Snorm,
        naga::StorageFormat::R16Float => wgpu::TextureFormat::R16Float,

        naga::StorageFormat::Rg8Unorm => wgpu::TextureFormat::Rg8Unorm,
        naga::StorageFormat::Rg8Snorm => wgpu::TextureFormat::Rg8Snorm,

        naga::StorageFormat::Rg8Uint => wgpu::TextureFormat::Rg8Uint,
        naga::StorageFormat::Rg8Sint => wgpu::TextureFormat::Rg8Sint,

        naga::StorageFormat::R32Uint => wgpu::TextureFormat::R32Uint,
        naga::StorageFormat::R32Sint => wgpu::TextureFormat::R32Sint,
        naga::StorageFormat::R32Float => wgpu::TextureFormat::R32Float,

        naga::StorageFormat::Rg16Uint => wgpu::TextureFormat::Rg16Uint,
        naga::StorageFormat::Rg16Sint => wgpu::TextureFormat::Rg16Sint,
        naga::StorageFormat::Rg16Float => wgpu::TextureFormat::Rg16Float,
        naga::StorageFormat::Rg16Unorm => wgpu::TextureFormat::Rg16Unorm,
        naga::StorageFormat::Rg16Snorm => wgpu::TextureFormat::Rg16Snorm,

        naga::StorageFormat::Rgba8Unorm => wgpu::TextureFormat::Rgba8Unorm,
        naga::StorageFormat::Rgba8Snorm => wgpu::TextureFormat::Rgba8Snorm,
        naga::StorageFormat::Rgba8Uint => wgpu::TextureFormat::Rgba8Uint,
        naga::StorageFormat::Rgba8Sint => wgpu::TextureFormat::Rgba8Sint,

        naga::StorageFormat::Rgb10a2Unorm => wgpu::TextureFormat::Rgb10a2Unorm,
        naga::StorageFormat::Rg11b10Float => wgpu::TextureFormat::Rg11b10Float,

        naga::StorageFormat::Rg32Uint => wgpu::TextureFormat::Rg32Uint,
        naga::StorageFormat::Rg32Sint => wgpu::TextureFormat::Rg32Sint,
        naga::StorageFormat::Rg32Float => wgpu::TextureFormat::Rg32Float,

        naga::StorageFormat::Rgba16Uint => wgpu::TextureFormat::Rgba16Uint,
        naga::StorageFormat::Rgba16Sint => wgpu::TextureFormat::Rgba16Sint,
        naga::StorageFormat::Rgba16Unorm => wgpu::TextureFormat::Rgba16Unorm,
        naga::StorageFormat::Rgba16Snorm => wgpu::TextureFormat::Rgba16Snorm,
        naga::StorageFormat::Rgba16Float => wgpu::TextureFormat::Rgba16Float,

        naga::StorageFormat::Rgba32Uint => wgpu::TextureFormat::Rgba32Uint,
        naga::StorageFormat::Rgba32Sint => wgpu::TextureFormat::Rgba32Sint,
        naga::StorageFormat::Rgba32Float => wgpu::TextureFormat::Rgba32Float,
    }
}

#[macro_export]
macro_rules! get_field_impl {
    ($map:expr, $field:expr => $ty:ty, $f:ident) => {
        $map.get($field)
            .ok_or_else(|| format!("Field `{}` not found", $field))
            .and_then(|f| {
                if f.type_id() != std::any::TypeId::of::<$ty>() {
                    return Err(format!(
                        "Field `{}` type mismatch: was {}, expected {}",
                        $field,
                        f.type_name(),
                        stringify!($ty),
                    ));
                }
                let v = f.$f()?;
                Ok(v)
            })
    };
}

#[macro_export]
macro_rules! get_field_alt {
    ($map:expr, $field:expr => f32) => {
        crate::get_field_impl!($map, $field => f32, as_float)
    };
    ($map:expr, $field:expr => i64) => {
        crate::get_field_impl!($map, $field => i64, as_int)
    };
    ($map:expr, $field:expr => i32) => {
        crate::get_field_impl!($map, $field => i64, as_int)
            .map(|i| i as i32)
    };
    ($map:expr, $field:expr => i16) => {
        crate::get_field_impl!($map, $field => i64, as_int)
            .map(|i| i as i16)
    };
    ($map:expr, $field:expr => i8) => {
        crate::get_field_impl!($map, $field => i64, as_int)
            .map(|i| i as i8)
    };
    ($map:expr, $field:expr => u64) => {
        crate::get_field_impl!($map, $field => i64, as_int)
            .map(|i| i as u64)
    };
    ($map:expr, $field:expr => u32) => {
        crate::get_field_impl!($map, $field => i64, as_int)
            .map(|i| i as u32)
    };
    ($map:expr, $field:expr => u16) => {
        crate::get_field_impl!($map, $field => i64, as_int)
            .map(|i| i as u16)
    };
    ($map:expr, $field:expr => u8) => {
        crate::get_field_impl!($map, $field => i64, as_int)
            .map(|i| i as u8)
    };
}

#[macro_export]
macro_rules! get_map_fields {
    ($map:expr, $field:ident => $ty:ty) => {
        let $field = crate::get_field_impl!($map, stringify!($field) => $ty, as_int);
    };
    ($map:expr, $field:ident => $ty:ty, $($next:ident => $nt:ty),*) => {
        get_map_fields!($map, $field => $ty:ty);
        get_map_fields!($map, $($next => $nt:ty),*);
    };
}

#[macro_export]
macro_rules! get_map_fields_map_err {
    ($post:expr, $map:expr, $field:ident => $ty:ty) => {
        let $field = {
            let value = crate::get_field_impl!($map, stringify!($field) => $ty, as_int);
            value.map_err($post)?
        };
    };
    ($post:expr, $map:expr, $field:ident => $ty:ty, $($next:ident => $nt:ty),*) => {
        get_map_fields_map_err!($post, $map, $field => $ty);
        get_map_fields_map_err!($post, $map, $($next => $nt),*);
    };
}
