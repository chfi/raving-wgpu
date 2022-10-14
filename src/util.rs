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
