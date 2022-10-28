use ultraviolet::{Mat4, Vec2};

pub struct DynamicCamera2d {
    size: Vec2,
    center: Vec2,
    prev_center: Vec2,

    accel: Vec2,
}

impl DynamicCamera2d {
    pub fn new(center: Vec2, size: Vec2) -> Self {
        Self {
            size,
            center,
            prev_center: center,
            accel: Vec2::zero(),
        }
    }

    pub fn update(&mut self, dt: f32) {
        let v = self.center - self.prev_center;
        self.prev_center = self.center;
        self.center += v * self.accel * dt * dt;
        self.accel = Vec2::zero();
    }

    pub fn to_matrix(&self) -> Mat4 {
        let p = self.center;
        let size = self.size;

        let min = p - size * 0.5;
        let max = p + size * 0.5;

        let left = min.x;
        let right = max.x;
        let bottom = min.y;
        let top = max.y;

        let near = 1.0;
        let far = 1000.0;

        ultraviolet::projection::rh_yup::orthographic_wgpu_dx(
            left, right, bottom, top, near, far,
        )
    }
}
