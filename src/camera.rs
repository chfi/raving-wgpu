use ultraviolet::{Isometry3, Mat4, Rotor3, Vec2, Vec3};

pub struct DynamicCamera2d {
    pub size: Vec2,
    pub center: Vec2,
    pub prev_center: Vec2,

    pub accel: Vec2,
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


    pub fn resize_relative(&mut self, scale: Vec2) {
        println!("resizing with {scale:?}");
        self.size *= scale;
    }

    pub fn nudge(&mut self, n: Vec2) {
        self.accel += n * self.size * self.size;
    }

    pub fn update(&mut self, dt: f32) {
        let v = self.center - self.prev_center;
        self.prev_center = self.center;
        self.center += v + self.accel * dt * dt;
        self.accel = Vec2::zero();
    }
    
    pub fn set_position(&mut self, center: Vec2) {
        self.center = center;
        self.prev_center = center;
    }

    pub fn stop(&mut self) {
        self.prev_center = self.center;
        self.accel = Vec2::zero();
    }

    pub fn to_matrix(&self) -> Mat4 {
        let right = self.size.x;
        let left = -right;
        let top = self.size.y;
        let bottom = -top;

        let near = 1.0;
        let far = 10.0;

        let proj = ultraviolet::projection::rh_yup::orthographic_wgpu_dx(
            left, right, bottom, top, near, far,
        );

        let p = self.center;
        let p_ = Vec3::new(p.x, p.y, 5.0);

        let view = Isometry3::new(p_, Rotor3::identity());

        proj * view.into_homogeneous_matrix().inversed()
    }
}
