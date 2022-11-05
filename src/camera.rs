use std::collections::VecDeque;

use ultraviolet::{Isometry3, Mat4, Rotor3, Vec2, Vec3};
use winit::event::WindowEvent;

pub struct TouchState {
    id: u64,
    start: Vec2,
    end: Vec2,
}

#[derive(Default)]
pub struct TouchHandler {
    touches: VecDeque<TouchState>,
}

impl TouchHandler {
    pub fn take<'a>(&'a mut self) -> impl Iterator<Item = TouchOutput> + 'a {
        self.touches.iter_mut().map(|touch| {
            let mut pos = touch.start;
            let mut end = touch.end;
            touch.start = end;
            
            // flip to match world Y axis
            pos.y *= -1.0;
            end.y *= -1.0;
            let delta = end - pos;

            TouchOutput {
                pos,
                delta
            }
        })
    }
    
    // returns `true` if consumed
    pub fn on_event(
        &mut self,
        window_dims: [u32; 2],
        event: &WindowEvent,
    ) -> bool {
        if let WindowEvent::Touch(touch) = event {
            let id = touch.id;

            let loc = touch.location;
            let loc = Vec2::new(loc.x as f32, loc.y as f32);

            let [w, h] = window_dims;
            let size = Vec2::new(w as f32, h as f32);

            // let pos = loc / size;
            let pos = Vec2::broadcast(-0.5) + (2.0 * loc) / size;
            // let pos = Vec2::new(0.5, 0.5) + loc / (size * 2.0);
            // let pos = loc / (size * 2.0);

            println!("{pos:?}");

            use winit::event::TouchPhase as Phase;

            match touch.phase {
                Phase::Started => {
                    self.touches.push_back(TouchState {
                        id,
                        start: pos,
                        end: pos,
                    });
                }
                Phase::Moved => {
                    if let Some(touch) =
                        self.touches.iter_mut().find(|t| t.id == id)
                    {
                        touch.end = pos;
                    }
                }
                Phase::Ended | Phase::Cancelled => {
                    self.touches.iter().position(|t| t.id == id).map(|i| {
                        self.touches.remove(i);
                    });
                }
            }

            return true;
        }

        false
    }
}

#[derive(Debug, Clone, Copy)]
pub struct TouchOutput {
    pub pos: Vec2,
    pub delta: Vec2,
}

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

    /// `fix` should be relative to center, in normalized screen coordinates
    pub fn scale_uniformly_around(&mut self, fix: Vec2, scale: f32) {
        let n = fix.normalized();

        // the current world distance
        let dist = (fix * self.size).mag();
        // the new world distance
        let n_dist = dist * scale;

        let diff = n_dist / dist;
        let sign = (scale - 1.0).signum();

        self.center -= (n * diff) * sign;
        self.prev_center = self.center;
        self.size = self.size * scale;
    }

    pub fn resize_relative(&mut self, scale: Vec2) {
        println!("resizing with {scale:?}");
        self.size *= scale;
    }

    pub fn blink(&mut self, n: Vec2) {
        self.center += n * self.size;
        self.prev_center = self.center;
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

        let view = Isometry3::new(p_, Rotor3::identity()).inversed();

        proj * view.into_homogeneous_matrix()
    }
}
