use std::collections::VecDeque;

use ultraviolet::{f32x4, Isometry3, Mat4, Rotor3, Vec2, Vec3};
use winit::event::WindowEvent;

// returns the distances in clockwise order from top: [U, R, D, L]
// distances *are* signed
pub fn dist_to_rect_sides(
    rect_center: Vec2,
    rect_size: Vec2,
    p: Vec2,
) -> f32x4 {
    let o = rect_center;
    let s = rect_size;

    let w2 = s.x / 2.0;
    let h2 = s.y / 2.0;

    let base = f32x4::new([o.y, o.x, o.y, o.x]);
    let ds = f32x4::new([-h2, w2, h2, -w2]);
    let p_ = f32x4::new([p.y, p.x, p.y, p.x]);
    let p_ = p_ - base;

    let sides = ds;

    let result = sides - p_;

    // flip top and left so that positive values are inside
    result.flip_signs([-1f32, 1.0, 1.0, -1.0].into())
}

// pub fn side_dists_to_2d(dists_urdl: f32x4) -> (Vec2, Vec2) {
//     let [u, r, d, l] = dists_urdl.to_array();
//     let w = r + l;
//     let h = u + d;
//     let pos = Vec2::new(r, u);
//     let size = Vec2::new(w, h);
//     (pos, size)
// }

pub struct TouchState {
    id: u64,
    origin: Vec2,
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
            let pos = touch.start;
            let end = touch.end;
            touch.start = end;

            let delta = end - pos;

            TouchOutput {
                origin: touch.origin,
                pos,
                delta,
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

            let pos = loc / size;
            // dbg!(&pos);

            use winit::event::TouchPhase as Phase;

            match touch.phase {
                Phase::Started => {
                    self.touches.push_back(TouchState {
                        id,
                        origin: pos,
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
    pub origin: Vec2,
    pub pos: Vec2,
    pub delta: Vec2,
}

impl TouchOutput {
    pub fn flip_y(mut self) -> Self {
        // i think this is correct...
        self.pos.y = 1.0 - self.pos.y;
        self.delta.y *= -1.0;
        // self.delta.y = 1.0 - self.delta.y;
        // self.pos.y *= -1.0;
        self
    }
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

    pub fn displacement(&self) -> Vec2 {
        self.center - self.prev_center
    }

    pub fn pinch_anchored(&mut self, anchor: Vec2, start: Vec2, end: Vec2) {
        let d0 = start - anchor;
        let d = end - start;

        // let s_dists = dist_to_rect_sides(self.center, self.size, start);
        // dbg!(s_dists.to_array());
        let s_dists = dist_to_rect_sides(
            self.center, //
            self.size,   //
            start,
        );

        let darray = s_dists.to_array();
        println!("{darray:.4?}");
        let [s_u, s_r, s_d, s_l] = s_dists.to_array();

        let p = start;

        // let top = self.size.y - start.y;
        // let bottom =

        // horizontal component
        let dw = d.x.abs();
        // let t = (p.y - self.size.y) / self.size.y;
        // let dh = d.

        let split_h = s_u / (s_u + s_d);
        let split_v = s_l / (s_l + s_r);

        let r_xy = self.size.x / self.size.y;
        let r_yx = self.size.y / self.size.x;

        // let prop_up = t;
        // let prop_down = 1.0 - t;

        let old_size = self.size;


        // dbg!(&(prop_up, prop_down));
        dbg!(&split_h);

        if d.x > 0.0 {
            // extrude to the right, then equalize
            self.size.x += dw;

            let new_h = self.size.x * r_yx;
            let diff = new_h - old_size.y;
            self.size.y = new_h;

            self.center.y -= diff * split_h;

        } else {
            // extrude to the left, then equalize
            self.size.x += dw;
            self.center.x -= dw;
            
            let new_h = self.size.x * r_yx;
            let diff = new_h - old_size.y;
            self.size.y = new_h;

            self.center.y -= diff * split_h;
        }

    }

    // all positions and deltas should be given in world units,
    // centered on the view
    pub fn pinch(
        &mut self,
        start_0: Vec2,
        end_0: Vec2,
        start_1: Vec2,
        end_1: Vec2,
    ) {
        let aspect = self.size.x / self.size.y;
        // map position axes to view sides

        // apply deltas to view sides
    }

    /// `fix` should be relative to center, in normalized screen coordinates
    pub fn scale_uniformly_around(&mut self, fix: Vec2, scale: f32) {
        let n = self.center - fix;
        // dbg!(&fix);
        // dbg!(&n);

        // the current world distance
        // let dist = (fix * self.size).mag();
        // the new world distance
        // let n_dist = dist * scale;

        let sign = (scale - 1.0).signum();

        // self.center -= (n * scale) * sign;
        self.prev_center = self.center;

        let size = self.size * scale;
        self.size += size;
    }

    pub fn resize_relative(&mut self, scale: Vec2) {
        // println!("resizing with {scale:?}");
        self.size *= scale;
    }

    pub fn blink(&mut self, n: Vec2) {
        let size = self.size;
        // println!("blink with {n:?} * {size:?} = {:?}", n * self.size);
        self.center += n * self.size;
        // self.prev_center = self.center;
    }

    pub fn nudge(&mut self, n: Vec2) {
        self.accel += n * self.size * self.size;
        // self.accel += n * self.size;
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
        let right = self.size.x / 2.0;
        let left = -right;
        let top = self.size.y / 2.0;
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
