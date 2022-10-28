use std::collections::VecDeque;

use ultraviolet::{DVec2, Vec2};
use winit::dpi::PhysicalPosition;

#[derive(Debug, Clone, Copy)]
struct TouchState {
    id: u64,
    start: DVec2,
    end: DVec2,
}

#[derive(Default)]
pub struct TouchHandler {
    touches: VecDeque<TouchState>,
    last_result: Option<TouchResult>,
}


#[derive(Debug, Clone, Copy)]
pub enum TouchResult {
    Drag {
        pos: DVec2,
        delta: DVec2,
    },
    Pinch {
        pos_0: DVec2,
        delta_0: DVec2,
        pos_1: DVec2,
        delta_1: DVec2,
    },
}

impl TouchHandler {
    pub fn take_current_result(&mut self) -> Option<TouchResult> {
        self.touches.iter_mut().for_each(|t| t.reset_to(t.end));
        self.last_result.take()
    }

    pub fn has_touch(&self) -> bool {
        !self.touches.is_empty()
    }

    pub fn handle_touch(
        &mut self,
        touch: &winit::event::Touch,
    ) -> Option<TouchResult> {
        use winit::event::TouchPhase;
        let loc: winit::dpi::PhysicalPosition<f64> = touch.location;
        let id = touch.id;

        match touch.phase {
            TouchPhase::Started => {
                let state = TouchState::new(id, loc);
                self.touches.push_back(state);
                None
            }
            TouchPhase::Moved => {
                if let Some(state) = self.find_state_mut(touch.id) {
                    state.update_end(touch.location);
                }
                let result = self.result();
                self.last_result = result;
                result
            }
            TouchPhase::Ended => {
                self.remove_state(touch.id);
                None
            }
            TouchPhase::Cancelled => {
                self.remove_state(touch.id);
                None
            }
        }
    }

    fn result(&self) -> Option<TouchResult> {
        match self.touches.len() {
            0 => None,
            1 => {
                let delta = self.touch_delta(0).unwrap();
                let pos = self.touches[0].end;
                Some(TouchResult::Drag {
                    pos,
                    delta,
                })
            }
            _n => {
                let delta_0 = self.touch_delta(0).unwrap();
                let delta_1 = self.touch_delta(1).unwrap();
                let p0 = self.touches[0].end;
                let p1 = self.touches[1].end;

                Some(TouchResult::Pinch {
                    pos_0: p0,
                    delta_0,
                    pos_1: p1,
                    delta_1,
                })
            }
        }
    }

    fn touch_delta(&self, index: usize) -> Option<DVec2> {
        let touch = self.touches.get(index)?;
        Some(touch.end - touch.start)
    }

    fn remove_state(&mut self, id: u64) {
        if let Some(ix) = self.touches.iter().position(|s| s.id == id) {
            self.touches.remove(ix);
        }
    }

    fn find_state_mut(&mut self, id: u64) -> Option<&mut TouchState> {
        self.touches.iter_mut().find(|s| s.id == id)
    }
}

impl TouchState {
    fn new(id: u64, start: PhysicalPosition<f64>) -> Self {
        let start = DVec2::new(start.x, start.y);
        Self {
            id,
            start,
            end: start,
        }
    }

    fn reset_to(&mut self, start: DVec2) {
        self.start = start;
        self.end = start;
    }

    fn update_end(&mut self, end: PhysicalPosition<f64>) {
        let end = DVec2::new(end.x, end.y);
        self.end = end;
    }
}
