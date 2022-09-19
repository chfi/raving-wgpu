
use raving_wgpu::State;


pub fn main() {
    if let Err(e) = pollster::block_on(raving_wgpu::run()) {
        log::error!("{:?}", e);
    }
}
