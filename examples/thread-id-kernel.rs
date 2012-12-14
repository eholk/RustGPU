#[abi = "rust-intrinsic"]
extern mod gpu {
    fn ptx_tid_x() -> i32;
}

#[kernel]
fn set_tid(x: &float, y: &float, z: &mut float) {
    use gpu::*;

    let idx = ptx_tid_x();
    //let idx = 3;
    if idx == 3 {
        let x = *x;
        let y = *y;
        *z = x * x + y * y * (idx as float);
    }
}