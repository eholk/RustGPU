#[abi = "rust-intrinsic"]
extern mod gpu {
    fn ptx_tid_x() -> i32;
}

#[abi = "rust-intrinsic"]
extern mod rusti {
    fn addr_of<T>(&&val: T) -> *T;
}

fn offset(x: *float, i: uint) -> *float {
    let x = x as uint;
    (x + i * 8) as *float
}

#[kernel]
fn add_vector(x: *float, y: *float, z: *float) unsafe {
    let id = gpu::ptx_tid_x() as uint;

    let x = offset(x, id);
    let y = offset(y, id);
    let z = offset(z, id) as *mut float;
    
    *z = *x + *y
}
