mod gpu;

#[abi = "rust-intrinsic"]
extern mod rusti {
    fn addr_of<T>(&&val: T) -> *T;
    fn reinterpret_cast<T, U>(&&e: T) -> U;
}

#[device]
fn offset(x: &float, i: i32) -> &float unsafe {
    let x: uint = rusti::reinterpret_cast(x);
    rusti::reinterpret_cast((x as i32 + i * 8) as uint)
}

#[kernel]
#[no_mangle]
fn add_vector(x: &float, y: &float, z: &float, n: uint) unsafe {
    let id = gpu::thread_id_x();

    let x = offset(x, id);
    let y = offset(y, id);
    let z: &mut float = rusti::reinterpret_cast(offset(z, id));

    if id < (n as i32) {
	*z = *x + *y
    }
}
