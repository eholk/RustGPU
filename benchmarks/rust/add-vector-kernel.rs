mod gpu;

#[abi = "rust-intrinsic"]
extern mod rusti {
    fn addr_of<T>(&&val: T) -> *T;
    fn reinterpret_cast<T, U>(&&e: T) -> U;
}

#[device]
fn offset(x: &float, i: uint) -> &float unsafe {
    let x: uint = rusti::reinterpret_cast(x);
    rusti::reinterpret_cast((x + i * 8))
}

#[kernel]
#[no_mangle]
fn add_vector(x: &float, y: &float, z: &float) unsafe {
    let id = gpu::thread_id_x();

    let x = offset(x, id);
    let y = offset(y, id);
    let z: &mut float = rusti::reinterpret_cast(offset(z, id));

    *z = *x + *y
}
