#[abi = "rust-intrinsic"]
extern mod gpu {
    fn ptx_tid_x() -> i32;
}

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

//#[kernel]
//fn add_vector(x: &float, y: &float, z: &float) unsafe {
//    let id = gpu::ptx_tid_x() as uint;
//
//    let x = offset(x, id);
//    let y = offset(y, id);
//    let z: &mut float = rusti::reinterpret_cast(offset(z, id));
//
//    *z = *x + *y
//}

macro_rules! make_map(
    { $id:ident($($x:ident : $t:ty),*) => $body:expr } => {
        fn $id($($x: &mut $t),*) {
            $body;
        }
    } 
)

make_map! {
    add_vector(x: float, y: float, z: float) => *z = *x + *y
}
