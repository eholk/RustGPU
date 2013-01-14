extern mod OpenCL;
use OpenCL::hl::*;
use OpenCL::CL::*;
use OpenCL::vector::Vector;

#[abi = "rust-intrinsic"]
#[cfg(gpu)]
extern mod gpu {
    fn ptx_tid_x() -> i32;
}

#[abi = "rust-intrinsic"]
extern mod rusti {
    fn addr_of<T>(&&val: T) -> *T;
    fn reinterpret_cast<T, U>(&&e: T) -> U;
}

#[device]
#[cfg(gpu)]
fn offset(x: &float, i: uint) -> &float unsafe {
    let x: uint = rusti::reinterpret_cast(x);
    rusti::reinterpret_cast((x + i * 8))
}

#[kernel]
#[no_mangle]
#[cfg(gpu)]
fn add_vector(x: &float, y: &float, z: &float) unsafe {
    let id = gpu::ptx_tid_x() as uint;

    let x = offset(x, id);
    let y = offset(y, id);
    let z: &mut float = rusti::reinterpret_cast(offset(z, id));

    *z = *x + *y
}

#[cfg(no_gpu)]
fn main() {
    let kernel_name = "add_vector";

    let ctx = create_compute_context_types([GPU]);
    let context = &ctx.ctx;
    let q = &ctx.q;

    let A = ~[1f, 2f, 3f, 4f];
    let B = ~[2f, 4f, 8f, 16f];
    let C = ~[mut 0f, 0f, 0f, 0f];

    let Ab = Vector::from_vec(ctx, A);
    let Bb = Vector::from_vec(ctx, B);
    let Cb = Vector::from_vec(ctx, C);
    
    let program = create_program_with_binary(
        context,
        ctx.device,
        &path::Path("add-vector1.ptx"));

    build_program(&program, ctx.device);

    let kernel = create_kernel(&program, kernel_name);

    kernel.set_arg(0, &0);
    kernel.set_arg(1, &0);
    kernel.set_arg(2, &Ab);
    kernel.set_arg(3, &Bb);
    kernel.set_arg(4, &Cb);    

    enqueue_nd_range_kernel(q, &kernel, 1, 0, 4, 4);

    let C = Cb.to_vec();

    io::println(fmt!("%? + %? = %?", A, B, C));
}
