extern mod OpenCL;
use OpenCL::hl::*;
use OpenCL::CL::*;

fn main() {
    let kernel_name = "_ZN10add_vector17_2950a9efd92916123_00E";

    let ctx = create_compute_context();
    let context = &ctx.ctx;
    let q = &ctx.q;

    let A = ~[1f, 2f, 3f, 4f];
    let B = ~[2f, 4f, 8f, 16f];
    let C = ~[mut 0f, 0f, 0f, 0f];

    let Ab = create_buffer(context,
                           (sys::size_of::<float>() * A.len()) as int,
                           CL_MEM_READ_ONLY);                    
    let Bb = create_buffer(context,                             
                           (sys::size_of::<float>() * B.len()) as int,
                           CL_MEM_READ_ONLY);                    
    let Cb = create_buffer(context,                             
                           (sys::size_of::<float>() * C.len()) as int,
                           CL_MEM_READ_ONLY);
    
    enqueue_write_buffer(q, &Ab, &A);
    enqueue_write_buffer(q, &Bb, &B);

    let program = create_program_with_binary(
        context,
        ctx.device,
        &path::Path("add-vector-kernel.ptx"));

    build_program(&program, ctx.device);

    let kernel = create_kernel(&program, kernel_name);

    kernel.set_arg(0, &ptr::null::<libc::c_void>());
    kernel.set_arg(1, &ptr::null::<libc::c_void>());
    kernel.set_arg(2, &Ab);
    kernel.set_arg(3, &Bb);
    kernel.set_arg(4, &Cb);    

    enqueue_nd_range_kernel(q, &kernel, 1, 0, 4, 4);

    enqueue_read_buffer(q, &Cb, &C);

    io::println(fmt!("%? + %? = %?", A, B, C));
}
