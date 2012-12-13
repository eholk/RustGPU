extern mod OpenCL;
use OpenCL::hl::*;
use OpenCL::CL::*;

fn main() {
    let kernel_name = "_ZN9add_float17_5ac86b11d5cd66c13_00E";

    let platforms = get_platforms();
    let devices = platforms[0].get_devices();
    let context = create_context(devices[0]);
    let q = create_commandqueue(&context, devices[0]);

    let A = ~[1f];
    let B = ~[2f];
    let C = ~[mut 0f];

    let Ab = create_buffer(&context,
                           sys::size_of::<float>() as int,
                           CL_MEM_READ_ONLY);
    let Bb = create_buffer(&context,
                           sys::size_of::<float>() as int,
                   
        CL_MEM_READ_ONLY);
    let Cb = create_buffer(&context,
                           sys::size_of::<float>() as int,
                           CL_MEM_READ_ONLY);
    
    enqueue_write_buffer(&q, &Ab, &A);
    enqueue_write_buffer(&q, &Bb, &B);

    let program = create_program_with_binary(
        &context,
        devices[0],
        &path::Path("add-float-kernel.ptx"));

    build_program(&program, devices[0]);

    let kernel = create_kernel(&program, kernel_name);

    kernel.set_arg(0, &ptr::null::<libc::c_void>());
    kernel.set_arg(1, &ptr::null::<libc::c_void>());
    kernel.set_arg(2, &Ab);
    kernel.set_arg(3, &Bb);
    kernel.set_arg(4, &Cb);    

    enqueue_nd_range_kernel(&q, &kernel, 1, 0, 1 as int, 1);

    enqueue_read_buffer(&q, &Cb, &C);

    io::println(fmt!("%? + %? = %?", A, B, C));
}
