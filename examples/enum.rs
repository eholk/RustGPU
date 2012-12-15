extern mod OpenCL;
use OpenCL::hl::*;
use OpenCL::CL::*;

mod common;

use common::*;

fn main() {
    let kernel_name = "_ZN8even_odd17_22ab609848a5fcaa3_00E";

    let ctx = create_compute_context();
    let context = &ctx.ctx;
    let q = &ctx.q;

    let A = ~[Float(33f)];
    let B = ~[Int(18)];
    let C = ~[mut Unknown];

    let Ab = create_buffer(context,
                           sys::size_of::<IntOrFloat>() as int,
                           CL_MEM_READ_ONLY);
    let Bb = create_buffer(context,
                           sys::size_of::<IntOrFloat>() as int,
                   
        CL_MEM_READ_ONLY);
    let Cb = create_buffer(context,
                           sys::size_of::<EvenOdd>() as int,
                           CL_MEM_READ_ONLY);
    
    enqueue_write_buffer(q, &Ab, &A);
    enqueue_write_buffer(q, &Bb, &B);

    let program = create_program_with_binary(
        context,
        ctx.device,
        &path::Path("enum-kernel.ptx"));

    build_program(&program, ctx.device);

    let kernel = create_kernel(&program, kernel_name);

    {
        kernel.set_arg(0, &ptr::null::<libc::c_void>());
        kernel.set_arg(1, &ptr::null::<libc::c_void>());
        kernel.set_arg(2, &Ab);
        kernel.set_arg(3, &Cb);    
        
        enqueue_nd_range_kernel(q, &kernel, 1, 0, 8, 8);
        
        enqueue_read_buffer(q, &Cb, &C);
        
        io::println(fmt!("Result: %? => %?", A, C));
    }

    {
        kernel.set_arg(0, &ptr::null::<libc::c_void>());
        kernel.set_arg(1, &ptr::null::<libc::c_void>());
        kernel.set_arg(2, &Bb);
        kernel.set_arg(3, &Cb);    
        
        enqueue_nd_range_kernel(q, &kernel, 1, 0, 8, 8);
        
        enqueue_read_buffer(q, &Cb, &C);
        
        io::println(fmt!("Result: %? => %?", B, C));
    }
}
