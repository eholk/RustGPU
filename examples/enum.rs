extern mod OpenCL;
use OpenCL::hl::*;
use OpenCL::CL::*;
use OpenCL::vector::Vector;

mod common;

use common::*;

fn main() {
    let kernel_name = "_ZN8even_odd16_4832a616ac265ea3_00E";

    let ctx = create_compute_context_types([GPU]);
    let context = &ctx.ctx;
    let q = &ctx.q;

    let A = ~[Float(33f)];
    let B = ~[Int(18)];
    let C = ~[mut Unknown];

    let Ab = Vector::from_vec(ctx, A);
    let Bb = Vector::from_vec(ctx, B);
    let Cb = Vector::from_vec(ctx, C);
    
    let program = create_program_with_binary(
        context,
        ctx.device,
        &path::Path("enum-kernel.ptx"));

    build_program(&program, ctx.device);

    let kernel = create_kernel(&program, kernel_name);

    {
        kernel.set_arg(0, &0);
        kernel.set_arg(1, &0);
        kernel.set_arg(2, &Ab);
        kernel.set_arg(3, &Cb);    
        
        enqueue_nd_range_kernel(q, &kernel, 1, 0, 8, 8);
        
        let C = Cb.to_vec();
        
        io::println(fmt!("Result: %? => %?", A, C));
    }

    let Cb = Vector::from_vec(ctx, C);

    {
        kernel.set_arg(0, &0);
        kernel.set_arg(1, &0);
        kernel.set_arg(2, &Bb);
        kernel.set_arg(3, &Cb);    
        
        enqueue_nd_range_kernel(q, &kernel, 1, 0, 8, 8);
        
        let C = Cb.to_vec();
        
        io::println(fmt!("Result: %? => %?", B, C));
    }
}
