extern mod OpenCL;
use OpenCL::hl::*;
use OpenCL::CL::*;
use OpenCL::vector::Vector;

fn main() {
    let kernel_name = "_ZN10add_vector17_2950a9efd92916123_00E";

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
        &path::Path("add-vector-kernel.ptx"));

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
