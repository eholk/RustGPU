extern mod OpenCL;
use OpenCL::hl::*;
use OpenCL::CL::*;
use OpenCL::vector::Vector;

fn macros() {
    include!("gpu_macros.rs")
}

fn main() {
    let ctx = create_compute_context_types([GPU]);

    let A = ~[1f, 2f, 3f, 4f];
    let B = ~[2f, 4f, 8f, 16f];
    let C = ~[mut 0f, 0f, 0f, 0f];

    let Ab = Vector::from_vec(ctx, A);
    let Bb = Vector::from_vec(ctx, B);
    let Cb = Vector::from_vec(ctx, C);
    
    let program = ctx.create_program_from_binary(
        include_str!("add-vector-kernel.ptx"));
    
    program.build(ctx.device);

    let kernel = program.create_kernel("add_vector");

    execute!(kernel[4, 4], &Ab, &Bb, &Cb);

    let C = Cb.to_vec();

    io::println(fmt!("%? + %? = %?", A, B, C));
}
