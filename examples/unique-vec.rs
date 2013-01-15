extern mod OpenCL;
use OpenCL::hl::*;
use OpenCL::CL::*;
use OpenCL::vector::{Vector, Unique};

fn macros() {
    include!("gpu_macros.rs")
}

fn main() {
    let ctx = create_compute_context_types([GPU]);

    let A = ~[1f, 2f, 3f, 4f];
    let B = ~[2f, 4f, 8f, 16f];
    let C = ~[mut 0f, 0f, 0f, 0f];

    let Ab = Unique::from_vec(ctx, copy A);
    let Bb = Unique::from_vec(ctx, copy B);
    let Cb = Unique::from_vec(ctx, C);
    
    let program = ctx.create_program_from_binary(
        include_str!("unique-vec-kernel.ptx"));
    program.build(ctx.device);

    let kernel = program.create_kernel("add_vectors");

    execute!(kernel[A.len(), 1], &A.len(), &Ab, &Bb, &Cb);

    let C = Cb.to_vec();

    io::println(fmt!("%? + %? = %?", A, B, C));
}
