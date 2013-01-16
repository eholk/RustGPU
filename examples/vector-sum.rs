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
    let sum = ~[mut 0f];

    let Ab = Unique::from_vec(ctx, copy A);
    let sum = Vector::from_vec(ctx, sum);
    
    let program = ctx.create_program_from_binary(
        include_str!("vector-sum-kernel.ptx"));
    program.build(ctx.device);

    let kernel = program.create_kernel("vector_sum");

    execute!(kernel[A.len(), 1], &Ab, &sum);

    let sum = sum.to_vec()[0];

    io::println(fmt!("+/%? = %?", A, sum));
}
