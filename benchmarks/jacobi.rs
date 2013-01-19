
use core::from_str::FromStr;

extern mod std;
use std::time::precise_time_s;

extern mod OpenCL;
use OpenCL::hl::*;
use OpenCL::CL::*;
use OpenCL::vector::{Vector, Unique};

#[path="../SciRust/src/matrix/matrix.rs"]
mod matrix;

use matrix::{Matrix, TransposeMatrix, Create};
use matrix::generate::{identity, rand_L1, zero_matrix};
use matrix::algorithms::{dot, mat_mul, transpose, cholesky_seq_inplace,
                        inverse, cholesky_blocked, par, mat_mul_blocked};
use matrix::util::to_str;

fn macros() {
    include!("../examples/gpu_macros.rs")
}

type M = Matrix<float>;

fn main() {
    let N = FromStr::from_str::<uint>(os::args()[1]).get();

    info!("Generating Matrix");
    let A = gen_matrix(N);
    let B: M = zero_matrix::<float, Matrix<float>>(N, N);
    info!("Matrix generated");

    let ctx = create_compute_context_types([GPU]);
    let program = ctx.create_program_from_binary(
        include_str!("jacobi-kernel.ptx"));
    program.build(ctx.device);

    let kernel = program.create_kernel("jacobi");

    benchmark(&A, &B, N, kernel, ctx);
}

fn benchmark(A: &M,
             B: &M,
             N: uint,
             kernel: Kernel,
             ctx: @ComputeContext)
{
    benchmark_one(A, B, N, &kernel, ctx)
}

fn benchmark_one(A: &M,
                 B: &M,
                 N: uint,
                 jacobi: &Kernel,
                 ctx: @ComputeContext)
{
    let start = precise_time_s();

    // Send the data to the GPU
    info!("Matrix buffer length is %?", A.data.len());
    let A = Vector::from_vec(ctx, A.data);
    let B = Vector::from_vec(ctx, B.data);

    let copied = precise_time_s();

    const block_size2: uint = 32;
    let num_threads2 =
        ((N + block_size2 - 1) / block_size2) * block_size2;

    const num_steps: uint = 500;

    for uint::range(0, num_steps) |k| {
        info!("begining iteration %?", k);

        execute!(jacobi[(num_threads2, num_threads2),
                        (block_size2, block_size2)],
                 &N, &A, &B);
        execute!(jacobi[(num_threads2, num_threads2),
                        (block_size2, block_size2)],
                 &N, &B, &A);
    }

    let computed = precise_time_s();

    A.to_vec();

    let stop = precise_time_s();

    let execution = computed - copied;
    let transfer = stop - start - execution;

    io::println(fmt!("%?\t%f\t%f",
                     N,
                     execution,
                     transfer));
}

fn gen_matrix(N: uint) -> M {
    rand_L1(N)
}
