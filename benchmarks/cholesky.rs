
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
use matrix::generate::{identity, rand_L1};
use matrix::algorithms::{dot, mat_mul, transpose, cholesky_seq_inplace,
                        inverse, cholesky_blocked, par, mat_mul_blocked};
use matrix::util::to_str;

fn macros() {
    include!("../examples/gpu_macros.rs")
}

type M = Matrix<float>;

struct CholeskyKernels {
    update_kk: Kernel,
    update_k: Kernel,
    update_block: Kernel,
}

fn main() {
    let N = FromStr::from_str::<uint>(os::args()[1]).get();

    info!("Generating Matrix");
    let A = gen_matrix(N);
    info!("Matrix generated");

    let ctx = create_compute_context_types([GPU]);
    let program = ctx.create_program_from_binary(
        include_str!("cholesky-kernel.ptx"));
    program.build(ctx.device);

    let kernels = CholeskyKernels {
        update_kk: program.create_kernel("update_kk"),
        update_k: program.create_kernel("update_k"),
        update_block: program.create_kernel("update_block")
    };

    info!("Kernel Stats:");
    macro_rules! kernel_stats (
        ($k:ident) => {
            info!("%s: work group size = %?, local mem size = %?, private mem size = %?",
                  stringify!($k),
                  kernels.$k.work_group_size(),  
                  kernels.$k.local_mem_size(),   
                  kernels.$k.private_mem_size());
        }
    );
    kernel_stats!(update_k);
    kernel_stats!(update_kk);
    kernel_stats!(update_block);

    benchmark(&A, N, &kernels, ctx);
}

fn benchmark(A: &M,
             N: uint,
             kernels: &CholeskyKernels,
             ctx: @ComputeContext)
{
    benchmark_one(A, N, kernels, ctx)
}

fn benchmark_one(A: &M,
                 N: uint,
                 kernels: &CholeskyKernels,
                 ctx: @ComputeContext)
{
    let start = precise_time_s();

    // Send the data to the GPU
    info!("Matrix buffer length is %?", A.data.len());
    let A = Vector::from_vec(ctx, A.data);

    let copied = precise_time_s();

    const block_size: uint = 1024;
    let num_threads = ((N + block_size - 1) / block_size) * block_size;

    const block_size2: uint = 32;
    let num_threads2 =
        ((N + block_size2 - 1) / block_size2) * block_size2;

    info!("num_threads = %?, block_size = %?", num_threads, block_size);

    let update_kk = &kernels.update_kk;
    let update_k = &kernels.update_k;
    let update_block = &kernels.update_block;

    for uint::range(0, N) |k| {
        info!("begining iteration %?", k);
        execute!(update_kk[1, 1],
                 &N, &A, &k);
        info!("updated_kk");
        execute!(update_k[num_threads, block_size],
                 &N, &A, &k);
        info!("updated_k");
        execute!(update_block[(num_threads2, num_threads2),
                              (block_size2, block_size2)],
                 &N, &A, &k);
        info!("updated_block");
    }

    let computed = precise_time_s();

    A.to_vec();

    let stop = precise_time_s();

    io::println(fmt!("%?\t%?",
                     computed - copied,
                     stop - start));
}

fn gen_matrix(N: uint) -> M {
    let L = rand_L1(N);
    let Lt = TransposeMatrix(&L);
    mat_mul(&L, &Lt)
}
