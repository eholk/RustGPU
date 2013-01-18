
extern mod std;

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

fn main() {
    const N: uint = 1200;

    let A = gen_matrix(N);

    let ctx = create_compute_context_types([GPU]);
    let program = ctx.create_program_from_binary(
        include_str!("cholesky-kernel.ptx"));
    program.build(ctx.device);


}

fn gen_matrix(N: uint) -> M {
    let L = rand_L1(N);
    let Lt = TransposeMatrix(&L);
    mat_mul(&L, &Lt)
}
