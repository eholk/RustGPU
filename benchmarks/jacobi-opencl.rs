
extern mod std;
use std::time::precise_time_s;

#[path="../SciRust/src/matrix/matrix.rs"]
mod matrix;

use matrix::{Matrix, TransposeMatrix, Create};
use matrix::generate::{identity, rand_L1, zero_matrix};
use matrix::algorithms::{dot, mat_mul, transpose, cholesky_seq_inplace,
                        inverse, cholesky_blocked, par, mat_mul_blocked};
use matrix::util::to_str;

#[nolink]
#[link_args = "-lOpenCL jacobi_cl.o"]
extern mod ocl {
    fn jacobi_init();
    fn jacobi(N: uint, A: *float, B: *float);
}

fn macros() {
    include!("../examples/gpu_macros.rs")
}

type M = Matrix<float>;

fn main() {
    let N = core::from_str::FromStr::from_str::<uint>(os::args()[1]).get();

    ocl::jacobi_init();

    info!("Generating Matrix");
    let A = gen_matrix(N);
    let B = zero_matrix::<float, M>(N, N);
    info!("Matrix generated");

    do vec::as_imm_buf(A.data) |a, _| {
        do vec::as_imm_buf(B.data) |b, _| {
            ocl::jacobi(N, a, b);
        };
    };
}

fn gen_matrix(N: uint) -> M {
    rand_L1(N)
}
