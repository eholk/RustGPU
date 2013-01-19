
extern mod std;
use std::time::precise_time_s;

#[path="../SciRust/src/matrix/matrix.rs"]
mod matrix;

use matrix::{Matrix, TransposeMatrix, Create};
use matrix::generate::{identity, rand_L1};
use matrix::algorithms::{dot, mat_mul, transpose, cholesky_seq_inplace,
                        inverse, cholesky_blocked, par, mat_mul_blocked};
use matrix::util::to_str;

#[nolink]
#[link_args = "-lOpenCL cholesky_cl.o"]
extern mod ocl {
    fn cholesky_init();
    fn cholesky(N: uint, data: *float);
}

fn macros() {
    include!("../examples/gpu_macros.rs")
}

type M = Matrix<float>;

fn main() {
    let N = core::from_str::FromStr::from_str::<uint>(os::args()[1]).get();

    ocl::cholesky_init();

    info!("Generating Matrix");
    let A = gen_matrix(N);
    info!("Matrix generated");

    do vec::as_imm_buf(A.data) |p, _| {
        ocl::cholesky(N, p);
    };
}

fn gen_matrix(N: uint) -> M {
    let L = rand_L1(N);
    let Lt = TransposeMatrix(&L);
    mat_mul(&L, &Lt)
}
