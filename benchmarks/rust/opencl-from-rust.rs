extern mod OpenCL;
extern mod std;
extern mod core;
use OpenCL::CL::*;
use OpenCL::hl::*;
use OpenCL::vector::Vector;
use std::time::*;
use core::*;
use dvec::DVec;

fn macros() {
    include!("gpu_macros.rs")
}

fn main() unsafe {

    let ker = 
        ~"#pragma OPENCL EXTENSION cl_khr_fp64 : enable 
          __kernel void vector_add(__global const float *A, __global const float *B, __global float *C, const int n) {
             int i = get_global_id(0);
             if (i < n){
                C[i] = A[i] + B[i];
             }
         }";

    let size: int = option::unwrap(int::from_str(os::args()[1]));

    let A = DVec();
    let B = DVec();
    let C = DVec();
    let n = DVec();

    let r = rand::Rng();

    for int::range(0, size) |i| {
        A.push(r.gen_float());
        B.push(r.gen_float());
        C.push(0f);
    }

    n.push(size as uint);

    let start = get_time();
    
    let ctx = create_compute_context_types([GPU]);

    let mut Ah = dvec::unwrap(move A);
    let mut Bh = dvec::unwrap(move B);
    let mut Ch = dvec::unwrap(move C);
    let mut Nh = dvec::unwrap(move n);

    let Ab = Vector::from_vec(ctx, Ah);
    let Bb = Vector::from_vec(ctx, Bh);
    let Cb = Vector::from_vec(ctx, Ch);
    let Nb = Vector::from_vec(ctx, Nh);

    let program = ctx.create_program_from_source(ker);
    
    program.build(ctx.device);

    let kernel = program.create_kernel("vector_add");

    let exec_start  = get_time();

    let n: i32 = size as i32;

    kernel.set_arg(0, &Ab);
    kernel.set_arg(1, &Bb);
    kernel.set_arg(2, &Cb);
    kernel.set_arg(3, &n);

    //execute!(kernel[size, 64], &Ab, &Bb, &Cb, &Nb);
    kernel.execute(size, 64);
    
    let exec_end = get_time();

    let C = Cb.to_vec();

    let end = get_time();

    //io::println(fmt!("%? + %? = %?", Ah, Bh, C));
    
    let setup_time = exec_start.sec * 1000000000 + exec_start.nsec as i64 - (start.sec * 1000000000 + start.nsec as i64);
    let exec_time = exec_end.sec * 1000000000 + exec_end.nsec as i64 - (exec_start.sec * 1000000000 + exec_start.nsec as i64);
    let copy_to_host = end.sec * 1000000000 + end.nsec as i64 - (exec_end.sec * 1000000000 + exec_end.nsec as i64);

    io::println(fmt!("%?,%?,%?,%?", size, setup_time, exec_time, copy_to_host));
}
