extern mod OpenCL;
extern mod std;
extern mod core;
use OpenCL::hl::*;
use OpenCL::CL::*;
use OpenCL::vector::Vector;
use std::time::*;
use core::*;
use dvec::DVec;

fn macros() {
    include!("gpu_macros.rs")
}

fn main() {
    let ctx = create_compute_context_types([GPU]);

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

    let program = ctx.create_program_from_binary(
        include_str!("add-vector-kernel.ptx"));
    
    program.build(ctx.device);

    let mut Ah = dvec::unwrap(move A);
    let mut Bh = dvec::unwrap(move B);
    let mut Ch = dvec::unwrap(move C);

    let kernel = program.create_kernel("add_vector");

    benchmark(&kernel, ctx, size, Ah, Bh, Ch);
    benchmark(&kernel, ctx, size, Ah, Bh, Ch);
    benchmark(&kernel, ctx, size, Ah, Bh, Ch);
    benchmark(&kernel, ctx, size, Ah, Bh, Ch);
    benchmark(&kernel, ctx, size, Ah, Bh, Ch);
    benchmark(&kernel, ctx, size, Ah, Bh, Ch);
}

fn benchmark(kernel: &Kernel, ctx: @ComputeContext,
             size: int,
             Ah: &[float], Bh: &[float], Ch: &[float]) {

    let start = get_time();

    let Ab = Vector::from_vec(ctx, Ah);
    let Bb = Vector::from_vec(ctx, Bh);
    let Cb = Vector::from_vec(ctx, Ch);

    let exec_start  = get_time();

    execute!(kernel[size, 64], &Ab, &Bb, &Cb, &size);
    
    let exec_end = get_time();

    let C = Cb.to_vec();

    let end = get_time();

    //io::println(fmt!("%? + %? = %?", Ah, Bh, C));
    
    let setup_time = exec_start.sec * 1000000000 + exec_start.nsec as i64 - (start.sec * 1000000000 + start.nsec as i64);
    let exec_time = exec_end.sec * 1000000000 + exec_end.nsec as i64 - (exec_start.sec * 1000000000 + exec_start.nsec as i64);
    let copy_to_host = end.sec * 1000000000 + end.nsec as i64 - (exec_end.sec * 1000000000 + exec_end.nsec as i64);

    io::println(fmt!("%?,%?,%?,%?", size, setup_time, exec_time, copy_to_host));
}
