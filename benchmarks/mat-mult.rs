extern mod OpenCL;
extern mod std;
extern mod core;

use OpenCL::hl::*;
use OpenCL::CL::*;
use OpenCL::vector::Vector;
use std::time::*;
use core::*;
use dvec::DVec;

fn main() {
    let kernel_name = "_ZN8mat_mult16_bea2bb16c38860a3_00E";

    //let start = get_time();

    let ctx = create_compute_context_types([GPU]);
    let context = &ctx.ctx;
    let q = &ctx.q;
    
    let A = DVec();
    let B = DVec();
    let C = DVec();
    
    let mut i = 0;
    let r = rand::Rng();
    while i < 16 {
    	  A.push(r.gen_float());
	  B.push(r.gen_float());
	  C.push(0f);	
	  i = i + 1;  
    }

    let mut Ah = dvec::unwrap(move A);
    let Ab = Vector::from_vec(ctx, Ah);
    let Bb = Vector::from_vec(ctx, dvec::unwrap(move B));
    let Cb = Vector::from_vec(ctx, dvec::unwrap(move C));
    
    let program = create_program_with_binary(
        context,
        ctx.device,
        &path::Path("mat-mult-kernel.ptx"));

    build_program(&program, ctx.device);

    let kernel = create_kernel(&program, kernel_name);

    kernel.set_arg(0, &0);
    kernel.set_arg(1, &0);
    kernel.set_arg(2, &Ab);
    kernel.set_arg(3, &Bb);
    kernel.set_arg(4, &Cb);    
    kernel.set_arg(5, &4u);

    let start = get_time();
    enqueue_nd_range_kernel(q, &kernel, 2, 0, 16, 1);

    let C = Cb.to_vec();
    
    let end = get_time();

    let elapsed = end.sec * 1000000000 + end.nsec as i64 - (start.sec * 1000000000 + start.nsec as i64);

    io::println(fmt!("lapsed time: %?", elapsed));
    io::println(fmt!("%? - %?", Ah[3194303],  C[3194303]));
}
