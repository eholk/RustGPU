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
    let kernel_name = "_ZN10add_vector17_a651b999936a2fc63_00E";

    //let start = get_time();
    let size: int = option::unwrap(int::from_str(os::args()[1]));

    let ctx = create_compute_context_types([GPU]);
    
    let A = DVec();
    let B = DVec();
    let C = DVec();
    let n = DVec();

    let mut i = 0;

    let r = rand::Rng();
    while i < size {
    	  A.push(r.gen_float());
	  B.push(r.gen_float());
	  C.push(0f);	
	  i = i + 1;  
    }

    n.push(size as uint);

    let start = get_time();

    let mut Ah = dvec::unwrap(move A);
    let mut Bh = dvec::unwrap(move B);
    let Ab = Vector::from_vec(ctx, Ah);
    let Bb = Vector::from_vec(ctx, Bh);
    let Cb = Vector::from_vec(ctx, dvec::unwrap(move C));
    let nb = Vector::from_vec(ctx, dvec::unwrap(move n));

    let program = ctx.create_program_from_binary(
        include_str!("add-vector-kernel.ptx"));

    program.build(ctx.device);

    let kernel = program.create_kernel(kernel_name);


    kernel.set_arg(0, &0);
    kernel.set_arg(1, &0);
    kernel.set_arg(2, &Ab);
    kernel.set_arg(3, &Bb);
    kernel.set_arg(4, &Cb);    
    kernel.set_arg(5, &nb);

    info!("Starting vector add kernel");

    let exec_start = get_time();

    let local_size: int = 64;
    let global_size: int = ((size + local_size - 1) / local_size) * local_size;
    kernel.execute(global_size, local_size);
    
    let exec_end = get_time();

    info!("Vector add kernel complete");

    let C = Cb.to_vec();
    
    let end = get_time();

    let setup_time = exec_start.sec * 1000000000 + exec_start.nsec as i64 - (start.sec * 1000000000 + start.nsec as i64);
    let exec_time = exec_end.sec * 1000000000 + exec_end.nsec as i64 - (exec_start.sec * 1000000000 + exec_start.nsec as i64);
    let copy_to_host = end.sec * 1000000000 + end.nsec as i64 - (exec_end.sec * 1000000000 + exec_end.nsec as i64);
    
    //io::println(fmt!("Size: %? Elapsed time: %? nanoseconds", size, elapsed));
    //io::println(fmt!("%? + %? = %?", Ah[size -1], Bh[size - 1],  C[size - 1]));

    io::println(fmt!("%?,%?,%?,%?", size, setup_time, exec_time, copy_to_host));
}
