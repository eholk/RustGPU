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
  let kernel_name = "_ZN8mat_mult16_fd4353fb2a31bbf3_00E";

  //let start = get_time();
  let size: int = option::unwrap(int::from_str(os::args()[1]));
  
  let ctx = create_compute_context_types([GPU]);
  
  let A = DVec();
  let B = DVec();
  let C = DVec();
    
  //let size:int = 3;
    
  let mut i = 0;
  let r = rand::Rng();
  while i < (size * size) {
    A.push(r.gen_float());
    B.push(r.gen_float());
    C.push(0f);	
    i = i + 1;  
  }

  //A.set(~[1f,0f,0f,0f,1f,0f,0f,0f,1f]);
  //B.set(~[1f,0f,0f,0f,1f,0f,0f,0f,1f]);
  //C.set(~[0f,0f,0f,0f,0f,0f,0f,0f,0f]);

  let mut Ah = dvec::unwrap(move A);
  let mut Bh = dvec::unwrap(move B);
  let mut Ch = dvec::unwrap(move C);
  
  let program = ctx.create_program_from_binary(
      include_str!("mat-mult-kernel.ptx"));
  
  program.build(ctx.device);

  let kernel = program.create_kernel(kernel_name);
  
  benchmark(&kernel, ctx, size, Ah, Bh, Ch);
  benchmark(&kernel, ctx, size, Ah, Bh, Ch);
  benchmark(&kernel, ctx, size, Ah, Bh, Ch);
  benchmark(&kernel, ctx, size, Ah, Bh, Ch);
  benchmark(&kernel, ctx, size, Ah, Bh, Ch);
}

fn benchmark(kernel: &Kernel, ctx: @ComputeContext,
	     size: int,
	     Ah: &[float], Bh: &[float], Ch: &[float]){
    

    let start = get_time();

    let Ab = Vector::from_vec(ctx, Ah);
    let Bb = Vector::from_vec(ctx, Bh);
    let Cb = Vector::from_vec(ctx, Ch);

    let exec_start = get_time();

    execute!(kernel[(size,size), (16,16)], &Ab, &Bb, &Cb, &size);

    let exec_end = get_time();

    let C = Cb.to_vec();
    
    let end = get_time();
    
    
    let setup_time = exec_start.sec * 1000000000 + exec_start.nsec as i64 - (start.sec * 1000000000 + start.nsec as i64);
    let exec_time = exec_end.sec * 1000000000 + exec_end.nsec as i64 - (exec_start.sec * 1000000000 + exec_start.nsec as i64);
    let copy_to_host = end.sec * 1000000000 + end.nsec as i64 - (exec_end.sec * 1000000000 + exec_end.nsec as i64);

    io::println(fmt!("%?,%?,%?,%?", size, setup_time, exec_time, copy_to_host));
}
