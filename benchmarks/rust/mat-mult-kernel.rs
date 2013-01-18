mod gpu;
use core::*;

#[abi = "rust-intrinsic"]
extern mod rusti {
    fn addr_of<T>(&&val: T) -> *T;
    fn reinterpret_cast<T, U>(&&e: T) -> U;
}

#[device]
fn offset(x: &float, i: i32) -> &float unsafe {
    let x: uint = rusti::reinterpret_cast(x);
    rusti::reinterpret_cast((x as i32 + i * 8) as uint)
}

#[kernel]
fn mat_mult(a: &float, b: &float, c: &float, n: uint) unsafe {
  let x = gpu::thread_id_x();
  let y = gpu::thread_id_y();
  let s: i32 = n as i32;

  if(x < s && y < s){
    let mut v: float = 0f;
    for i32::range(0, s) |i| { 
	let a = offset(a, (y * s + i));
	let b = offset(b, (i * s + x));
	
	v = v + (*a * *b);
      }

    let c: &mut float = rusti::reinterpret_cast(offset(c, (y * s + x)));
    
    *c = v;
  }
}
