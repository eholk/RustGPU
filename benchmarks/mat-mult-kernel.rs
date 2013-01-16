extern mod core;
use core::*;

#[abi = "rust-intrinsic"]
extern mod gpu {
    fn ptx_tid_x() -> i32;
    fn ptx_ctaid_x() -> i32;
    fn ptx_ntid_x() -> i32;
    fn ptx_tid_y() -> i32;
    fn ptx_ctaid_y() -> i32;
    fn ptx_ntid_y() -> i32;
}

#[abi = "rust-intrinsic"]
extern mod rusti {
    fn addr_of<T>(&&val: T) -> *T;
    fn reinterpret_cast<T, U>(&&e: T) -> U;
}

#[kernel]
fn mat_mult(a: &float, b: &float, c: &float, n: uint) unsafe {
    let x = ((gpu::ptx_ctaid_x() as uint) * (gpu::ptx_ntid_x() as uint)) + (gpu::ptx_tid_x() as uint);
    let y = ((gpu::ptx_ctaid_y() as uint) * (gpu::ptx_ntid_y() as uint)) + (gpu::ptx_tid_y() as uint);
    //let x = gpu::ptx_tid_x() as uint;
    //let y = gpu::ptx_tid_y() as uint;
    
    if(x < 4 && y < 4){
        let mut v: float = 0f;
        for uint::range(0u, 3) |i| { 
	        let a: &float = {
	            let a: uint = rusti::reinterpret_cast(a);
	            rusti::reinterpret_cast(a + ((y * n + i) * 8))
	        };
	        
	        let b: &float = {
	            let b: uint = rusti::reinterpret_cast(b);
	            rusti::reinterpret_cast(b + ((i * n + x) * 8))
	        };
	        
	        v = v + (*a * *b);
        }
        
        let c: & mut float = {
            let c: uint = rusti::reinterpret_cast(c);
            rusti::reinterpret_cast(c + ((y * n + x) * 8))
        };
        
        *c = v;
    }
}
