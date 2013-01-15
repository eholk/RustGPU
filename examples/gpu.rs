#[abi = "rust-intrinsic"]
extern mod gpui {
    fn ptx_tid_x() -> i32;
    fn ptx_ntid_x() -> i32;
    fn ptx_ctaid_x() -> i32;
}

#[device]
#[inline(always)]
pub fn thread_id_x() -> uint {
    (gpui::ptx_ctaid_x() * gpui::ptx_ntid_x() + gpui::ptx_tid_x()) as uint
}

// This function should return void, but instead we do polymorphism to
// make sure it's monomorphized. Otherwise, the PTX backend can't
// generate code that receives a function pointer as an argument.
#[device]
#[inline(always)]
pub fn range<T>(start: uint, stop: uint, f: fn&(i: uint) -> T) -> Option<T> {
    let range = stop - start;

    let i = thread_id_x();
    
    if i < range {
        Some(f(i + start))
    }
    else {
        None
    }
}
