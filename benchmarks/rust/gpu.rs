#[abi = "rust-intrinsic"]
extern mod gpui {
    fn ptx_tid_x() -> i32;
    fn ptx_ntid_x() -> i32;
    fn ptx_ctaid_x() -> i32;
}

#[abi = "rust-intrinsic"]
extern mod rusti {
    fn addr_of<T>(&&val: T) -> *T;
    fn reinterpret_cast<T, U>(&&e: T) -> U;
}

#[device]
#[inline(always)]
pub fn thread_id_x() -> i32 {
    (gpui::ptx_ctaid_x() * gpui::ptx_ntid_x() + gpui::ptx_tid_x())
}

// This function should return void, but instead we do polymorphism to
// make sure it's monomorphized. Otherwise, the PTX backend can't
// generate code that receives a function pointer as an argument.
#[device]
#[inline(always)]
pub fn range<T>(start: i32, stop: i32, f: fn&(i: i32) -> T) -> Option<T> {
    let range = stop - start;

    let i = thread_id_x();
    
    if i < range {
        Some(f(i + start))
    }
    else {
        None
    }
}

#[device]
#[inline(always)]
pub fn reduce_into<T: Copy>(dst: &mut T,
                            init: T,
                            data: &[const T],
                            f: fn&(T, T) -> T)
{
    if thread_id_x() == 0 {
        let len = unsafe {
            let v : *(*const T,uint) =
                rusti::reinterpret_cast(rusti::addr_of(data));
            let (_buf,len) = *v;
            len
        };

        let mut t = init;
        for uint::range(0, len) |i| {
            t = f(copy t, data[i]);
        }
        *dst = t;
    }
}
