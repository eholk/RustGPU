mod gpu;

#[kernel]
#[no_mangle]
fn add_vectors(N: uint, A: ~[float], B: ~[float], C: ~[mut float]) {
    do gpu::range(0, N) |i| {
        C[i] = A[i] + B[i]
    };

    unsafe {
        cast::forget(A);
        cast::forget(B);
        cast::forget(C);
    }
}
