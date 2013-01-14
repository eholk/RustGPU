
#[kernel]
#[no_mangle]
fn add_vectors(A: ~[float], B: ~[float], C: ~[mut float]) {
    C[0] = A[0] + B[0];
}
