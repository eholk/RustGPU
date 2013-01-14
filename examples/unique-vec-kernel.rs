
#[kernel]
#[no_mangle]
fn add_vectors(A: ~[float], B: ~[float], C: ~[mut float]) {
    //C[1] = A.len() as float;
    //C[2] = B.len() as float;
    C[0] = A[0] + B[0];
    C[1] = A[1] + B[1];
    C[2] = A[2] + B[2];
    C[3] = A[3] + B[3];

    unsafe {
        cast::forget(A);
        cast::forget(B);
        cast::forget(C);
    }
}
