
#[path="../examples/gpu.rs"]
mod gpu;

#[abi = "rust-intrinsic"]
extern mod rusti {
    fn addr_of<T>(&&val: T) -> *T;
    fn reinterpret_cast<T, U>(&&e: T) -> U;
    fn sqrtf64(x: f64) -> f64;
}

fn sqrt(x: float) -> float {
    rusti::sqrtf64(x as f64) as float
}

#[device]
fn offset(x: &float, i: uint) -> &mut float unsafe {
    let x: uint = rusti::reinterpret_cast(x);
    rusti::reinterpret_cast((x + i * 8))
}

fn recombobulate<T>(ptr: &a/T, len: uint) -> &a/[T] {
    rusti::reinterpret_cast((ptr, len))
}

fn recombobulate_mut<T>(ptr: &a/T, len: uint) -> &a/[mut T] {
    rusti::reinterpret_cast((ptr, len))
}

struct SquareMatrix {
    data: &float,
    N: uint
}

impl SquareMatrix {
    fn get(i: uint, j: uint) -> float {
        let k = (i * self.N + j);
        if k < self.N * self.N {
            *offset(self.data, k)
        }
        else {
            -1e9f
        }
    }

    fn set(i: uint, j: uint, x: float) {
        let k = (i * self.N + j);
        if k < self.N * self.N {
            *offset(self.data, k) = x
        }
    }
}

fn recombobulate_matrix(data: &a/float, N: uint) -> SquareMatrix/&a {
    SquareMatrix {
        data: data,
        N: N
    }
}

#[kernel]
#[no_mangle]
fn update_kk(N: uint, data: &float, k: uint) {
    let A = recombobulate_matrix(data, N * N);
    let Akk = *offset(data, k * N + k);
    //A.set(k, k, sqrt(A.get(k, k)))
    *offset(data, k * N + k) = sqrt(Akk);
}

#[kernel]
#[no_mangle]
fn update_k(N: uint, data: &float, k: uint) {
    let A = recombobulate_matrix(data, N * N);
    
    let i = gpu::thread_id_x();

    if i > k && i < N {
        //let Akk = A.get(k, k);
        let Akk = *offset(data, k * N + k);
        //A.set(i, k, A.get(i, k) / Akk);
        *offset(data, i * N + k) = *offset(data, i * N + k) / Akk;
        // zero out the top half of the matrix so we can read the results.
        //A.set(k, i, 0f)
        *offset(data, k * N + i) = 0f;
    }
}

#[kernel]
#[no_mangle]
fn update_block(N: uint, data: &float, k: uint) {
    let A = recombobulate_matrix(data, N * N);

    let (i, j) = gpu::thread_id2();

    if i <= k || j <= k { return }
    if i >= N || j > i  { return }

    //let Aik = A.get(i, k);
    //let Ajk = A.get(j, k);
    //let Aij = A.get(i, j);
    //
    //A.set(i, j, Aij - Aik * Ajk);    

    let Aik = *offset(data, i * N + k);
    let Ajk = *offset(data, j * N + k);
    let Aij = *offset(data, i * N + j);
    
    *offset(data, i * N + j) = Aij - Aik * Ajk;
}
