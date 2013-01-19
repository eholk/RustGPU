
#[path="../examples/gpu.rs"]
mod gpu;

#[abi = "rust-intrinsic"]
extern mod rusti {
    fn addr_of<T>(&&val: T) -> *T;
    fn reinterpret_cast<T, U>(&&e: T) -> U;
}

fn recombobulate<T>(ptr: &a/T, len: uint) -> &a/[T] {
    rusti::reinterpret_cast((ptr, len))
}

fn recombobulate_mut<T>(ptr: &a/T, len: uint) -> &a/[mut T] {
    rusti::reinterpret_cast((ptr, len))
}

struct SquareMatrix {
    data: &[mut float],
    N: uint
}

impl SquareMatrix {
    fn get(i: uint, j: uint) -> float {
        self.data[i * self.N + j]
    }

    fn set(i: uint, j: uint, x: float) {
        self.data[i * self.N + j] = x
    }
}

fn recombobulate_matrix(data: &a/float, N: uint) -> SquareMatrix/&a {
    SquareMatrix {
        data: recombobulate_mut(data, N),
        N: N
    }
}

#[device]
fn offset(x: &float, i: uint) -> &mut float unsafe {
    let x: uint = rusti::reinterpret_cast(x);
    rusti::reinterpret_cast((x + i * 8))
}

#[kernel]
#[no_mangle]
fn jacobi(N: uint, src: &float, dst: &float) {
    let (i, j) = gpu::thread_id2();
    let (i, j) = (i + 1, j + 1);

    if i >= N-1 || j >= N-1 { return }

    let u = *offset(src, (i - 1) * N + j);
    let d = *offset(src, (i + 1) * N + j);
    let l = *offset(src, (i) * N + (j - 1));
    let r = *offset(src, (i) * N + (j + 1));

    *offset(dst, i * N + j) = (u + d + l + r) / 4f;
}
