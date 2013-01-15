extern mod OpenCL;
use OpenCL::hl::*;
use OpenCL::CL::*;
use OpenCL::vector::Vector;

macro_rules! set_args (
    ($kernel:expr, $i: expr) => {()};

    ($kernel:expr, $i:expr, $arg:expr) => {{
        $kernel.set_arg($i, $arg);
    }};

    ($kernel:expr, $i:expr, $arg:expr $(, $args:expr)*) => {{
        $kernel.set_arg($i, $arg);
        set_args!($kernel, $i + 1 $(, $args)*);
    }};
)

macro_rules! execute (
    ($kernel:ident [$global:expr, $local:expr] ,
     $($args:expr),*) => {{
        $kernel.set_arg(0, &0);
        $kernel.set_arg(1, &0);
        set_args!($kernel, 2, $($args),*);
        $kernel.execute($global, $local)
    }}
)

fn main() {
    let ctx = create_compute_context_types([GPU]);

    let A = ~[1f, 2f, 3f, 4f];
    let B = ~[2f, 4f, 8f, 16f];
    let C = ~[mut 0f, 0f, 0f, 0f];

    let Ab = Vector::from_vec(ctx, A);
    let Bb = Vector::from_vec(ctx, B);
    let Cb = Vector::from_vec(ctx, C);
    
    let program = ctx.create_program_from_binary(
        include_str!("add-vector-kernel.ptx"));
    
    program.build(ctx.device);

    let kernel = program.create_kernel("add_vector");

    execute!(kernel[4, 4], &Ab, &Bb, &Cb);

    let C = Cb.to_vec();

    io::println(fmt!("%? + %? = %?", A, B, C));
}
