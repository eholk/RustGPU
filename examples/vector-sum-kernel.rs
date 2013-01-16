mod gpu;

#[kernel]
#[no_mangle]
fn vector_sum(++src: ~[float],
                dst: &mut float)
{
    gpu::reduce_into(dst, 0f, src,
                     |a, b| a + b);
}
