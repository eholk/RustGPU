#[kernel]
fn add_float(x: &float, y: &float, z: &mut float) {
    *z = *x + *y;
}
