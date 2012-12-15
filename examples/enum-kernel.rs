mod common;

use common::*;

#[kernel]
fn even_odd(x: &IntOrFloat, y: &mut EvenOdd) {
    match *x {
        Int(i) => *y = if i % 2 == 0 { Even } else { Odd },
        Float(f) => *y = if (f as int) % 2 == 0 { Even } else { Odd },
    }
}
