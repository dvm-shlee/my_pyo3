use super::*; // this allows use modules or functions defined at lib.rs
use numpy::ndarray::{IxDyn, Array};

#[test]
fn test_mult() {
    let shape = IxDyn(&[2, 2, 2]);

    let a: f64 = 2.0;
    let mut x = Array::<f64, IxDyn>::from_elem(shape.to_owned(), 2.0);
    rsfunc::mult(a, x.view_mut());
    assert_eq!(x, Array::<f64, IxDyn>::from_elem(shape, 4.0));
}

#[test]
fn test_axpy() {
    let shape = IxDyn(&[2, 2, 2]);

    let a: f64 = 2.0;
    let x = Array::<f64, IxDyn>::from_elem(shape.to_owned(), 1.0);
    let y = Array::<f64, IxDyn>::from_elem(shape, 1.0);
    let z = rsfunc::axpy(a, x.view(), y.view());
    assert_eq!(z, a * x + y);
}