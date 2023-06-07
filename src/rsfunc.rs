use numpy::ndarray::{ArrayD, ArrayViewD, ArrayViewMutD};

// rust function rust functions 
pub fn axpy(a: f64, x: ArrayViewD<'_, f64>, y: ArrayViewD<'_, f64>) -> ArrayD<f64> {
    a * &x + &y
}

pub fn mult(a: f64, mut x: ArrayViewMutD<'_, f64>) {
    x *= a;
}