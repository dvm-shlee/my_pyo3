use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use numpy::{IntoPyArray, PyArrayDyn, PyReadonlyArrayDyn};
use numpy::{PyArray2, PyReadonlyArray2};
use ndarray_linalg::solve::Inverse;
use crate::rsfunc::{axpy, mult};

// this function takes input as read-only reference, calculate then output new python array
#[pyfunction]
#[pyo3(name = "axpy")] // this indicates what namespace that this function will take when it binds to the module
fn axpy_py<'py>(
    py: Python<'py>,
    a: f64,
    x: PyReadonlyArrayDyn<f64>,
    y: PyReadonlyArrayDyn<f64>,
) -> &'py PyArrayDyn<f64> {
    let x = x.as_array();
    let y = y.as_array();
    let z = axpy(a, x, y);

    // convert to python array
    z.into_pyarray(py)
}

// this function takes input of python array then convert to mutable reference
// resulted in change the original value
#[pyfunction]
#[pyo3(name = "mult")]
    fn mult_py(_py: Python<'_>, a: f64, x: &PyArrayDyn<f64>) {
        let x = unsafe { x.as_array_mut() };
        mult(a, x);
    }

// this function use rust crate directly instead of using custom function
#[pyfunction] // without separate rust code
fn inv<'py>(py: Python<'py>, x: PyReadonlyArray2<'py, f64>) -> PyResult<&'py PyArray2<f64>> {
    let x = x.as_array();
    let y = x
        .inv()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok(y.into_pyarray(py))
}

#[pymodule]
fn my_rs(py: Python, m: &PyModule) -> PyResult<()> {
    // Add the axpy_py function to the Python module
    m.add_function(wrap_pyfunction!(axpy_py, py)?)?;
    m.add_function(wrap_pyfunction!(mult_py, py)?)?;
    m.add_function(wrap_pyfunction!(inv, py)?)?;

    Ok(())
}
