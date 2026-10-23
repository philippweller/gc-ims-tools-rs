//! PyO3 bindings for the gc-ims-tools Rust kernel.
//!
//! All heavy numerics (PLS NIPALS fits, VIP, selectivity ratio, AsLS
//! baselines, scaling, normalization) run in Rust; the Python package
//! keeps its public API and plotting.

use numpy::{AllowTypeChange, PyArray1, PyArray2, PyArrayLike1, PyArrayLike2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

mod pls;
mod preproc;

fn arr2_to_py<'py>(
    py: Python<'py>,
    a: &ndarray::Array2<f64>,
) -> Bound<'py, PyArray2<f64>> {
    PyArray2::from_owned_array_bound(py, a.clone())
}

fn arr1_to_py<'py>(
    py: Python<'py>,
    a: &ndarray::Array1<f64>,
) -> Bound<'py, PyArray1<f64>> {
    PyArray1::from_owned_array_bound(py, a.clone())
}

/// Fitted PLS model (Rust kernel). Mirrors the sklearn attributes the
/// gc-ims-tools classes consume after `fit`.
#[pyclass(name = "PlsFit", module = "gc_ims_rs")]
#[derive(Clone)]
pub struct PyPlsFit {
    inner: pls::PlsFit,
}

#[pymethods]
impl PyPlsFit {
    #[getter]
    fn x_weights<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        arr2_to_py(py, &self.inner.x_weights)
    }
    #[getter]
    fn y_weights<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        arr2_to_py(py, &self.inner.y_weights)
    }
    #[getter]
    fn x_loadings<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        arr2_to_py(py, &self.inner.x_loadings)
    }
    #[getter]
    fn y_loadings<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        arr2_to_py(py, &self.inner.y_loadings)
    }
    #[getter]
    fn x_rotations<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        arr2_to_py(py, &self.inner.x_rotations)
    }
    #[getter]
    fn y_rotations<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        arr2_to_py(py, &self.inner.y_rotations)
    }
    /// (n_features, n_targets) -- layout of sklearn 1.x `coef_`.
    #[getter]
    fn coef<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        arr2_to_py(py, &self.inner.coef)
    }
    #[getter]
    fn intercept<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        arr1_to_py(py, &self.inner.intercept)
    }
    #[getter]
    fn x_mean<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        arr1_to_py(py, &self.inner.x_mean)
    }
    #[getter]
    fn y_mean<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        arr1_to_py(py, &self.inner.y_mean)
    }

    /// transform(X) == (X - x_mean) @ x_rotations.
    fn transform<'py>(
        &self,
        py: Python<'py>,
        x: PyArrayLike2<'py, f64, AllowTypeChange>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let m = pls::pls_transform(&x.as_array(), &self.inner);
        Ok(arr2_to_py(py, &m))
    }

    /// predict(X) == (X - x_mean) @ coef + y_mean.
    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyArrayLike2<'py, f64, AllowTypeChange>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let m = pls::pls_predict(&x.as_array(), &self.inner);
        Ok(arr2_to_py(py, &m))
    }

    /// y scores == (Y - y_mean) @ y_rotations (mirrors sklearn
    /// `transform(X, Y)` returning the second element).
    fn transform_y<'py>(
        &self,
        py: Python<'py>,
        y: PyArrayLike2<'py, f64, AllowTypeChange>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let yc = &y.as_array() - &self.inner.y_mean;
        let m = pls::pls_reconstruct(&yc, &self.inner.y_rotations);
        Ok(arr2_to_py(py, &m))
    }
}

/// Fit a PLS regression (sklearn PLSRegression-compatible, scale=False,
/// NIPALS) and return the fitted model.
///
/// X: (n_samples, n_features), Y: (n_samples, n_targets), typically the
/// one-hot encoded labels for PLS-DA.
#[pyfunction]
#[pyo3(signature = (x, y, n_components, max_iter=500, tol=1e-6))]
pub fn pls_fit(
    py: Python<'_>,
    x: PyArrayLike2<'_, f64, AllowTypeChange>,
    y: PyArrayLike2<'_, f64, AllowTypeChange>,
    n_components: usize,
    max_iter: usize,
    tol: f64,
) -> PyResult<PyPlsFit> {
    let fit = pls::pls_fit(&x.as_array(), &y.as_array(), n_components, max_iter, tol)
        .map_err(PyValueError::new_err)?;
    let _ = py; // GIL token retained for signature clarity
    Ok(PyPlsFit { inner: fit })
}

/// VIP scores: gc-ims-tools `utils.vip_scores(W, T, Q)`.
/// W: x_weights (p, A); T: x scores (n, A) as returned by transform;
/// Q: y loadings (q, A).
#[pyfunction]
pub fn vip_scores<'py>(
    py: Python<'py>,
    w: PyArrayLike2<'py, f64, AllowTypeChange>,
    t: PyArrayLike2<'py, f64, AllowTypeChange>,
    q: PyArrayLike2<'py, f64, AllowTypeChange>,
) -> Bound<'py, PyArray1<f64>> {
    arr1_to_py(py, &pls::vip_scores(&w.as_array(), &t.as_array(), &q.as_array()))
}

/// Selectivity ratio: gc-ims-tools `utils.selectivity_ratio(X, B)` with
/// B = coefficients^T.
#[pyfunction]
pub fn selectivity_ratio<'py>(
    py: Python<'py>,
    x: PyArrayLike2<'py, f64, AllowTypeChange>,
    coef: PyArrayLike2<'py, f64, AllowTypeChange>,
) -> Bound<'py, PyArray1<f64>> {
    arr1_to_py(py, &pls::selectivity_ratio(&x.as_array(), &coef.as_array()))
}

/// Asymmetric least squares baseline (ims.utils.asymcorr), O(n) banded
/// solve per iteration.
#[pyfunction]
#[pyo3(signature = (y, lam=1e7, p=1e-3, niter=20))]
pub fn asymcorr<'py>(
    py: Python<'py>,
    y: PyArrayLike1<'py, f64, AllowTypeChange>,
    lam: f64,
    p: f64,
    niter: usize,
) -> Bound<'py, PyArray1<f64>> {
    arr1_to_py(py, &preproc::asymcorr(&y.as_array(), lam, p, niter))
}

/// Feature scaling (Dataset.scaling): method in {"auto", "pareto", "var",
/// "none"} with optional mean centering, ddof=0, non-finite weights -> 0.
#[pyfunction]
#[pyo3(signature = (x, method="pareto", mean_centering=true))]
pub fn scale<'py>(
    py: Python<'py>,
    x: PyArrayLike2<'py, f64, AllowTypeChange>,
    method: &str,
    mean_centering: bool,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let out = preproc::scale(&x.as_array(), method, mean_centering).map_err(PyValueError::new_err)?;
    Ok(arr2_to_py(py, &out))
}

/// Min/max normalization per row (one flattened spectrum per row).
#[pyfunction]
pub fn normalize_rows<'py>(
    py: Python<'py>,
    x: PyArrayLike2<'py, f64, AllowTypeChange>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let out = preproc::normalize_rows(&x.as_array()).map_err(PyValueError::new_err)?;
    Ok(arr2_to_py(py, &out))
}

#[pymodule]
fn gc_ims_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPlsFit>()?;
    m.add_function(wrap_pyfunction!(pls_fit, m)?)?;
    m.add_function(wrap_pyfunction!(vip_scores, m)?)?;
    m.add_function(wrap_pyfunction!(selectivity_ratio, m)?)?;
    m.add_function(wrap_pyfunction!(asymcorr, m)?)?;
    m.add_function(wrap_pyfunction!(scale, m)?)?;
    m.add_function(wrap_pyfunction!(normalize_rows, m)?)?;
    Ok(())
}
