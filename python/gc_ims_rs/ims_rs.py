"""Drop-in Rust backend for gc-ims-tools.

`import gc_ims_rs.ims_rs` (auto-installs) rebinds the PLS engine inside the
*installed* `ims` package (plsda.PLS_DA, plsr.PLSR) so that every PLS
fit / predict / transform runs in the Rust kernel instead of scikit-learn.
All class methods, attributes, plotting and the public API are unchanged —
only the numerical engine is swapped.

If the compiled extension is missing (shouldn't happen inside this package),
the `ims` classes silently keep using scikit-learn.
"""

import gc_ims_rs as _rs

import numpy as np

__all__ = ["RustPLS", "install", "is_active", "RUST_AVAILABLE"]

try:
    # smoke-test that the compiled functions are reachable through the shim
    _rs.pls_fit
    RUST_AVAILABLE = True
except AttributeError:  # pragma: no cover - extension not linked into shim
    RUST_AVAILABLE = False


def _as2(a):
    a = np.asarray(a)
    return (
        np.ascontiguousarray(a.astype(np.float64))
        if a.dtype != np.float64
        else np.ascontiguousarray(a)
    )


class RustPLS:
    """sklearn-PLSRegression-compatible facade over the Rust PlsFit.

    Exposes exactly the surface the `ims` classes consume: fit / transform(X, Y)
    / predict / x_weights_ / x_loadings_ / y_weights_ / y_loadings_ / coef_
    (n_features, n_targets) / intercept_.
    """

    def __init__(self, n_components=2, **kwargs):
        self.n_components = int(n_components)
        self._fit = None

    def fit(self, X, Y):
        X = _as2(X)
        Y = np.asarray(Y)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        Y = _as2(Y)
        self._fit = _rs.pls_fit(X, Y, self.n_components)
        return self

    def transform(self, X, Y=None, copy=True):
        xs = np.asarray(self._fit.transform(_as2(X)))
        if Y is None:
            return xs
        ys = np.asarray(self._fit.transform_y(_as2(Y)))
        return xs, ys

    def predict(self, X, copy=True):
        return np.asarray(self._fit.predict(_as2(X)))

    @property
    def x_weights_(self):
        return np.asarray(self._fit.x_weights)

    @property
    def x_loadings_(self):
        return np.asarray(self._fit.x_loadings)

    @property
    def y_weights_(self):
        return np.asarray(self._fit.y_weights)

    @property
    def y_loadings_(self):
        return np.asarray(self._fit.y_loadings)

    @property
    def coef_(self):
        return np.asarray(self._fit.coef)

    @property
    def intercept_(self):
        return np.asarray(self._fit.intercept)


def _rust_ctor(n_components=2, **kwargs):
    """Construct the Rust PLS engine (sklearn-kwarg compatible)."""
    return RustPLS(n_components=n_components)


def _rust_vip_scores(W, T, Q):
    return np.asarray(_rs.vip_scores(_as2(W), _as2(T), _as2(Q)))


def _rust_selectivity_ratio(X, B):
    # kernel convention: B = coefficients, (n_features, n_targets). The
    # numpy implementation in ims.utils crashes for q > 1 targets; the
    # per-column Rust version returns the correct q == 1 result and is
    # well-defined for q > 1.
    return np.asarray(_rs.selectivity_ratio(_as2(X), _as2(B)))


_installed = False


def install():
    """Swap the PLS engine (+ VIP) inside the installed `ims` package to Rust.

    Idempotent; safely callable multiple times. No-op if `ims` is not
    installed (sklearn stays the engine).
    """
    global _installed
    if _installed or not RUST_AVAILABLE:
        return
    try:
        import ims.plsda as _plsda
        import ims.plsr as _plsr
        import ims.utils as _utils
    except ImportError:
        return
    # Rebinding the PLSRegression symbol the classes construct internally
    # leaves every method, plot and the public API untouched.
    _plsda.PLSRegression = _rust_ctor
    _plsr.PLSRegression = _rust_ctor
    # VIP is computed on every fit (incl. cross-validation) -> route to Rust.
    _utils.vip_scores = _rust_vip_scores
    # Selectivity ratio: numpy implementation broadcasts only for one target.
    _utils.selectivity_ratio = _rust_selectivity_ratio
    _installed = True


# Drop-in promise: patch on import.
install()


def is_active():
    """True if Rust is installed and the `ims` package routes through it."""
    if not RUST_AVAILABLE:
        return False
    try:
        import ims.plsda as _plsda

        return _plsda.PLSRegression is _rust_ctor
    except ImportError:
        return False