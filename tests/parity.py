"""Parity check: Rust kernel (gc_ims_rs) vs. the exact Python reference
(scikit-learn 1.1.x PLSRegression + the gc-ims-tools wrapper logic).

Runs:
  - PLS-DA multi-class (iris, wine) and binary (breast cancer)
  - PLS regression (1 and 2 targets, random data)
  - VIP scores & selectivity ratio as computed by ims.utils
  - asymcorr (AsLS) vs. the original sparse implementation
  - scaling (auto/pareto/var/none) vs. the original numpy code
  - normalize_rows vs. Spectrum.normalization logic

Fails hard (SystemExit) on any max-abs difference above tolerance.
"""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from sklearn.cross_decomposition import PLSRegression
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.preprocessing import LabelBinarizer

import gc_ims_rs

TOL = 1e-8


def check(name, a, b, tol=TOL):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    assert a.shape == b.shape, f"{name}: shape {a.shape} != {b.shape}"
    d = np.max(np.abs(a - b)) if a.size else 0.0
    ok = np.allclose(a, b, rtol=1e-6, atol=tol)
    print(f"  {name:34s} max|diff| = {d:.3e}  {'OK' if ok else 'FAIL'}")
    if not ok:
        raise SystemExit(f"PARITY FAILED: {name} (max|diff|={d:.3e})")


# ----------------------------------------------------------------------
def pls_ref(X, Y, n_components):
    """sklearn reference + the exact gc-ims-tools post-processing."""
    pls = PLSRegression(n_components=n_components, scale=False)
    pls.fit(X, Y)
    x_scores, y_scores = pls.transform(X, Y)
    return {
        "x_scores": x_scores,
        "y_scores": y_scores,
        "x_weights": pls.x_weights_,
        "x_loadings": pls.x_loadings_,
        "y_weights": pls.y_weights_,
        "y_loadings": pls.y_loadings_,
        "coef": pls.coef_,
        "intercept": pls.intercept_,
        "pred": pls.predict(X),
    }


def vip_ref(W, T, Q):
    W0 = W / np.sqrt(np.sum(W**2, 0))
    p, _ = W.shape
    sum_sq = np.sum(T**2, 0) * np.sum(Q**2, 0)
    return np.sqrt(p * np.sum(sum_sq * (W0**2), 1) / np.sum(sum_sq))


def sr_ref(X, B):
    """Per-column SR with B in (n_features, n_targets) layout; for q == 1
    identical to the numpy broadcast in ims.utils.selectivity_ratio(
    X, coefficients.T) with sklearn >= 1.3 coef_ = (q, p)."""
    B = np.asarray(B, float)
    t = X @ B / np.linalg.norm(B)
    q = t.shape[1]
    p = X.shape[1]
    ptp = np.zeros((p, q))
    for j in range(q):
        tj = t[:, j]
        ptp[:, j] = X.T @ tj / (tj @ tj)
    exp_var = (t @ ptp.T) ** 2
    exp_res = (X - exp_var) ** 2
    return exp_var.sum(0) / exp_res.sum(0)


def compare_pls(name, X, Y, n_components):
    print(f"[{name}]  X={X.shape} Y={Y.shape} n_components={n_components}")
    ref = pls_ref(X, Y, n_components)
    fit = gc_ims_rs.pls_fit(X, Y, n_components)
    x_scores = fit.transform(X)
    y_scores = fit.transform_y(Y)
    check("x_scores", ref["x_scores"], x_scores)
    check("y_scores", ref["y_scores"], y_scores)
    check("x_weights", ref["x_weights"], fit.x_weights)
    check("x_loadings", ref["x_loadings"], fit.x_loadings)
    check("y_weights", ref["y_weights"], fit.y_weights)
    check("y_loadings", ref["y_loadings"], fit.y_loadings)
    check("coef", ref["coef"], fit.coef)  # both (p, q)
    check("intercept", ref["intercept"], fit.intercept)
    pred = fit.predict(X)
    check("predict", ref["pred"], pred)
    # VIP exactly like ims.utils.vip_scores(x_weights, transform_T, y_loadings)
    vip = gc_ims_rs.vip_scores(fit.x_weights, x_scores, fit.y_loadings)
    check("vip_scores", vip_ref(ref["x_weights"], ref["x_scores"], ref["y_loadings"]), vip)
    # selectivity ratio (PLSR path; kernel convention: coef (p, q))
    sr = gc_ims_rs.selectivity_ratio(X, fit.coef)
    check("selectivity_ratio", sr_ref(X, ref["coef"]), sr)
    # transform on unseen data (same as train here is enough: check API)
    check("transform(X_test=x)", x_scores, fit.transform(X))
    return ref, fit


# ----------------------------------------------------------------------
print("== multi-class PLS-DA ==")
iris = load_iris()
iy = iris.target_names[iris.target]
compare_pls("iris_uv_2", iris.data, LabelBinarizer().fit_transform(iy), 2)
compare_pls("iris_uv_3", iris.data, LabelBinarizer().fit_transform(iy), 3)

wine = load_wine()
wy = wine.target_names[wine.target]
compare_pls("wine_uv_2", wine.data, LabelBinarizer().fit_transform(wy), 2)
compare_pls("wine_uv_5", wine.data, LabelBinarizer().fit_transform(wy), 5)

print("== binary PLS-DA (LabelBinarizer -> single column, q=1) ==")
bc = load_breast_cancer()
by = bc.target_names[bc.target]
compare_pls("bcancer_uv_2", bc.data, LabelBinarizer().fit_transform(by), 2)

print("== PLS regression ==")
rng = np.random.default_rng(42)
Xr = rng.normal(size=(120, 40))
y1 = Xr @ rng.normal(size=40) + 0.1 * rng.normal(size=120)
compare_pls("regr_1target", Xr, y1.reshape(-1, 1), 4)
Y2 = Xr @ rng.normal(size=(40, 2)) + 0.1 * rng.normal(size=(120, 2))
compare_pls("regr_2target", Xr, Y2, 4)

print("== class-inverse (LabelBinarizer round trip) ==")
ref, fit = pls_ref(iris.data, LabelBinarizer().fit_transform(iy), 2), None
binarizer = LabelBinarizer().fit(iy)
pred_sk = binarizer.inverse_transform(ref["pred"])
pred_rs = binarizer.inverse_transform(gc_ims_rs.pls_fit(iris.data, LabelBinarizer().fit_transform(iy), 2).predict(iris.data))
check("class predictions", pred_sk == pred_rs, np.ones_like(pred_sk, bool))

# ----------------------------------------------------------------------
print("== asymcorr (AsLS) vs original sparse implementation ==")


def asymcorr_py(y, lam=1e7, p=1e-3, niter=20):
    y = np.asarray(y, float)
    L = len(y)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
    w = np.ones(L)
    for _ in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return y - z


for L, lam in [(400, 1e7), (2500, 1e5), (150, 1e9)]:
    xx = np.linspace(0, 1, L)
    sig = np.exp(-((xx - 0.3) ** 2) / 1e-3) * 3 + np.exp(-((xx - 0.7) ** 2) / 5e-4)
    base = 0.2 + 1.2 * xx + 0.8 * xx**2
    yv = sig + base + 0.02 * np.sin(200 * xx)
    z_rs = gc_ims_rs.asymcorr(yv, lam, 1e-3, 20)
    z_py = asymcorr_py(yv, lam, 1e-3, 20)
    check(f"asymcorr L={L} lam={lam:.0e}", z_py, z_rs, tol=1e-6)

# ----------------------------------------------------------------------
print("== scaling vs original numpy implementation ==")


def scale_py(X, method="pareto", mean_centering=True):
    X = np.asarray(X, float).copy()
    if method == "auto":
        weights = 1 / np.std(X, 0)
    elif method == "pareto":
        weights = 1 / np.sqrt(np.std(X, 0))
    elif method == "var":
        weights = 1 / np.var(X, 0)
    elif method is None:
        weights = 1
    else:
        raise ValueError(f"{method} is not a supported method!")
    weights = np.nan_to_num(np.asarray(weights, float), posinf=0, neginf=0)
    if mean_centering:
        return (X - np.mean(X, 0)) * weights
    return X * weights


rng2 = np.random.default_rng(7)
Xs = rng2.normal(size=(50, 12))
Xs[:, 3] = 5.0  # constant column -> inf weight -> 0
for meth in ["auto", "pareto", "var", "none"]:
    for mc in [True, False]:
        r = gc_ims_rs.scale(Xs, meth, mc)
        check(f"scale {meth} mean_centering={mc}", scale_py(Xs, meth if meth != "none" else None, mc), np.asarray(r))

print("== normalize_rows ==")
Xn = rng2.normal(size=(8, 20))
Xn[2] = 3.0
ref_n = (Xn - Xn.min(1, keepdims=True)) / (Xn.max(1, keepdims=True) - Xn.min(1, keepdims=True))
check("normalize_rows", ref_n, np.asarray(gc_ims_rs.normalize_rows(Xn)))
try:
    gc_ims_rs.normalize_rows(np.ones((2, 5)))
    raise SystemExit("FAIL: constant row must raise")
except ValueError:
    print("  constant-row ValueError             OK")

print("\nALL PARITY CHECKS PASSED")
