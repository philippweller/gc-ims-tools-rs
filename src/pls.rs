//! PLS kernel: exact reimplementation of scikit-learn 1.1.x
//! `PLSRegression(n_components, scale=False, algorithm="nipals")`,
//! which is what gc-ims-tools wraps in PLS_DA / PLSR.
//!
//! Reference algorithm (sklearn/cross_decomposition/_pls.py, v1.1.1):
//! - center X and Y (no scaling),
//! - per component: power method on the cross-covariance, sign flip,
//!   regression-mode deflation,
//! - rotations via pinv (SVD, rcond = eps * max(M, N)),
//! - predict: (X - x_mean) @ coef + y_mean.

use ndarray::{s, Array1, Array2, ArrayBase, Axis, Data, Ix2};
pub const EPS: f64 = f64::EPSILON;

/// Symmetric eigen-decomposition via cyclic Jacobi rotations.
/// Returns (eigenvalues descending, eigenvectors as columns).
/// Only used on small (n_components x n_components) Gram matrices.
fn jacobi_eigh(mut a: Array2<f64>) -> (Array1<f64>, Array2<f64>) {
    let n = a.nrows();
    let mut v = Array2::<f64>::eye(n);
    let mut off = 0.0f64;
    for i in 0..n {
        for j in 0..n {
            if i != j {
                off += a[[i, j]] * a[[i, j]];
            }
        }
    }
    off = off.sqrt();
    let mut sweeps = 0;
    while off > 1e-15 && sweeps < 100 {
        off = 0.0;
        for p in 0..n {
            for q in (p + 1)..n {
                let apq = a[[p, q]];
                if apq.abs() <= 1e-300 {
                    continue;
                }
                let app = a[[p, p]];
                let aqq = a[[q, q]];
                let theta = (aqq - app) / (2.0 * apq);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                // rotate rows/cols p,q
                for k in 0..n {
                    if k != p && k != q {
                        let akp = a[[k, p]];
                        let akq = a[[k, q]];
                        a[[k, p]] = c * akp - s * akq;
                        a[[p, k]] = a[[k, p]];
                        a[[k, q]] = s * akp + c * akq;
                        a[[q, k]] = a[[k, q]];
                    }
                }
                a[[p, p]] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
                a[[q, q]] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
                a[[p, q]] = 0.0;
                a[[q, p]] = 0.0;
                // accumulate eigenvectors
                for k in 0..n {
                    let vkp = v[[k, p]];
                    let vkq = v[[k, q]];
                    v[[k, p]] = c * vkp - s * vkq;
                    v[[k, q]] = s * vkp + c * vkq;
                }
            }
        }
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    off += a[[i, j]] * a[[i, j]];
                }
            }
        }
        off = off.sqrt();
        sweeps += 1;
    }
    let mut evals: Array1<f64> = a.diag().to_owned();
    // sort descending, permute eigenvectors
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| evals[j].partial_cmp(&evals[i]).unwrap());
    let mut vs = Array2::<f64>::zeros((n, n));
    let mut es = Array1::<f64>::zeros(n);
    for (pos, &idx) in order.iter().enumerate() {
        es[pos] = evals[idx].max(0.0);
        vs.column_mut(pos).assign(&v.column(idx));
    }
    (es, vs)
}

/// Moore-Penrose pseudo-inverse (scipy `pinv2`-style rcond cutoff):
/// pinv(M) = V diag(1/lambda) V^T M^T from the eigen-decomposition of
/// M^T M. Inputs are small (n_components x n_components) rotation
/// matrices, so the Gram approach is accurate enough and keeps the crate
/// free of LAPACK/OpenBLAS dependencies.
pub fn pinv_rcond(m: &Array2<f64>, rcond: f64) -> Array2<f64> {
    let a = m.t().dot(m);
    let (evals, v) = jacobi_eigh(a);
    let lmax = evals[0].max(f64::MIN_POSITIVE);
    let mut s_inv = Array2::<f64>::zeros((m.ncols(), m.ncols()));
    for k in 0..evals.len() {
        if evals[k] > rcond * lmax {
            s_inv[[k, k]] = 1.0 / evals[k];
        }
    }
    v.dot(&s_inv).dot(&v.t()).dot(&m.t())
}

fn argmax_abs(v: &Array1<f64>) -> usize {
    let mut idx = 0usize;
    let mut best = 0.0f64;
    for (i, &x) in v.iter().enumerate() {
        let ax = x.abs();
        if ax > best {
            best = ax;
            idx = i;
        }
    }
    idx
}

/// sklearn `_svd_flip_1d`: flip sign so the largest-|u| entry is positive.
fn svd_flip_1d(u: &mut Array1<f64>, v: &mut Array1<f64>) {
    let sign = u[argmax_abs(u)].signum();
    *u *= sign;
    *v *= sign;
}

/// sklearn `_get_first_singular_vectors_power_method` (mode "A",
/// norm_y_weights = false). Returns (x_weights, y_weights, n_iter).
fn power_method(
    x: &Array2<f64>,
    y: &Array2<f64>,
    max_iter: usize,
    tol: f64,
    q: usize,
) -> Result<(Array1<f64>, Array1<f64>, usize), String> {
    // initial y_score: first Y column with any |value| > eps
    let j0 = (0..q)
        .find(|&j| y.column(j).iter().any(|&v| v.abs() > EPS))
        .ok_or_else(|| "Y residual is constant".to_string())?;
    let mut y_score = y.column(j0).to_owned();

    let p = x.ncols();
    let mut x_weights = Array1::<f64>::zeros(p);
    let mut y_weights = Array1::<f64>::zeros(q);
    let mut x_weights_old: Option<Array1<f64>> = None;

    let mut n_iter = 0usize;
    for i in 0..max_iter {
        // x_weights = X^T y_score / (y_score . y_score)
        let yy = y_score.dot(&y_score);
        x_weights = x.t().dot(&y_score) / yy;
        x_weights /= x_weights.dot(&x_weights).sqrt() + EPS;

        let x_score = x.dot(&x_weights);
        let ss = x_score.dot(&x_score);

        // y_weights = Y^T x_score / (x_score . x_score)
        y_weights = y.t().dot(&x_score) / ss;

        // y_score = Y y_weights / (y_weights . y_weights + eps)
        let wy = y_weights.dot(&y_weights) + EPS;
        y_score = y.dot(&y_weights) / wy;

        // convergence on x_weights (first iteration: compare to 100.0,
        // numpy broadcasts the initial scalar -> huge diff -> no break)
        let diff = match &x_weights_old {
            None => x_weights.mapv(|v| v - 100.0),
            Some(old) => &x_weights - old,
        };
        n_iter = i + 1;
        if diff.dot(&diff) < tol || q == 1 {
            break;
        }
        x_weights_old = Some(x_weights.clone());
    }
    Ok((x_weights, y_weights, n_iter))
}

/// Result of a PLS fit, mirroring the sklearn attributes gc-ims-tools uses.
#[derive(Clone)]
pub struct PlsFit {
    pub x_weights: Array2<f64>,  // (p, A)
    pub y_weights: Array2<f64>,   // (q, A)
    pub x_loadings: Array2<f64>,  // (p, A)
    pub y_loadings: Array2<f64>,  // (q, A)
    pub x_rotations: Array2<f64>, // (p, A)
    pub y_rotations: Array2<f64>, // (q, A)
    pub coef: Array2<f64>,        // (p, q) -- like sklearn coef_ (pre-1.3 layout)
    pub intercept: Array1<f64>,   // (q,)
    pub x_mean: Array1<f64>,      // (p,)
    pub y_mean: Array1<f64>,      // (q,)
}

/// Exact port of `PLSRegression.fit` (scale=False, nipals, regression
/// deflation) plus the rotations/coef/intercept computation.
pub fn pls_fit<SX, SY>(
    x: &ArrayBase<SX, Ix2>,
    y: &ArrayBase<SY, Ix2>,
    n_components: usize,
    max_iter: usize,
    tol: f64,
) -> Result<PlsFit, String>
where
    SX: Data<Elem = f64>,
    SY: Data<Elem = f64>,
{
    let (n, p) = x.dim();
    let (yn, q) = y.dim();
    if n != yn {
        return Err("X and Y have inconsistent numbers of samples".into());
    }
    let a = n_components.min(p);
    if a == 0 {
        return Err("n_components must be >= 1".into());
    }

    let x_mean = x.mean_axis(Axis(0)).ok_or("empty X")?;
    let y_mean = y.mean_axis(Axis(0)).ok_or("empty Y")?;
    let mut xk = x - &x_mean;
    let mut yk = y - &y_mean;

    let mut x_weights = Array2::<f64>::zeros((p, a));
    let mut y_weights = Array2::<f64>::zeros((q, a));
    let mut x_loadings = Array2::<f64>::zeros((p, a));
    let mut y_loadings = Array2::<f64>::zeros((q, a));

    for k in 0..a {
        // sklearn: zero out Y columns that are (near-)constant w.r.t. eps
        if q > 0 {
            let mut found = false;
            for j in 0..q {
                let col = yk.column(j);
                let const_col = col.iter().all(|&v| v.abs() < 10.0 * EPS);
                if const_col {
                    yk.slice_mut(s![.., j]).fill(0.0);
                } else {
                    found = true;
                }
            }
            if !found {
                // "Y residual is constant" -> sklearn warns and stops early
                break;
            }
        }

        let (wu, wv, _nit) = power_method(&xk, &yk, max_iter, tol, q)?;
        let mut wu = wu;
        let mut wv = wv;
        svd_flip_1d(&mut wu, &mut wv);

        // scores
        let x_scores = xk.dot(&wu); // (n,)
        let y_ss = wv.dot(&wv);
        // y_scores = Yk v / (v.v)  (needed for _y_scores; not for rotations)
        let _y_scores = yk.dot(&wv) / y_ss;

        // loadings + regression-mode deflation
        let x_ss = x_scores.dot(&x_scores);
        let pl = xk.t().dot(&x_scores) / x_ss; // (p,)
        let ql = yk.t().dot(&x_scores) / x_ss; // (q,)

        for (mut row, &t) in xk.outer_iter_mut().zip(x_scores.iter()) {
            row -= &(&pl * t);
        }
        for (mut row, &t) in yk.outer_iter_mut().zip(x_scores.iter()) {
            row -= &(&ql * t);
        }

        x_weights.slice_mut(s![.., k]).assign(&wu);
        y_weights.slice_mut(s![.., k]).assign(&wv);
        x_loadings.slice_mut(s![.., k]).assign(&pl);
        y_loadings.slice_mut(s![.., k]).assign(&ql);
    }

    // rotations: W @ pinv(P^T W), same rcond convention as scipy pinv2
    let rcond = EPS * a as f64;
    let xtw = x_loadings.t().dot(&x_weights);
    let ytw = y_loadings.t().dot(&y_weights);
    let x_rotations = x_weights.dot(&pinv_rcond(&xtw, rcond));
    let y_rotations = y_weights.dot(&pinv_rcond(&ytw, rcond));

    // coef (p, q); sklearn stores (q, p) internally and exposes coef_ = .T
    let coef = x_rotations.dot(&y_loadings.t());

    Ok(PlsFit {
        x_weights,
        y_weights,
        x_loadings,
        y_loadings,
        x_rotations,
        y_rotations,
        coef,
        intercept: y_mean.clone(),
        x_mean,
        y_mean,
    })
}

/// transform(X) == (X - x_mean) @ x_rotations  (scale=False).
pub fn pls_transform<S: Data<Elem = f64>>(
    x: &ArrayBase<S, Ix2>,
    fit: &PlsFit,
) -> Array2<f64> {
    (x - &fit.x_mean).dot(&fit.x_rotations)
}

/// Generic (M - mean) @ rotations helper for y scores.
pub fn pls_reconstruct<S: Data<Elem = f64>>(
    m: &ArrayBase<S, Ix2>,
    rotations: &Array2<f64>,
) -> Array2<f64> {
    m.dot(rotations)
}

/// predict(X) == (X - x_mean) @ coef + y_mean.
pub fn pls_predict<S: Data<Elem = f64>>(
    x: &ArrayBase<S, Ix2>,
    fit: &PlsFit,
) -> Array2<f64> {
    let mut out = (x - &fit.x_mean).dot(&fit.coef);
    for mut row in out.outer_iter_mut() {
        row += &fit.intercept;
    }
    out
}

/// VIP scores exactly as gc-ims-tools' `ims.utils.vip_scores`:
/// T here is the *transform* scores (rotated), matching what the Python
/// package passes in (`model.pls.transform(...)`).
pub fn vip_scores<S1, S2, S3>(
    w: &ArrayBase<S1, Ix2>,
    t: &ArrayBase<S2, Ix2>,
    q: &ArrayBase<S3, Ix2>,
) -> Array1<f64>
where
    S1: Data<Elem = f64>,
    S2: Data<Elem = f64>,
    S3: Data<Elem = f64>,
{
    let (p, a) = w.dim();
    // W0 = W / sqrt(sum(W^2, axis=0))
    let mut w0 = w.to_owned();
    for j in 0..a {
        let mut col = w0.slice_mut(s![.., j]);
        let nrm = col.dot(&col).sqrt();
        col /= nrm;
    }
    // sum_sq = sum(T^2,0) * sum(Q^2,0)
    let mut sum_sq = Array1::<f64>::zeros(a);
    for j in 0..a {
        let tj = t.column(j);
        let qj = q.column(j);
        sum_sq[j] = tj.dot(&tj) * qj.dot(&qj);
    }
    let total = sum_sq.sum();
    let mut vips = Array1::<f64>::zeros(p);
    if total > 0.0 {
        for i in 0..p {
            let mut acc = 0.0;
            for j in 0..a {
                let wn = w0[[i, j]];
                acc += sum_sq[j] * wn * wn;
            }
            vips[i] = (p as f64 * acc / total).sqrt();
        }
    }
    vips
}

/// Selectivity ratio as `ims.utils.selectivity_ratio(X, coefficients.T)`.
/// `coef` has the (n_features, n_targets) layout (sklearn 1.x coef_).
/// The original numpy expression only broadcasts sensibly for a single
/// target; we implement the same math per column (exactly equivalent
/// for q == 1, well-defined for q > 1).
pub fn selectivity_ratio<S1, S2>(
    x: &ArrayBase<S1, Ix2>,
    coef: &ArrayBase<S2, Ix2>,
) -> Array1<f64>
where
    S1: Data<Elem = f64>,
    S2: Data<Elem = f64>,
{
    let b_norm = coef.iter().map(|v| v * v).sum::<f64>().sqrt();
    let t = x.dot(coef) / b_norm; // (n, q)
    let (_, q) = t.dim();
    let p = x.ncols();
    let mut p_tp = Array2::<f64>::zeros((p, q));
    for j in 0..q {
        let tj = t.column(j);
        let s = tj.dot(&tj);
        p_tp.slice_mut(s![.., j]).assign(&(x.t().dot(&tj) / s));
    }
    let exp_var = t.dot(&p_tp.t()); // (n, p)
    let ss_expl = exp_var.mapv(|v| v * v).sum_axis(Axis(0));
    let exp_res = (x - &exp_var.mapv(|v| v * v)).mapv(|v| v * v);
    let ss_res = exp_res.sum_axis(Axis(0));
    ss_expl / ss_res
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tiny smoke test: 10x4 centered-ish data, 2 components, 2 targets.
    #[test]
    fn fit_smoke() {
        let x = Array2::from_shape_vec((10, 4), (0..40).map(|v| v as f64).collect()).unwrap();
        let y = Array2::from_shape_vec((10, 2), (0..20).map(|v| (v % 3) as f64).collect()).unwrap();
        let fit = pls_fit(&x, &y, 2, 500, 1e-6).unwrap();
        assert_eq!(fit.coef.dim(), (4, 2));
        assert_eq!(fit.x_rotations.dim(), (4, 2));
        let pred = pls_predict(&x, &fit);
        assert_eq!(pred.dim(), (10, 2));
        assert!(pred.iter().all(|v| v.is_finite()));
        let vip = vip_scores(&fit.x_weights, &pls_transform(&x, &fit), &fit.y_loadings);
        assert_eq!(vip.len(), 4);
        assert!(vip.iter().all(|v| v.is_finite()));
        let sr = selectivity_ratio(&x, &fit.coef);
        assert_eq!(sr.len(), 4);
    }
}
