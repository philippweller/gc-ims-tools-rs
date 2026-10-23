//! Preprocessing primitives: AsLS baseline (asymcorr), feature scaling,
//! min/max normalization. Ports of ims.utils.asymcorr / Dataset.scaling /
//! Spectrum.normalization + Dataset.normalization.

use ndarray::{Array1, Array2, ArrayBase, Axis, Data, Ix1, Ix2};

/// Asymmetric least squares baseline (Eilers), port of
/// `ims.utils.asymcorr(y, lam, p, niter)`. Solves the pentadiagonal
/// system (W + lam * D D^T) z = w y with a band-Cholesky, O(n) per
/// iteration instead of a sparse solve per row.
pub fn asymcorr<S: Data<Elem = f64>>(
    y: &ArrayBase<S, Ix1>,
    lam: f64,
    p: f64,
    niter: usize,
) -> Array1<f64> {
    let l = y.len();
    if l < 3 {
        return y.to_owned();
    }
    let mut weights = Array1::<f64>::ones(l);
    let mut z = y.to_owned();

    // Bands of A = D D^T (D: (L, L-2) second-difference operator).
    // Built by explicit rank-1 accumulation: A += D[:,k] D[:,k]^T.
    // a0 = diag, a1 = first off-diagonal, a2 = second off-diagonal.
    let mut a0 = Array1::<f64>::zeros(l);
    let mut a1 = Array1::<f64>::zeros(l - 1);
    let mut a2 = Array1::<f64>::zeros(l - 2);
    for k in 0..l - 2 {
        // D[k,k] = 1, D[k+1,k] = -2, D[k+2,k] = 1
        let cols = [1.0f64, -2.0, 1.0];
        for (r1, &c1) in cols.iter().enumerate() {
            let i = k + r1;
            for (r2, &c2) in cols.iter().enumerate() {
                let j = k + r2;
                let contrib = c1 * c2;
                match j as isize - i as isize {
                    0 => a0[i] += contrib,
                    1 => a1[i] += contrib,
                    2 => a2[i] += contrib,
                    _ => {}
                }
            }
        }
    }

    // Cholesky solve of (W + lam*A) z = b, band-optimized.
    // Returns z.
    let solve = |b: &Array1<f64>, w: &Array1<f64>| -> Array1<f64> {
        // band storage: d[i] = Z[i,i], l1[i] = Z[i+1,i], l2[i] = Z[i+2,i]
        let mut d = a0.clone();
        d *= lam;
        d += w;
        let l1 = a1.mapv(|v| v * lam);
        let l2 = a2.mapv(|v| v * lam);

        let mut c = Array1::<f64>::zeros(l); // L[i,i]
        let mut s1 = Array1::<f64>::zeros(l - 1); // L[i+1,i]
        let mut s2 = Array1::<f64>::zeros(l - 2); // L[i+2,i]

        for i in 0..l {
            let mut acc = d[i];
            if i >= 1 {
                acc -= s1[i - 1] * s1[i - 1];
            }
            if i >= 2 {
                acc -= s2[i - 2] * s2[i - 2];
            }
            c[i] = acc.sqrt();
            if i + 1 < l {
                let mut acc1 = l1[i];
                if i >= 1 {
                    acc1 -= s2[i - 1] * s1[i - 1];
                }
                s1[i] = acc1 / c[i];
            }
            if i + 2 < l {
                let mut acc2 = l2[i];
                if i >= 1 {
                    acc2 -= s2[i - 1] * s1[i - 1];
                }
                s2[i] = acc2 / c[i];
            }
        }

        // forward: L z1 = b
        let mut z1 = Array1::<f64>::zeros(l);
        for i in 0..l {
            let mut acc = b[i];
            if i >= 1 {
                acc -= s1[i - 1] * z1[i - 1];
            }
            if i >= 2 {
                acc -= s2[i - 2] * z1[i - 2];
            }
            z1[i] = acc / c[i];
        }
        // backward: L^T z = z1
        let mut z = Array1::<f64>::zeros(l);
        for i in (0..l).rev() {
            let mut acc = z1[i];
            if i + 1 < l {
                acc -= s1[i] * z[i + 1];
            }
            if i + 2 < l {
                acc -= s2[i] * z[i + 2];
            }
            z[i] = acc / c[i];
        }
        z
    };

    for _ in 0..niter {
        let rhs = &weights * y;
        z = solve(&rhs, &weights);
        // w = p*(y > z) + (1-p)*(y < z); equality -> 0.0 (numpy semantics)
        for i in 0..l {
            if y[i] > z[i] {
                weights[i] = p;
            } else if y[i] < z[i] {
                weights[i] = 1.0 - p;
            } else {
                weights[i] = 0.0;
            }
        }
    }
    y - z
}

/// Port of Dataset.scaling on a (n, m) matrix (spectra x flattened pixels),
/// ddof=0 standard deviation, nan_to_num with posinf/neginf -> 0.
/// method: "auto" | "pareto" | "var" | None; mean_centering as in the
/// package. Returns the scaled matrix.
pub fn scale<S: Data<Elem = f64>>(
    x: &ArrayBase<S, Ix2>,
    method: &str,
    mean_centering: bool,
) -> Result<Array2<f64>, String> {
    let mut weights = match method {
        "auto" => {
            let std = x.std_axis(Axis(0), 0.0);
            std.mapv(|s| 1.0 / s)
        }
        "pareto" => {
            let std = x.std_axis(Axis(0), 0.0);
            std.mapv(|s| 1.0 / s.sqrt())
        }
        "var" => {
            let var = x.var_axis(Axis(0), 0.0);
            var.mapv(|v| 1.0 / v)
        }
        "" | "none" | "None" => Array1::<f64>::ones(x.ncols()),
        other => return Err(format!("{other} is not a supported method!")),
    };
    // nan_to_num(weights, posinf=0, neginf=0)
    for w in weights.iter_mut() {
        if !w.is_finite() {
            *w = 0.0;
        }
    }
    let out = if mean_centering {
        let mean = x.mean_axis(Axis(0)).ok_or("empty X")?;
        let mut xc = x - &mean;
        for (mut row, &w) in xc.outer_iter_mut().zip(weights.iter()) {
            row *= w;
        }
        xc
    } else {
        let mut xr = x.to_owned();
        for (mut row, &w) in xr.outer_iter_mut().zip(weights.iter()) {
            row *= w;
        }
        xr
    };
    Ok(out)
}

/// Port of Spectrum.normalization on a (n, m) matrix: each row is one
/// flattened spectrum, scaled to [0, 1] by its own min/max.
/// Constant rows raise (as in the Python original).
pub fn normalize_rows<S: Data<Elem = f64>>(
    x: &ArrayBase<S, Ix2>,
) -> Result<Array2<f64>, String> {
    let mut out = x.to_owned();
    for mut row in out.outer_iter_mut() {
        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;
        for &v in row.iter() {
            if v < min {
                min = v;
            }
            if v > max {
                max = v;
            }
        }
        if !(max > min) {
            return Err(
                "spectrum cannot be normalized because it has constant or zero values".into(),
            );
        }
        let denom = max - min;
        for v in row.iter_mut() {
            *v = (*v - min) / denom;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    #[test]
    fn asymcorr_smoke() {
        let y = arr1(&[1.0, 2.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.5, 0.2, 0.1]);
        let z = asymcorr(&y, 1e5, 1e-2, 10);
        assert_eq!(z.len(), y.len());
        assert!(z.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn scale_pareto_smoke() {
        let x = Array2::from_shape_vec((4, 3), (0..12).map(|v| v as f64).collect()).unwrap();
        let out = scale(&x, "pareto", true).unwrap();
        assert_eq!(out.shape(), &[4, 3]);
        // mean-centered: column means ~ 0
        let m = out.mean_axis(Axis(0)).unwrap();
        assert!(m.iter().all(|&v| v.abs() < 1e-12));
    }

    #[test]
    fn normalize_smoke() {
        let x = Array2::from_shape_vec((2, 3), vec![0.0, 5.0, 10.0, 1.0, 1.0, 1.0]).unwrap();
        // second row constant -> error, first row ok
        assert!(normalize_rows(&x).is_err());
        let x2 = Array2::from_shape_vec((1, 3), vec![0.0, 5.0, 10.0]).unwrap();
        let out = normalize_rows(&x2).unwrap();
        assert_eq!(out[[0, 0]], 0.0);
        assert_eq!(out[[0, 2]], 1.0);
    }
}
