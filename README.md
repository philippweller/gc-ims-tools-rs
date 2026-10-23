# gc-ims-tools-rs

Rust numerical kernel for [gc-ims-tools](https://github.com/Charisma-Mannheim/gc-ims-tools)
(chemometrics for GC-IMS data), exposed to Python via PyO3.

The goal: keep the gc-ims-tools Python API and plotting unchanged while running
all heavy numerics in Rust — fused, allocation-light single-pass preprocessing,
a true-NIPALS PLS kernel, and the chemometrics models (PCA/PLS-DA/PLS-R/SIMCA/HCA
planned) with rayon parallelism where it pays.

## Status

Verified byte-parity with the Python reference (scikit-learn 1.1.x
`PLSRegression(scale=False)` + gc-ims-tools wrapper logic) on iris, wine,
breast-cancer and synthetic regression data — max |diff| ≤ 1e-13 on every
checked quantity (`x/y_scores`, weights, loadings, `coef`, intercept, predict,
VIP, selectivity ratio). See `tests/parity.py`.

- PLS kernel (fit / predict / transform / VIP / selectivity ratio): done & parity-checked
- Preprocessing: AsLS baseline (asymcorr), scaling, min/max normalization: implemented
  (asymcorr still has a numerical fix in progress)
- Python wrapper (plsda.py / plsr.py using the kernel with sklearn fallback): in progress
- Planned: PCA, SIMCA, HCA, wavelets, DTW, peak finding, file I/O

## Build (macOS, x86_64 conda Python)

The user's scientific Python is an x86_64 conda build under Rosetta, so build
the wheel for that target explicitly:

```sh
maturin build --target x86_64-apple-darwin \
  --interpreter /Users/philippweller/opt/miniconda3/bin/python --release
pip install --force-reinstall target/wheels/gc_ims_tools_rs-*.whl
```

## License

BSD-3-Clause.