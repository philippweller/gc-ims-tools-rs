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
VIP, selectivity ratio). `tests/parity.py` and `tests/dropin.py` both pass.

- PLS kernel (fit / predict / transform / VIP / selectivity ratio): done & parity-checked
- Preprocessing: AsLS baseline (asymcorr, band-Cholesky O(n)), scaling, min/max normalization: done & parity-checked
- Drop-in Python wrapper: `import gc_ims_rs` reroutes the *installed* `ims`
  package (PLS_DA / PLSR) to Rust — identical API, plots and results; sklearn
  fallback if the extension is missing. Also fixes the upstream `utils
  .selectivity_ratio` crash for multi-target regression (per-column, safe for
  q > 1).
- Planned: PCA, SIMCA, HCA, wavelets, DTW, peak finding, file I/O

## Usage (drop-in)

```python
import gc_ims_rs          # auto-patches the installed ims package
from ims import PLS_DA, PLSR   # now running on the Rust kernel
print(gc_ims_rs.is_active())
```

## CLI (no Python needed)

`gcims` is a standalone Rust binary (install: `cargo install --path .` or copy
`target/release/gcims`). Input CSVs: one sample per row; a column named
`class`/`label` is the target (or pass `--labels <file>`).

```sh
# fit a PLS-DA model (pareto scaling, 3 components), save model + train acc
gcims plsda fit -x train.csv -c 3 --scaling pareto -o model.bin

# predict classes for new samples (uses the model's stored scaling + classes)
gcims plsda pred -m model.bin -x test.csv -o preds.csv

# variable importance (VIP), write feature index/value
gcims plsda vip -m model.bin -o vip.csv

# AsLS baseline correction, per row
gcims asymcorr -x data.csv --lam 1e7 -o corrected.csv
```

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