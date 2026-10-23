"""End-to-end drop-in test: the *installed* ims.PLS_DA / ims.PLSR running on
the Rust kernel (after `import gc_ims_rs`) must match the scikit-learn
results the package would otherwise produce.

Because `import gc_ims_rs` auto-patches, we compare a PLS_DA built under Rust
against a reference PLSRegression fit with the same binarization, and the
class's predict() class labels / accuracy against sklearn.
"""
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

import gc_ims_rs
assert gc_ims_rs.is_active(), "drop-in patch must be active"

from ims.plsda import PLS_DA
from ims.plsr import PLSR

iris = load_iris()
X, y = iris.data, iris.target_names[iris.target]

# --- PLS_DA on Rust ---
m = PLS_DA(None, n_components=3)
assert type(m._sk_pls).__name__ == "RustPLS", type(m._sk_pls)
m.fit(X, y)
y_pred = m.predict(X)
acc_rust = accuracy_score(y, y_pred)

# --- reference: same binarization via LabelBinarizer, sklearn PLSRegression ---
from sklearn.preprocessing import LabelBinarizer

Yb = LabelBinarizer().fit_transform(y)
pls = PLSRegression(n_components=3, scale=False).fit(X, Yb)
pred_bin = pls.predict(X)
groups = m.groups
y_ref = np.array([groups[i] for i in pred_bin.argmax(1)])
acc_ref = accuracy_score(y, y_ref)

print("PLS_DA engine :", type(m._sk_pls).__name__)
print("acc (rust)    :", round(acc_rust, 4))
print("acc (sklearn) :", round(acc_ref, 4))
assert acc_rust == acc_ref, "accuracy mismatch rust vs sklearn"
assert np.array_equal(y_pred, y_ref), "class labels mismatch"

# --- PLSR on Rust (multi-target regression) ---
rng = np.random.default_rng(0)
Xr = rng.normal(size=(100, 20))
Yr = Xr @ rng.normal(size=(20, 3)) + 0.1 * rng.normal(size=(100, 3))
mr = PLSR(None, n_components=4)
assert type(mr._sk_pls).__name__ == "RustPLS"
mr.fit(Xr, Yr)
pred = np.asarray(mr.predict(Xr))
ref_pred = PLSRegression(n_components=4, scale=False).fit(Xr, Yr).predict(Xr)
d = np.max(np.abs(pred - ref_pred))
print("PLSR max|pred diff|:", d)
assert d < 1e-10, "PLSR predict mismatch"

print("\nDROP-IN OK: ims.PLS_DA / ims.PLSR run on the Rust kernel")