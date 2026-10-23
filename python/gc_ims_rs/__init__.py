"""gc_ims_rs: Rust numerical kernel for gc-ims-tools, with drop-in patch.

`import gc_ims_rs` gives the compiled kernels (pls_fit, PlsFit, vip_scores,
selectivity_ratio, asymcorr, scale, normalize_rows) and triggers the optional
drop-in patch that reroutes the *installed* `ims` package (PLS_DA / PLSR) to
this Rust engine.
"""

# Re-export the compiled symbols into the package namespace:
# the extension lives here as gc_ims_rs.abi3.so (submodule `gc_ims_rs`).
from . import gc_ims_rs as _kem  # noqa: E402
from .gc_ims_rs import *  # noqa: F401,F403,E402
from . import ims_rs  # noqa: E402
from .ims_rs import install as _patch, is_active  # noqa: E402

# Drop-in: reroute the installed ims package to Rust (no-op if ims missing).
_patch()

__version__ = "0.1.0"