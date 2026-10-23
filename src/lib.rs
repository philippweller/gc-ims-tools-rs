//! Rust numerical kernel for gc-ims-tools.
//!
//! Pure numerics (PLS NIPALS, VIP, selectivity ratio, AsLS baselines,
//! scaling, normalization) live in [`pls`] and [`preproc`], usable from Rust
//! (the `gcims` CLI) and, when built with the `python` feature, from Python
//! via PyO3 (module `gc_ims_rs`).

pub mod pls;
pub mod preproc;

/// PyO3 bindings (the Python extension module). Only compiled with the
/// `python` feature, which the maturin build activates.
#[cfg(feature = "python")]
mod pybindings;