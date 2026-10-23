//! `gcims` — standalone CLI for the gc-ims-tools Rust kernel.
//!
//! No Python needed. Subcommands:
//!   plsda fit   -X <train.csv> [--labels <file>] [-c N] [--scaling pareto] -o <model.bin>
//!   plsda pred  -m <model.bin> -X <test.csv> [--labels <file>] [-o <preds.csv>]
//!   plsda vip   -m <model.bin> [-o vip.csv]
//!   asymcorr    -X <data.csv> [--lam 1e7] [--p 1e-3] [--niter 20] -o <out.csv>
//!
//! Input CSV: one sample per row, one feature per column, optional header.
//! The class/label column (header "class" or "label", case-insensitive) is
//! used as y when `--labels` is not given. Otherwise that column is skipped.

use clap::{Parser, Subcommand};
use gc_ims_rs::{pls, preproc};
use ndarray::{Array1, Array2, Axis};
use std::collections::BTreeSet;
use std::error::Error;

// --------------------------------------------------------------------- model

#[derive(serde::Serialize, serde::Deserialize)]
struct Scalers {
    method: String,
    centered: bool,
    mean: Vec<f64>,
    div: Vec<f64>,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct Model {
    fit: pls::PlsFit,
    groups: Vec<String>,
    scaling: Scalers,
    n_components: usize,
    vip: Vec<f64>,
}

// --------------------------------------------------------------------- I/O

/// Read a CSV into numeric features + (optional) labels. A column whose
/// header equals "class"/"label" (case-insensitive) becomes `labels`;
/// `forced_labels` (from --labels) overrides it.
fn read_x_y(path: &str, forced_labels: Option<&str>) -> Result<(Array2<f64>, Option<Vec<String>>), Box<dyn Error>> {
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(true)
        .flexible(true)
        .from_path(path)?;
    let headers = rdr.headers()?.clone();
    let mut feat_idx: Vec<usize> = Vec::new();
    let mut label_idx: Option<usize> = None;
    for (i, h) in headers.iter().enumerate() {
        let hl = h.trim().to_lowercase();
        if forced_labels.is_none() && (hl == "class" || hl == "label") {
            label_idx = Some(i);
        } else {
            feat_idx.push(i);
        }
    }
    if label_idx.is_none() && forced_labels.is_some() {
        // labels come from a separate file: all columns are features
        feat_idx = (0..headers.len()).collect();
    }

    let mut rows: Vec<Vec<f64>> = Vec::new();
    let mut labels: Vec<String> = Vec::new();
    for rec in rdr.records() {
        let rec = rec?;
        let row: Vec<f64> = feat_idx
            .iter()
            .map(|&i| {
                rec.get(i)
                    .unwrap_or("")
                    .trim()
                    .parse::<f64>()
                    .map_err(|e| format!("non-numeric feature at col {i}: {e}"))
            })
            .collect::<Result<_, _>>()?;
        rows.push(row);
        if let Some(li) = label_idx {
            labels.push(rec.get(li).unwrap_or("").trim().to_string());
        }
    }
    let n = rows.len();
    let m = rows.first().map(|r| r.len()).unwrap_or(0);
    let flat: Vec<f64> = rows.into_iter().flat_map(|r| r).collect();
    let x = Array2::from_shape_vec((n, m), flat)?;

    // separate --labels file (one label per line)
    let labels = match (forced_labels, labels.is_empty()) {
        (Some(f), _) => {
            let mut ls = Vec::new();
            for line in std::fs::read_to_string(f)?.lines() {
                let t = line.trim();
                if !t.is_empty() {
                    ls.push(t.to_string());
                }
            }
            Some(ls)
        }
        (None, false) => Some(labels),
        _ => None,
    };
    Ok((x, labels))
}

fn write_cols(path: &str, rows: &[Vec<String>]) -> Result<(), Box<dyn Error>> {
    let mut w = csv::WriterBuilder::new().from_path(path)?;
    for r in rows {
        w.write_record(r)?;
    }
    w.flush()?;
    Ok(())
}

// --------------------------------------------------------------------- scaling

fn fit_scalers(x: &Array2<f64>, method: &str, centered: bool) -> Scalers {
    let mean: Vec<f64> = x.mean_axis(Axis(0)).unwrap_or_else(|| Array1::zeros(x.ncols())).iter().copied().collect();
    let std: Vec<f64> = x.std_axis(Axis(0), 0.0).iter().copied().collect();
    let mut div = match method {
        "auto" => std.iter().map(|s| 1.0 / s).collect::<Vec<f64>>(),
        "pareto" => std.iter().map(|s| 1.0 / s.sqrt()).collect::<Vec<f64>>(),
        "var" => std.iter().map(|s| 1.0 / (s * s)).collect::<Vec<f64>>(),
        _ => vec![1.0; x.ncols()],
    };
    for d in div.iter_mut() {
        if !d.is_finite() {
            *d = 0.0;
        }
    }
    Scalers { method: method.to_string(), centered, mean, div }
}

fn apply_scalers(x: &Array2<f64>, s: &Scalers) -> Array2<f64> {
    let mut out = x.to_owned();
    if s.centered {
        for (mut col, (&m, &d)) in out.axis_iter_mut(Axis(1)).zip(s.mean.iter().zip(s.div.iter())) {
            for v in col.iter_mut() {
                *v = (*v - m) * d;
            }
        }
    } else {
        for (mut col, &d) in out.axis_iter_mut(Axis(1)).zip(s.div.iter()) {
            for v in col.iter_mut() {
                *v *= d;
            }
        }
    }
    out
}

// --------------------------------------------------------------------- pls-da helpers

fn onehot(y: &[String], groups: &[String]) -> Array2<f64> {
    let n = y.len();
    let g = groups.len();
    let mut out = Array2::<f64>::zeros((n, g));
    for (r, label) in y.iter().enumerate() {
        if let Some(pos) = groups.iter().position(|g| g == label) {
            out[[r, pos]] = 1.0;
        }
    }
    out
}

fn argmax(vals: ndarray::ArrayView1<f64>) -> usize {
    let mut best = 0usize;
    let mut bv = f64::NEG_INFINITY;
    for (i, &v) in vals.iter().enumerate() {
        if v > bv {
            bv = v;
            best = i;
        }
    }
    best
}

// --------------------------------------------------------------------- CLI

#[derive(Parser)]
#[command(name = "gcims", version, about = "GC-IMS chemometrics CLI (Rust kernel)")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    Plsda {
        #[command(subcommand)]
        op: PlsdaOp,
    },
    Asymcorr {
        #[arg(short, long)]
        x: String,
        #[arg(long, default_value_t = 1e7)]
        lam: f64,
        #[arg(long, default_value_t = 1e-3)]
        p: f64,
        #[arg(long, default_value_t = 20)]
        niter: usize,
        #[arg(short, long)]
        out: String,
    },
}

#[derive(Subcommand)]
enum PlsdaOp {
    Fit {
        #[arg(short, long)]
        x: String,
        #[arg(long)]
        labels: Option<String>,
        #[arg(short, long, default_value_t = 3)]
        components: usize,
        #[arg(long, default_value = "pareto")]
        scaling: String,
        #[arg(short, long)]
        out: String,
    },
    Pred {
        #[arg(short, long)]
        model: String,
        #[arg(short, long)]
        x: String,
        #[arg(long)]
        labels: Option<String>,
        #[arg(short, long)]
        out: Option<String>,
    },
    Vip {
        #[arg(short, long)]
        model: String,
        #[arg(short, long)]
        out: Option<String>,
    },
}

fn cmd_fit(x: &str, labels: Option<String>, comps: usize, scaling: &str, out: &str) -> Result<(), Box<dyn Error>> {
    let (raw_x, y_opt) = read_x_y(x, labels.as_deref())?;
    let y = y_opt.ok_or("no labels: add a 'class'/'label' column or pass --labels")?;
    if raw_x.nrows() != y.len() {
        return Err("X rows != labels length".into());
    }
    let mut groups: Vec<String> = y.iter().cloned().collect::<BTreeSet<_>>().into_iter().collect();
    if groups.is_empty() {
        groups.push("0".into());
    }
    let x_scaled = apply_scalers(&raw_x, &fit_scalers(&raw_x, scaling, true));
    let yh = onehot(&y, &groups);
    let fit = pls::pls_fit(&x_scaled, &yh, comps, 500, 1e-6)?;
    let xs = pls::pls_transform(&x_scaled, &fit);
    let vip = pls::vip_scores(&fit.x_weights, &xs, &fit.y_loadings);

    // train accuracy
    let dv = pls::pls_predict(&x_scaled, &fit);
    let n = dv.nrows();
    let correct = (0..n).filter(|&r| groups[argmax(dv.row(r))] == y[r]).count();
    let train_acc = correct as f64 / n as f64;
    let model = Model {
        fit,
        groups,
        scaling: fit_scalers(&raw_x, scaling, true),
        n_components: comps,
        vip: vip.iter().copied().collect(),
    };
    std::fs::write(out, bincode::serialize(&model)?)?;
    println!("fit ok: n={n} p={} c={comps} scaling={scaling} train_acc={train_acc:.4} saved={out}", raw_x.ncols());
    Ok(())
}

fn cmd_pred(model_path: &str, x: &str, labels: Option<String>, out: Option<String>) -> Result<(), Box<dyn Error>> {
    let model: Model = bincode::deserialize(&std::fs::read(model_path)?)?;
    let (raw_x, actual_opt) = read_x_y(x, labels.as_deref())?;
    let x_scaled = apply_scalers(&raw_x, &model.scaling);
    let dv = pls::pls_predict(&x_scaled, &model.fit);
    let n = dv.nrows();
    let preds: Vec<String> = (0..n).map(|r| model.groups[argmax(dv.row(r))].clone()).collect();

    let mut metric = String::new();
    if let Some(actual) = actual_opt {
        if actual.len() == n {
            let acc = (0..n).filter(|&r| preds[r] == actual[r]).count() as f64 / n as f64;
            metric = format!("actual given, acc={:.4}", acc);
        }
    }
    match out {
        Some(path) => {
            let rows: Vec<Vec<String>> = preds.iter().map(|p| vec![p.clone()]).collect();
            write_cols(&path, &rows)?;
            println!("pred ok: n={n} {metric} saved={path}");
        }
        None => {
            for p in &preds {
                println!("{p}")
            }
            println!("# {metric}");
        }
    }
    Ok(())
}

fn cmd_vip(model_path: &str, out: Option<String>) -> Result<(), Box<dyn Error>> {
    let model: Model = bincode::deserialize(&std::fs::read(model_path)?)?;
    if let Some(path) = out {
        let rows: Vec<Vec<String>> = model.vip.iter().map(|v| vec![v.to_string()]).collect();
        write_cols(&path, &rows)?;
        println!("vip saved={path} ({})", model.vip.len());
    } else {
        for (i, v) in model.vip.iter().enumerate() {
            println!("{i}\t{v}")
        }
    }
    Ok(())
}

fn cmd_asymcorr(x: &str, lam: f64, p: f64, niter: usize, out: &str) -> Result<(), Box<dyn Error>> {
    let (data, _) = read_x_y(x, None)?;
    let mut rows = Vec::with_capacity(data.nrows());
    for r in data.outer_iter() {
        rows.push(preproc::asymcorr(&r.to_owned(), lam, p, niter));
    }
    let n = rows.len();
    let m = rows.first().map(|r| r.len()).unwrap_or(0);
    let out_mat = Array2::from_shape_vec((n, m), rows.into_iter().flat_map(|r| r.to_vec()).collect())?;
    let rows_out: Vec<Vec<String>> = out_mat.rows().into_iter()
        .map(|row| row.iter().map(|v| v.to_string()).collect())
        .collect();
    write_cols(out, &rows_out)?;
    println!("asymcorr ok: n={n} m={m} lam={lam} saved={out}");
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Plsda { op } => match op {
            PlsdaOp::Fit { x, labels, components, scaling, out } => cmd_fit(&x, labels, components, &scaling, &out)?,
            PlsdaOp::Pred { model, x, labels, out } => cmd_pred(&model, &x, labels, out)?,
            PlsdaOp::Vip { model, out } => cmd_vip(&model, out)?,
        },
        Cmd::Asymcorr { x, lam, p, niter, out } => cmd_asymcorr(&x, lam, p, niter, &out)?,
    }
    Ok(())
}