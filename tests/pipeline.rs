//! End-to-end runs: reproducibility and thread-count independence.

use std::path::Path;

use splatter_sc::params::{Layout, Method, OutputParams, SplatParams};

/// Small groups + batches parameter set writing every layout to `dir`.
///
/// ### Params
///
/// * `dir` - Output directory
/// * `seed` - Seed
///
/// ### Returns
///
/// Resolved parameters spanning several chunks.
fn small_params(dir: &Path, seed: u64) -> SplatParams {
    SplatParams {
        n_genes: 300,
        batch_cells: vec![5000, 4000],
        group_prob: vec![0.2, 0.3, 0.5],
        method: Method::Groups,
        lib_loc: 8.0,
        seed: Some(seed),
        output: OutputParams {
            dir: dir.to_path_buf(),
            layouts: vec![Layout::Parse, Layout::TenxMtx],
        },
        ..Default::default()
    }
    .resolve()
    .unwrap()
}

/// Run a simulation on a pool with `threads` threads.
///
/// ### Params
///
/// * `params` - Resolved parameters
/// * `threads` - Rayon thread count
fn run_with_threads(params: SplatParams, threads: usize) {
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap()
        .install(|| splatter_sc::run(params).unwrap());
}

/// Files whose bytes must match between runs.
const FILES: &[&str] = &[
    "DGE.mtx",
    "cell_metadata.csv",
    "all_genes.csv",
    "cells_truth.tsv.gz",
    "genes_truth.tsv.gz",
    "tenx/matrix.mtx.gz",
    "tenx/barcodes.tsv.gz",
    "tenx/features.tsv.gz",
];

#[test]
fn test_run_identical_across_thread_counts() {
    let a = tempfile::tempdir().unwrap();
    let b = tempfile::tempdir().unwrap();
    run_with_threads(small_params(a.path(), 7), 1);
    run_with_threads(small_params(b.path(), 7), 4);
    for f in FILES {
        let x = std::fs::read(a.path().join(f)).unwrap();
        let y = std::fs::read(b.path().join(f)).unwrap();
        assert!(x == y, "{f} differs between 1 and 4 threads");
    }
}

#[test]
fn test_run_different_seed_differs() {
    let a = tempfile::tempdir().unwrap();
    let b = tempfile::tempdir().unwrap();
    run_with_threads(small_params(a.path(), 7), 2);
    run_with_threads(small_params(b.path(), 8), 2);
    let x = std::fs::read(a.path().join("DGE.mtx")).unwrap();
    let y = std::fs::read(b.path().join("DGE.mtx")).unwrap();
    assert!(x != y);
}

#[test]
fn test_run_parse_mtx_header_matches_body() {
    let a = tempfile::tempdir().unwrap();
    run_with_threads(small_params(a.path(), 9), 2);
    let text = std::fs::read_to_string(a.path().join("DGE.mtx")).unwrap();
    let mut lines = text.lines();
    assert_eq!(
        lines.next().unwrap(),
        "%%MatrixMarket matrix coordinate integer general"
    );
    let dims: Vec<usize> = lines
        .next()
        .unwrap()
        .split(' ')
        .map(|x| x.parse().unwrap())
        .collect();
    assert_eq!(&dims[..2], &[9000, 300]);
    let body: Vec<&str> = lines.collect();
    assert_eq!(body.len(), dims[2]);
    let last: Vec<usize> = body
        .last()
        .unwrap()
        .split(' ')
        .map(|x| x.parse().unwrap())
        .collect();
    assert!(last[0] <= 9000 && last[1] <= 300 && last[2] > 0);
    assert!(!a.path().join("DGE.mtx.body.tmp").exists());
}
