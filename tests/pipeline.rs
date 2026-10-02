//! End-to-end runs: reproducibility and thread-count independence.

use std::io::Read;
use std::path::Path;

use flate2::read::MultiGzDecoder;

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
            layouts: vec![Layout::Parse, Layout::TenxMtx, Layout::H5ad, Layout::TenxH5],
            h5_compression: Some(1),
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

/// Files whose bytes must match between runs. The HDF5 files are left out:
/// libhdf5 stores object modification times by default. Their content is
/// checked against the mtx by `test_run_h5_and_tenx_match_parse_mtx`.
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

/// Zero-based `(cell, gene, count)` triplets.
type Triplets = Vec<(usize, usize, u32)>;

/// `(cell, gene, count)` triplets, zero-based, sorted, from mtx text.
///
/// ### Params
///
/// * `text` - The mtx file
/// * `cells_first` - Whether rows are cells
///
/// ### Returns
///
/// `(dims, triplets)` with dims as `(n_cells, n_genes)`.
fn mtx_triplets(text: &str, cells_first: bool) -> ((usize, usize), Triplets) {
    let mut lines = text.lines().skip(1);
    let parse = |l: &str| -> Vec<usize> { l.split(' ').map(|x| x.parse().unwrap()).collect() };
    let d = parse(lines.next().unwrap());
    let mut t: Vec<_> = lines
        .map(|l| {
            let v = parse(l);
            let (c, g) = if cells_first {
                (v[0], v[1])
            } else {
                (v[1], v[0])
            };
            (c - 1, g - 1, v[2] as u32)
        })
        .collect();
    assert_eq!(t.len(), d[2]);
    t.sort_unstable();
    let dims = if cells_first {
        (d[0], d[1])
    } else {
        (d[1], d[0])
    };
    (dims, t)
}

/// `(cell, gene, count)` triplets, sorted, from a cell-major CSR.
///
/// ### Params
///
/// * `indptr` - Row pointers over cells
/// * `indices` - Gene indices
/// * `data` - Counts
///
/// ### Returns
///
/// The triplets.
fn csr_triplets(indptr: &[i64], indices: &[i64], data: &[u32]) -> Triplets {
    let mut t = Vec::with_capacity(data.len());
    for (c, w) in indptr.windows(2).enumerate() {
        for j in w[0] as usize..w[1] as usize {
            t.push((c, indices[j] as usize, data[j]));
        }
    }
    t.sort_unstable();
    t
}

#[test]
fn test_run_h5_and_tenx_match_parse_mtx() {
    let a = tempfile::tempdir().unwrap();
    run_with_threads(small_params(a.path(), 11), 2);
    let text = std::fs::read_to_string(a.path().join("DGE.mtx")).unwrap();
    let (dims, parse) = mtx_triplets(&text, true);
    assert_eq!(dims, (9000, 300));

    // The 10x mtx is several gzip members back to back.
    let mut text = String::new();
    MultiGzDecoder::new(std::fs::File::open(a.path().join("tenx/matrix.mtx.gz")).unwrap())
        .read_to_string(&mut text)
        .unwrap();
    assert_eq!(mtx_triplets(&text, false), (dims, parse.clone()));

    let f = hdf5::File::open(a.path().join("DGE.h5ad")).unwrap();
    let x = f.group("X").unwrap();
    let shape: Vec<i64> = x.attr("shape").unwrap().read_raw().unwrap();
    assert_eq!(shape, vec![9000, 300]);
    let indptr: Vec<i64> = x.dataset("indptr").unwrap().read_raw().unwrap();
    let indices: Vec<i32> = x.dataset("indices").unwrap().read_raw().unwrap();
    let data: Vec<f32> = x.dataset("data").unwrap().read_raw().unwrap();
    let indices: Vec<i64> = indices.into_iter().map(i64::from).collect();
    let data: Vec<u32> = data.into_iter().map(|v| v as u32).collect();
    assert_eq!(indptr.len(), 9001);
    assert_eq!(csr_triplets(&indptr, &indices, &data), parse);

    let f = hdf5::File::open(a.path().join("matrix.h5")).unwrap();
    let m = f.group("matrix").unwrap();
    let shape: Vec<i32> = m.dataset("shape").unwrap().read_raw().unwrap();
    assert_eq!(shape, vec![300, 9000]);
    let indptr: Vec<i64> = m.dataset("indptr").unwrap().read_raw().unwrap();
    let indices: Vec<i64> = m.dataset("indices").unwrap().read_raw().unwrap();
    let data: Vec<i32> = m.dataset("data").unwrap().read_raw().unwrap();
    let data: Vec<u32> = data.into_iter().map(|v| v as u32).collect();
    assert_eq!(indptr.len(), 9001);
    assert_eq!(csr_triplets(&indptr, &indices, &data), parse);
}
