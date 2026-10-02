//! Names and ground-truth tables, using splatter's `colData` / `rowData`
//! column names.

use std::io::Write;

use crate::cells::CellMeta;
use crate::genes::GeneTruth;
use crate::params::{Method, SplatParams};

/////////////
// Helpers //
/////////////

/// Cell name as splatter writes it.
///
/// ### Params
///
/// * `i` - Zero-based cell index
///
/// ### Returns
///
/// `Cell<i + 1>`.
pub fn cell_name(i: usize) -> String {
    format!("Cell{}", i + 1)
}

/// Gene name as splatter writes it.
///
/// ### Params
///
/// * `g` - Zero-based gene index
///
/// ### Returns
///
/// `Gene<g + 1>`.
pub fn gene_name(g: usize) -> String {
    format!("Gene{}", g + 1)
}

/// Per-cell truth table: id, batch, group (groups mode only, as splatter),
/// expected library size.
///
/// ### Params
///
/// * `params` - Resolved parameters
/// * `cells` - Cell metadata
/// * `sep` - Field separator
/// * `id_col` - Header of the id column (`Cell`, or `bc_wells` for Parse)
/// * `batch_col` - Header of the batch column (`Batch`, or `sample` for Parse)
///
/// ### Returns
///
/// The table as bytes, with a header line.
pub fn cell_table(
    params: &SplatParams,
    cells: &CellMeta,
    sep: char,
    id_col: &str,
    batch_col: &str,
) -> Vec<u8> {
    let groups = params.method == Method::Groups;
    let mut out = Vec::with_capacity(cells.n_cells() * 40);
    write!(out, "{id_col}{sep}{batch_col}").unwrap();
    if groups {
        write!(out, "{sep}Group").unwrap();
    }
    writeln!(out, "{sep}ExpLibSize").unwrap();
    for c in 0..cells.n_cells() {
        write!(out, "Cell{}{sep}Batch{}", c + 1, cells.batch[c] + 1).unwrap();
        if groups {
            write!(out, "{sep}Group{}", cells.group[c] + 1).unwrap();
        }
        writeln!(out, "{sep}{}", cells.exp_lib_size[c]).unwrap();
    }
    out
}

/// Per-gene truth table: `Gene`, `BaseGeneMean`, `OutlierFactor`,
/// `GeneMean`, `BatchFacBatch<b>`, `DEFacGroup<k>`, and `BCVChiFac`, which
/// splatter does not store but is part of the truth.
///
/// ### Params
///
/// * `genes` - Gene truth
/// * `sep` - Field separator
///
/// ### Returns
///
/// The table as bytes, with a header line.
pub fn gene_table(genes: &GeneTruth, sep: char) -> Vec<u8> {
    let mut out = Vec::with_capacity(genes.n_genes() * 80);
    write!(out, "Gene{sep}BaseGeneMean{sep}OutlierFactor{sep}GeneMean").unwrap();
    for b in 0..genes.batch_fac.len() {
        write!(out, "{sep}BatchFacBatch{}", b + 1).unwrap();
    }
    for k in 0..genes.de_fac.len() {
        write!(out, "{sep}DEFacGroup{}", k + 1).unwrap();
    }
    writeln!(out, "{sep}BCVChiFac").unwrap();
    for g in 0..genes.n_genes() {
        write!(
            out,
            "Gene{}{sep}{}{sep}{}{sep}{}",
            g + 1,
            genes.base_gene_mean[g],
            genes.outlier_factor[g],
            genes.gene_mean[g]
        )
        .unwrap();
        for v in genes.batch_fac.iter().chain(&genes.de_fac) {
            write!(out, "{sep}{}", v[g]).unwrap();
        }
        writeln!(out, "{sep}{}", genes.bcv_chi_fac[g]).unwrap();
    }
    out
}
