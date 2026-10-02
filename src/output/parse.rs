//! Parse Biosciences layout, as single-cell-benchmark reads it:
//! `DGE.mtx` (cells x genes, 1-based, sorted by cell then gene, no comment
//! lines), `cell_metadata.csv` (`bc_wells`, `sample` = batch, ...) and
//! `all_genes.csv` (`gene_id,gene_name,genome`).

use std::io::Write;
use std::path::Path;

use super::{MTX_BANNER, MtxBody, Sink, create};
use crate::cells::{CellChunk, CellMeta};
use crate::errors::SplatErrors;
use crate::output::tables::cell_table;
use crate::params::SplatParams;

/// Genome label in `all_genes.csv`.
const GENOME: &str = "sim";

/// Writer for the Parse layout.
pub struct ParseSink {
    /// Streaming `DGE.mtx` body
    body: MtxBody,
    /// Number of cells
    n_cells: usize,
    /// Number of genes
    n_genes: usize,
}

impl ParseSink {
    /// Write the metadata files and open the matrix.
    ///
    /// ### Params
    ///
    /// * `dir` - Output directory
    /// * `params` - Resolved parameters
    /// * `cells` - Cell metadata
    ///
    /// ### Returns
    ///
    /// The sink.
    pub fn new(dir: &Path, params: &SplatParams, cells: &CellMeta) -> Result<Self, SplatErrors> {
        let mut f = create(&dir.join("cell_metadata.csv"))?;
        f.write_all(&cell_table(params, cells, ',', "bc_wells", "sample"))?;
        f.flush()?;

        let mut f = create(&dir.join("all_genes.csv"))?;
        writeln!(f, "gene_id,gene_name,genome")?;
        for g in 1..=params.n_genes {
            writeln!(f, "Gene{g},Gene{g},{GENOME}")?;
        }
        f.flush()?;

        Ok(Self {
            body: MtxBody::new(dir.join("DGE.mtx"))?,
            n_cells: params.n_cells(),
            n_genes: params.n_genes,
        })
    }
}

/// MatrixMarket lines `cell gene count`, 1-based.
///
/// ### Params
///
/// * `chunk` - Counts in cell-major CSR
///
/// ### Returns
///
/// The encoded lines.
pub fn encode(chunk: &CellChunk) -> Vec<u8> {
    let mut out = Vec::with_capacity(chunk.counts.len() * 16);
    for (i, w) in chunk.indptr.windows(2).enumerate() {
        let c = chunk.start + i + 1;
        for j in w[0]..w[1] {
            writeln!(out, "{c} {} {}", chunk.indices[j] + 1, chunk.counts[j]).unwrap();
        }
    }
    out
}

impl Sink for ParseSink {
    fn write_chunk(&mut self, chunk: &CellChunk, encoded: &[u8]) -> Result<(), SplatErrors> {
        self.body.append(encoded, chunk.counts.len())
    }

    fn finish(self: Box<Self>) -> Result<(), SplatErrors> {
        let header = format!(
            "{MTX_BANNER}{} {} {}\n",
            self.n_cells, self.n_genes, self.body.nnz
        );
        self.body.finish(header.as_bytes())
    }
}
