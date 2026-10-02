//! 10x Genomics (Cell Ranger v3) mtx directory under `<dir>/tenx/`:
//! `matrix.mtx.gz` (genes x cells, 1-based, sorted by cell),
//! `barcodes.tsv.gz` and `features.tsv.gz`.

use std::io::Write;
use std::path::Path;

use super::{MTX_BANNER, MtxBody, Sink, gzip, write_gz};
use crate::cells::CellChunk;
use crate::errors::SplatErrors;
use crate::params::SplatParams;

/// Writer for the 10x mtx layout.
pub struct TenxMtxSink {
    /// Streaming `matrix.mtx.gz` body, a sequence of gzip members
    body: MtxBody,
    /// Number of cells
    n_cells: usize,
    /// Number of genes
    n_genes: usize,
}

impl TenxMtxSink {
    /// Write barcodes and features and open the matrix.
    ///
    /// ### Params
    ///
    /// * `dir` - Output directory; files go into `dir/tenx`
    /// * `params` - Resolved parameters
    ///
    /// ### Returns
    ///
    /// The sink.
    pub fn new(dir: &Path, params: &SplatParams) -> Result<Self, SplatErrors> {
        let dir = dir.join("tenx");
        std::fs::create_dir_all(&dir)?;

        let mut b = Vec::new();
        for c in 1..=params.n_cells() {
            writeln!(b, "Cell{c}")?;
        }
        write_gz(&dir.join("barcodes.tsv.gz"), &b)?;

        let mut f = Vec::new();
        for g in 1..=params.n_genes {
            writeln!(f, "Gene{g}\tGene{g}\tGene Expression")?;
        }
        write_gz(&dir.join("features.tsv.gz"), &f)?;

        Ok(Self {
            body: MtxBody::new(dir.join("matrix.mtx.gz"))?,
            n_cells: params.n_cells(),
            n_genes: params.n_genes,
        })
    }
}

/// MatrixMarket lines `gene cell count`, 1-based.
///
/// ### Params
///
/// * `chunk` - Counts in cell-major CSR
///
/// ### Returns
///
/// The encoded, uncompressed lines.
pub fn encode(chunk: &CellChunk) -> Vec<u8> {
    let mut out = Vec::with_capacity(chunk.counts.len() * 16);
    for (i, w) in chunk.indptr.windows(2).enumerate() {
        let c = chunk.start + i + 1;
        for j in w[0]..w[1] {
            writeln!(out, "{} {c} {}", chunk.indices[j] + 1, chunk.counts[j]).unwrap();
        }
    }
    out
}

impl Sink for TenxMtxSink {
    fn write_chunk(&mut self, chunk: &CellChunk, encoded: &[u8]) -> Result<(), SplatErrors> {
        self.body.append(encoded, chunk.counts.len())
    }

    fn finish(self: Box<Self>) -> Result<(), SplatErrors> {
        let header = format!(
            "{MTX_BANNER}{} {} {}\n",
            self.n_genes, self.n_cells, self.body.nnz
        );
        let header = gzip(header.as_bytes())?;
        self.body.finish(&header)
    }
}
