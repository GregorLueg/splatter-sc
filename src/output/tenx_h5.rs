//! Cell Ranger v3 style `matrix.h5`: genes x cells CSC under `/matrix`
//! (`indptr` over cells, `indices` = gene rows), which is exactly our
//! cell-major CSR. Strings are fixed-length ASCII as Cell Ranger writes
//! them; h5py hands those back as bytes that scanpy decodes cleanly.

use std::path::Path;

use hdf5::types::FixedAscii;
use hdf5::{File, Group};

use super::Sink;
use super::h5::{Appender, write_1d};
use crate::cells::CellChunk;
use crate::errors::{SplatErrors, invalid};
use crate::output::tables::{cell_name, gene_name};
use crate::params::SplatParams;

////////////
// Consts //
////////////

/// Fixed string width; generated names are far shorter.
type Name = FixedAscii<31>;

/// Genome label in `features/genome`.
const GENOME: &str = "sim";

/////////////
// Helpers //
/////////////

/// Fixed-width ASCII names.
///
/// ### Params
///
/// * `values` - Strings of at most 31 ASCII bytes
///
/// ### Returns
///
/// The HDF5 strings.
fn names(values: impl Iterator<Item = String>) -> Result<Vec<Name>, SplatErrors> {
    values
        .map(|s| Name::from_ascii(&s).map_err(|e| invalid("name", format!("{s}: {e}"))))
        .collect()
}

////////////////
// TenxH5Sink //
////////////////

/// Writer for the 10x h5 layout.
pub struct TenxH5Sink {
    /// Open file; kept so it closes after the last write
    _file: File,
    /// `/matrix` group, for `indptr` at the end
    matrix: Group,
    /// `matrix/data`
    data: Appender<i32>,
    /// `matrix/indices`
    indices: Appender<i64>,
    /// `matrix/indptr`, accumulated
    indptr: Vec<i64>,
}

impl TenxH5Sink {
    /// Write barcodes, features and shape, and open the matrix.
    ///
    /// ### Params
    ///
    /// * `dir` - Output directory
    /// * `params` - Resolved parameters
    /// * `deflate` - Optional gzip level for `data` and `indices`
    ///
    /// ### Returns
    ///
    /// The sink.
    pub fn new(dir: &Path, params: &SplatParams, deflate: Option<u8>) -> Result<Self, SplatErrors> {
        let file = File::create(dir.join("matrix.h5"))?;
        let matrix = file.create_group("matrix")?;
        let (n_cells, n_genes) = (params.n_cells(), params.n_genes);

        write_1d(&matrix, "barcodes", &names((0..n_cells).map(cell_name))?)?;
        write_1d(&matrix, "shape", &[n_genes as i32, n_cells as i32])?;

        let features = matrix.create_group("features")?;
        write_1d(
            &features,
            "_all_tag_keys",
            &names(std::iter::once("genome".into()))?,
        )?;
        write_1d(&features, "id", &names((0..n_genes).map(gene_name))?)?;
        write_1d(&features, "name", &names((0..n_genes).map(gene_name))?)?;
        let ft = names(std::iter::repeat_n("Gene Expression".to_string(), n_genes))?;
        write_1d(&features, "feature_type", &ft)?;
        let genome = names(std::iter::repeat_n(GENOME.to_string(), n_genes))?;
        write_1d(&features, "genome", &genome)?;

        let mut indptr = Vec::with_capacity(n_cells + 1);
        indptr.push(0);
        Ok(Self {
            data: Appender::new(&matrix, "data", deflate)?,
            indices: Appender::new(&matrix, "indices", deflate)?,
            matrix,
            indptr,
            _file: file,
        })
    }
}

impl Sink for TenxH5Sink {
    fn write_chunk(&mut self, chunk: &CellChunk, _encoded: &[u8]) -> Result<(), SplatErrors> {
        let data: Vec<i32> = chunk.counts.iter().map(|&x| x as i32).collect();
        let indices: Vec<i64> = chunk.indices.iter().map(|&g| g as i64).collect();
        self.data.append(&data)?;
        self.indices.append(&indices)?;
        let base = *self.indptr.last().expect("indptr starts with 0");
        self.indptr
            .extend(chunk.indptr[1..].iter().map(|&p| base + p as i64));
        Ok(())
    }

    fn finish(self: Box<Self>) -> Result<(), SplatErrors> {
        write_1d(&self.matrix, "indptr", &self.indptr)?;
        Ok(())
    }
}
