//! Cell Ranger v3 style `matrix.h5`: genes x cells CSC under `/matrix`
//! (`indptr` over cells, `indices` = gene rows), which is exactly our
//! cell-major CSR. Strings are fixed-length ASCII as Cell Ranger writes
//! them; h5py hands those back as bytes that scanpy decodes cleanly.
//! Filters, chunks and root attributes follow Cell Ranger's h5py writer as
//! seen in Space Ranger 2.1 files: `data`, `indices`, `indptr` and `shape`
//! resizable with fixed chunks and shuffle plus gzip, strings gzipped over
//! h5py's guessed chunk without shuffle, `_all_tag_keys` contiguous.

use std::path::Path;

use hdf5::types::FixedAscii;
use hdf5::{File, Group};

use super::Sink;
use super::h5::{ChunkWriter, H5Chunks, attr_str, write_1d, write_1d_resizable};
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

/// Element type of `matrix/data`.
pub(crate) type Data = i32;

/// Element type of `matrix/indices`.
pub(crate) type Indices = i64;

/// Byte-shuffle before gzip, as Cell Ranger writes the matrix.
pub(crate) const SHUFFLE: bool = true;

/// Elements per chunk of `data`, `indices`, `indptr` and `shape`
/// (Cell Ranger's `H5_CHUNK_SIZE`).
pub(crate) const CHUNK: usize = 80000;

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
    data: ChunkWriter,
    /// `matrix/indices`
    indices: ChunkWriter,
    /// `matrix/indptr`, accumulated
    indptr: Vec<i64>,
    /// Optional gzip level
    deflate: Option<u8>,
}

impl TenxH5Sink {
    /// Write barcodes, features and shape, and open the matrix.
    ///
    /// ### Params
    ///
    /// * `dir` - Output directory
    /// * `params` - Resolved parameters
    /// * `deflate` - Optional gzip level for every dataset but `_all_tag_keys`
    ///
    /// ### Returns
    ///
    /// The sink.
    pub fn new(dir: &Path, params: &SplatParams, deflate: Option<u8>) -> Result<Self, SplatErrors> {
        let file = File::create(dir.join("matrix.h5"))?;
        attr_str(&file, "filetype", "matrix")?;
        file.new_attr::<i64>()
            .shape(())
            .create("version")?
            .write_scalar(&2)?;
        attr_str(
            &file,
            "software_version",
            concat!("splatter-sc-", env!("CARGO_PKG_VERSION")),
        )?;
        attr_str(&file, "chemistry_description", "Splat simulation")?;
        file.new_attr::<Name>()
            .shape(1)
            .create("library_ids")?
            .write_raw(&names(std::iter::once("sim".into()))?)?;
        file.new_attr::<i64>()
            .shape(1)
            .create("original_gem_groups")?
            .write_raw(&[1])?;

        let matrix = file.create_group("matrix")?;
        let (n_cells, n_genes) = (params.n_cells(), params.n_genes);

        let barcodes = names((0..n_cells).map(cell_name))?;
        write_1d(&matrix, "barcodes", &barcodes, deflate)?;
        let shape = [n_genes as i32, n_cells as i32];
        write_1d_resizable(&matrix, "shape", &shape, CHUNK, deflate, SHUFFLE)?;

        let features = matrix.create_group("features")?;
        let tag_keys = names(std::iter::once("genome".into()))?;
        write_1d(&features, "_all_tag_keys", &tag_keys, None)?;
        let ids = names((0..n_genes).map(gene_name))?;
        write_1d(&features, "id", &ids, deflate)?;
        write_1d(&features, "name", &ids, deflate)?;
        let ft = names(std::iter::repeat_n("Gene Expression".to_string(), n_genes))?;
        write_1d(&features, "feature_type", &ft, deflate)?;
        let genome = names(std::iter::repeat_n(GENOME.to_string(), n_genes))?;
        write_1d(&features, "genome", &genome, deflate)?;

        let mut indptr = Vec::with_capacity(n_cells + 1);
        indptr.push(0);
        Ok(Self {
            data: ChunkWriter::new::<Data>(&matrix, "data", deflate, SHUFFLE, CHUNK)?,
            indices: ChunkWriter::new::<Indices>(&matrix, "indices", deflate, SHUFFLE, CHUNK)?,
            matrix,
            indptr,
            deflate,
            _file: file,
        })
    }
}

impl Sink for TenxH5Sink {
    fn write_chunk(&mut self, chunk: &CellChunk, _encoded: &[u8]) -> Result<(), SplatErrors> {
        let base = *self.indptr.last().expect("indptr starts with 0");
        self.indptr
            .extend(chunk.indptr[1..].iter().map(|&p| base + p as i64));
        Ok(())
    }

    fn write_h5(&mut self, chunks: &H5Chunks) -> Result<(), SplatErrors> {
        self.data.write(&chunks.data, chunks.data_len)?;
        self.indices.write(&chunks.indices, chunks.indices_len)
    }

    fn finish(self: Box<Self>) -> Result<(), SplatErrors> {
        write_1d_resizable(
            &self.matrix,
            "indptr",
            &self.indptr,
            CHUNK,
            self.deflate,
            SHUFFLE,
        )
    }
}
