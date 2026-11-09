//! Cell Ranger v3 style `matrix.h5`: genes x cells CSC under `/matrix`
//! (`indptr` over cells, `indices` = gene rows), which is exactly our
//! cell-major CSR. Strings are NUL-padded fixed-length ASCII as wide as
//! their longest entry, as Cell Ranger writes them; h5py hands those back
//! as bytes that scanpy decodes cleanly.
//! Filters, chunks and root attributes follow Cell Ranger's h5py writer as
//! seen in Space Ranger 2.1 files: `data`, `indices`, `indptr` and `shape`
//! resizable with fixed chunks and shuffle plus gzip, strings gzipped over
//! h5py's guessed chunk without shuffle, `_all_tag_keys` contiguous.

use std::path::Path;

use hdf5::types::TypeDescriptor;
use hdf5::{File, Group, Location};
use hdf5_sys::h5a::H5Awrite;
use hdf5_sys::h5d::H5Dwrite;
use hdf5_sys::h5p::H5P_DEFAULT;
use hdf5_sys::h5s::H5S_ALL;

use super::Sink;
use super::h5::{ChunkWriter, H5Chunks, attr_str, guess_chunk, write_1d_resizable};
use crate::cells::CellChunk;
use crate::errors::{SplatErrors, invalid};
use crate::output::tables::{cell_name, gene_name};
use crate::params::SplatParams;

////////////
// Consts //
////////////

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

/// Turn a negative HDF5 status into an error.
///
/// ### Params
///
/// * `status` - Return value of an HDF5 call
/// * `what` - Dataset or attribute name, for the message
///
/// ### Returns
///
/// `Ok(())` or an HDF5 error.
fn check(status: i32, what: &str) -> Result<(), SplatErrors> {
    if status < 0 {
        return Err(hdf5::Error::from(format!("writing {what} failed")).into());
    }
    Ok(())
}

///////////
// Names //
///////////

/// Strings packed as NUL-padded fixed-width ASCII, the width being the
/// longest string. The width is only known at run time, so the bytes go
/// through `H5Dwrite` / `H5Awrite` with the stored type as memory type.
struct Names {
    /// Bytes per string
    width: usize,
    /// Number of strings
    len: usize,
    /// `len * width` bytes
    bytes: Vec<u8>,
}

impl Names {
    /// Pack strings.
    ///
    /// ### Params
    ///
    /// * `values` - ASCII strings
    ///
    /// ### Returns
    ///
    /// The packed strings.
    fn new(values: impl Iterator<Item = String>) -> Result<Self, SplatErrors> {
        let v: Vec<String> = values.collect();
        let width = v.iter().map(String::len).max().unwrap_or(0).max(1);
        let mut bytes = vec![0u8; width * v.len()];
        for (s, dst) in v.iter().zip(bytes.chunks_exact_mut(width)) {
            if !s.is_ascii() {
                return Err(invalid("name", format!("{s}: not ASCII")));
            }
            dst[..s.len()].copy_from_slice(s.as_bytes());
        }
        Ok(Self {
            width,
            len: v.len(),
            bytes,
        })
    }

    /// Write as a 1-D dataset: contiguous without `deflate`, else gzip over
    /// h5py's guessed chunk.
    ///
    /// ### Params
    ///
    /// * `group` - Parent group
    /// * `name` - Dataset name
    /// * `deflate` - Optional gzip level
    ///
    /// ### Returns
    ///
    /// `Ok(())` or an HDF5 error.
    fn write(&self, group: &Group, name: &str, deflate: Option<u8>) -> Result<(), SplatErrors> {
        let mut b = group
            .new_dataset_builder()
            .empty_as(TypeDescriptor::FixedAscii(self.width))
            .shape(self.len);
        if let Some(level) = deflate {
            b = b.chunk(guess_chunk(self.len, self.width)).deflate(level);
        }
        let ds = b.create(name)?;
        let dt = ds.dtype()?;
        let status = hdf5::sync::sync(|| unsafe {
            H5Dwrite(
                ds.id(),
                dt.id(),
                H5S_ALL,
                H5S_ALL,
                H5P_DEFAULT,
                self.bytes.as_ptr().cast(),
            )
        });
        check(status, name)
    }

    /// Write as a 1-D attribute.
    ///
    /// ### Params
    ///
    /// * `loc` - Group or dataset
    /// * `name` - Attribute name
    ///
    /// ### Returns
    ///
    /// `Ok(())` or an HDF5 error.
    fn write_attr(&self, loc: &Location, name: &str) -> Result<(), SplatErrors> {
        let a = loc
            .new_attr_builder()
            .empty_as(TypeDescriptor::FixedAscii(self.width))
            .shape(self.len)
            .create(name)?;
        let dt = a.dtype()?;
        let status =
            hdf5::sync::sync(|| unsafe { H5Awrite(a.id(), dt.id(), self.bytes.as_ptr().cast()) });
        check(status, name)
    }
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
        Names::new(std::iter::once("sim".into()))?.write_attr(&file, "library_ids")?;
        file.new_attr::<i64>()
            .shape(1)
            .create("original_gem_groups")?
            .write_raw(&[1])?;

        let matrix = file.create_group("matrix")?;
        let (n_cells, n_genes) = (params.n_cells(), params.n_genes);

        Names::new((0..n_cells).map(cell_name))?.write(&matrix, "barcodes", deflate)?;
        let shape = [n_genes as i32, n_cells as i32];
        write_1d_resizable(&matrix, "shape", &shape, CHUNK, deflate, SHUFFLE)?;

        let features = matrix.create_group("features")?;
        Names::new(std::iter::once("genome".into()))?.write(&features, "_all_tag_keys", None)?;
        let ids = Names::new((0..n_genes).map(gene_name))?;
        ids.write(&features, "id", deflate)?;
        ids.write(&features, "name", deflate)?;
        Names::new(std::iter::repeat_n("Gene Expression".to_string(), n_genes))?.write(
            &features,
            "feature_type",
            deflate,
        )?;
        Names::new(std::iter::repeat_n(GENOME.to_string(), n_genes))?
            .write(&features, "genome", deflate)?;

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
