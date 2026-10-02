//! Output layouts. Each layout is a [`Sink`] that receives chunks in cell
//! order on one writer thread. Text encoding of a chunk happens beforehand
//! on the worker threads via [`encode`], so the writer only appends bytes.

mod h5;
pub mod h5ad;
pub mod parse;
pub mod tables;
pub mod tenx_h5;
pub mod tenx_mtx;

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use flate2::Compression;
use flate2::write::GzEncoder;

use crate::cells::CellChunk;
use crate::errors::SplatErrors;
use crate::params::Layout;

/// Size of the `BufWriter` in front of every output file.
const WRITE_BUF: usize = 1 << 20;

/// MatrixMarket banner shared by both mtx layouts.
pub(crate) const MTX_BANNER: &str = "%%MatrixMarket matrix coordinate integer general\n";

/// A destination for simulated chunks.
pub trait Sink: Send {
    /// Append one chunk. Chunks arrive in cell order.
    ///
    /// ### Params
    ///
    /// * `chunk` - Counts in cell-major CSR
    /// * `encoded` - This layout's output of [`encode`] for the chunk
    ///
    /// ### Returns
    ///
    /// `Ok(())` or an IO error.
    fn write_chunk(&mut self, chunk: &CellChunk, encoded: &[u8]) -> Result<(), SplatErrors>;

    /// Flush and finalise all files.
    ///
    /// ### Returns
    ///
    /// `Ok(())` or an IO error.
    fn finish(self: Box<Self>) -> Result<(), SplatErrors>;
}

/// Encode a chunk for a layout on a worker thread.
///
/// ### Params
///
/// * `layout` - Target layout
/// * `chunk` - Counts in cell-major CSR
///
/// ### Returns
///
/// The bytes the layout's sink appends: plain MatrixMarket lines for Parse,
/// one gzip member for 10x mtx, nothing for the HDF5 layouts, which take the
/// CSR directly.
pub fn encode(layout: Layout, chunk: &CellChunk) -> Result<Vec<u8>, SplatErrors> {
    match layout {
        Layout::Parse => Ok(parse::encode(chunk)),
        Layout::TenxMtx => gzip(&tenx_mtx::encode(chunk)),
        Layout::H5ad | Layout::TenxH5 => Ok(Vec::new()),
    }
}

/// Compress bytes into one gzip member. Members concatenate into a valid
/// gzip stream, which is what lets chunks compress in parallel.
///
/// ### Params
///
/// * `bytes` - Uncompressed bytes
///
/// ### Returns
///
/// One complete gzip member.
pub(crate) fn gzip(bytes: &[u8]) -> Result<Vec<u8>, SplatErrors> {
    let mut enc = GzEncoder::new(Vec::with_capacity(bytes.len() / 3), Compression::default());
    enc.write_all(bytes)?;
    Ok(enc.finish()?)
}

/// Create a buffered file.
///
/// ### Params
///
/// * `path` - File to create or truncate
///
/// ### Returns
///
/// The writer.
pub(crate) fn create(path: &Path) -> Result<BufWriter<File>, SplatErrors> {
    Ok(BufWriter::with_capacity(WRITE_BUF, File::create(path)?))
}

/// Write bytes to a new file as one gzip member.
///
/// ### Params
///
/// * `path` - File to create
/// * `bytes` - Uncompressed content
///
/// ### Returns
///
/// `Ok(())` or an IO error.
pub(crate) fn write_gz(path: &Path, bytes: &[u8]) -> Result<(), SplatErrors> {
    let mut f = create(path)?;
    f.write_all(&gzip(bytes)?)?;
    f.flush()?;
    Ok(())
}

/// The body of a MatrixMarket file whose header is only known at the end.
///
/// The header carries the nonzero count, so the body streams into a
/// temporary file next to the target, and [`MtxBody::finish`] writes the
/// header and copies the body behind it.
pub(crate) struct MtxBody {
    /// Final file path
    path: PathBuf,
    /// Temporary body path
    tmp: PathBuf,
    /// Writer on the temporary body
    body: BufWriter<File>,
    /// Nonzeros written so far
    pub nnz: u64,
}

impl MtxBody {
    /// Open the temporary body for `path`.
    ///
    /// ### Params
    ///
    /// * `path` - Final file path
    ///
    /// ### Returns
    ///
    /// The body writer.
    pub fn new(path: PathBuf) -> Result<Self, SplatErrors> {
        let mut tmp = path.clone().into_os_string();
        tmp.push(".body.tmp");
        let tmp = PathBuf::from(tmp);
        Ok(Self {
            body: create(&tmp)?,
            path,
            tmp,
            nnz: 0,
        })
    }

    /// Append encoded entries.
    ///
    /// ### Params
    ///
    /// * `bytes` - Encoded entries
    /// * `nnz` - Number of entries in `bytes`
    ///
    /// ### Returns
    ///
    /// `Ok(())` or an IO error.
    pub fn append(&mut self, bytes: &[u8], nnz: usize) -> Result<(), SplatErrors> {
        self.body.write_all(bytes)?;
        self.nnz += nnz as u64;
        Ok(())
    }

    /// Write `header` then the body to the final path, and remove the body.
    ///
    /// ### Params
    ///
    /// * `header` - Already encoded header bytes
    ///
    /// ### Returns
    ///
    /// `Ok(())` or an IO error.
    pub fn finish(self, header: &[u8]) -> Result<(), SplatErrors> {
        let mut body = self.body.into_inner().map_err(|e| e.into_error())?;
        body.flush()?;
        drop(body);
        let mut out = create(&self.path)?;
        out.write_all(header)?;
        std::io::copy(&mut File::open(&self.tmp)?, &mut out)?;
        out.flush()?;
        std::fs::remove_file(&self.tmp)?;
        Ok(())
    }
}
