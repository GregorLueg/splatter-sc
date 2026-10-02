//! Output layouts. Each layout is a [`Sink`] that receives chunks in cell
//! order on one writer thread. Text encoding of a chunk happens beforehand
//! on the worker threads via [`encode`], so the writer only appends bytes.

mod h5;
pub mod h5ad;
mod mtx;
pub mod parse;
pub mod tables;
pub mod tenx_h5;
pub mod tenx_mtx;

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use flate2::Compression;
use flate2::write::GzEncoder;

use crate::Simulation;
use crate::cells::CellChunk;
use crate::errors::SplatErrors;
use crate::params::Layout;

/// Size of the `BufWriter` in front of every output file.
const WRITE_BUF: usize = 1 << 20;

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
        Layout::Parse => Ok(mtx::encode_lines(chunk, true)),
        Layout::TenxMtx => gzip(&mtx::encode_lines(chunk, false)),
        Layout::H5ad | Layout::TenxH5 => Ok(Vec::new()),
    }
}

/// Open the sink of every requested layout.
///
/// ### Params
///
/// * `sim` - The simulation
///
/// ### Returns
///
/// One sink per layout, in `output.layouts` order.
pub fn open_sinks(sim: &Simulation) -> Result<Vec<Box<dyn Sink>>, SplatErrors> {
    let (p, dir) = (&sim.params, &sim.params.output.dir);
    let deflate = p.output.h5_compression;
    p.output
        .layouts
        .iter()
        .map(|l| -> Result<Box<dyn Sink>, SplatErrors> {
            Ok(match l {
                Layout::Parse => Box::new(parse::ParseSink::new(dir, p, &sim.cells)?),
                Layout::TenxMtx => Box::new(tenx_mtx::TenxMtxSink::new(dir, p)?),
                Layout::H5ad => Box::new(h5ad::H5adSink::new(
                    dir, p, &sim.cells, &sim.genes, deflate,
                )?),
                Layout::TenxH5 => Box::new(tenx_h5::TenxH5Sink::new(dir, p, deflate)?),
            })
        })
        .collect()
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
