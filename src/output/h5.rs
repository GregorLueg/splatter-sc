//! HDF5 helpers shared by the h5ad and 10x h5 layouts.
//!
//! The streamed `data` / `indices` datasets are written chunk by chunk with
//! `H5Dwrite_chunk`. [`H5Encoder`] runs on the simulation pool: it cuts the
//! values into HDF5 chunks and applies the dataset's filters (shuffle, then
//! zlib) in parallel, so the single writer thread only stores bytes.

use std::io::Write;

use flate2::Compression;
use flate2::write::ZlibEncoder;
use hdf5::types::VarLenUnicode;
use hdf5::{Dataset, Group, H5Type, Location};
use hdf5_sys::h5::hsize_t;
use hdf5_sys::h5d::H5Dwrite_chunk;
use hdf5_sys::h5p::H5P_DEFAULT;
use rayon::prelude::*;

use super::{h5ad, tenx_h5};
use crate::cells::CellChunk;
use crate::errors::SplatErrors;
use crate::params::{Layout, OutputParams};

////////////
// Consts //
////////////

/// Elements per HDF5 chunk of the streamed `data` / `indices` datasets.
/// 2^18 elements is 1 MB of `f32`, in the range anndata and Cell Ranger
/// files use; large enough that chunk indexing stays cheap at 1e10 entries.
const H5_CHUNK: usize = 1 << 18;

/////////////
// Helpers //
/////////////

/// A variable-length UTF-8 string, as h5py and anndata write them.
///
/// ### Params
///
/// * `s` - String without NUL bytes
///
/// ### Returns
///
/// The HDF5 string.
pub(crate) fn vlu(s: &str) -> VarLenUnicode {
    s.parse().expect("generated names contain no NUL byte")
}

/// Write a scalar string attribute.
///
/// ### Params
///
/// * `loc` - Group or dataset
/// * `name` - Attribute name
/// * `value` - Attribute value
///
/// ### Returns
///
/// `Ok(())` or an HDF5 error.
pub(crate) fn attr_str(loc: &Location, name: &str, value: &str) -> Result<(), SplatErrors> {
    loc.new_attr::<VarLenUnicode>()
        .shape(())
        .create(name)?
        .write_scalar(&vlu(value))?;
    Ok(())
}

/// Write anndata's `encoding-type` and `encoding-version` attributes.
///
/// ### Params
///
/// * `loc` - Group or dataset
/// * `kind` - Encoding type, e.g. `csr_matrix`
/// * `version` - Encoding version, e.g. `0.1.0`
///
/// ### Returns
///
/// `Ok(())` or an HDF5 error.
pub(crate) fn encoding(loc: &Location, kind: &str, version: &str) -> Result<(), SplatErrors> {
    attr_str(loc, "encoding-type", kind)?;
    attr_str(loc, "encoding-version", version)
}

/// Write a 1-D dataset in one go.
///
/// ### Params
///
/// * `group` - Parent group
/// * `name` - Dataset name
/// * `data` - Values
///
/// ### Returns
///
/// The dataset.
pub(crate) fn write_1d<T: H5Type>(
    group: &Group,
    name: &str,
    data: &[T],
) -> Result<Dataset, SplatErrors> {
    Ok(group.new_dataset_builder().with_data(data).create(name)?)
}

//////////////
// Appender //
//////////////

/// A numeric element of a streamed dataset.
pub(crate) trait ChunkElem: H5Type + Copy {
    /// Append the value's native-endian bytes, which is how HDF5 stores a
    /// dataset created with the native type.
    ///
    /// ### Params
    ///
    /// * `out` - Byte buffer to extend
    fn push_ne(self, out: &mut Vec<u8>);
}

macro_rules! chunk_elem {
    ($($t:ty),*) => {
        $(impl ChunkElem for $t {
            #[inline]
            fn push_ne(self, out: &mut Vec<u8>) {
                out.extend_from_slice(&self.to_ne_bytes());
            }
        })*
    };
}

chunk_elem!(f32, i32, i64);

/// Run one chunk through the dataset's filter pipeline, as HDF5 would on a
/// regular write: byte-shuffle (if set), then zlib. HDF5's deflate filter
/// stores the zlib format, not gzip.
///
/// ### Params
///
/// * `raw` - Native bytes of exactly one chunk, zero-padded if partial
/// * `elem` - Bytes per element
/// * `deflate` - Optional zlib level; without it no filter runs
/// * `shuffle` - Whether the pipeline has the shuffle filter
///
/// ### Returns
///
/// The bytes to store for the chunk.
fn filter_chunk(
    raw: &[u8],
    elem: usize,
    deflate: Option<u8>,
    shuffle: bool,
) -> Result<Vec<u8>, SplatErrors> {
    let Some(level) = deflate else {
        return Ok(raw.to_vec());
    };
    let shuffled;
    let bytes = if shuffle && elem > 1 {
        // Byte j of element i goes to j * n + i: all first bytes, then all
        // second bytes, and so on.
        let n = raw.len() / elem;
        let mut out = vec![0u8; raw.len()];
        for (i, e) in raw.chunks_exact(elem).enumerate() {
            for (j, &b) in e.iter().enumerate() {
                out[j * n + i] = b;
            }
        }
        shuffled = out;
        &shuffled[..]
    } else {
        raw
    };
    let mut enc = ZlibEncoder::new(
        Vec::with_capacity(bytes.len() / 4),
        Compression::new(level as u32),
    );
    enc.write_all(bytes)?;
    Ok(enc.finish()?)
}

/////////////////
// ChunkStream //
/////////////////

/// Producer-side buffer of one streamed dataset: collects values and cuts
/// them into filtered HDF5 chunks.
struct ChunkStream {
    /// Bytes per element
    elem: usize,
    /// Pending native bytes; less than one chunk after each `take`
    buf: Vec<u8>,
    /// Elements pushed so far
    len: usize,
    /// Optional zlib level
    deflate: Option<u8>,
    /// Whether to byte-shuffle before zlib
    shuffle: bool,
}

impl ChunkStream {
    /// Empty stream for element type `T`.
    ///
    /// ### Params
    ///
    /// * `deflate` - Optional zlib level, as set on the dataset
    /// * `shuffle` - Shuffle flag, as set on the dataset
    ///
    /// ### Returns
    ///
    /// The stream.
    fn new<T: ChunkElem>(deflate: Option<u8>, shuffle: bool) -> Self {
        Self {
            elem: size_of::<T>(),
            buf: Vec::new(),
            len: 0,
            deflate,
            shuffle,
        }
    }

    /// Append values.
    ///
    /// ### Params
    ///
    /// * `values` - Values of the stream's element type
    fn push<T: ChunkElem>(&mut self, values: impl Iterator<Item = T>) {
        debug_assert_eq!(size_of::<T>(), self.elem);
        let before = self.buf.len();
        for v in values {
            v.push_ne(&mut self.buf);
        }
        self.len += (self.buf.len() - before) / self.elem;
    }

    /// Filter every complete chunk in parallel and drop it from the buffer.
    ///
    /// ### Params
    ///
    /// * `tail` - Also emit the final partial chunk, zero-padded to full
    ///   size as HDF5 stores edge chunks
    ///
    /// ### Returns
    ///
    /// The filtered chunks in order, and the dataset length once they are
    /// written.
    fn take(&mut self, tail: bool) -> Result<(Vec<Vec<u8>>, usize), SplatErrors> {
        let cb = H5_CHUNK * self.elem;
        let full = self.buf.len() / cb * cb;
        if tail && self.buf.len() > full {
            self.buf.resize(full + cb, 0);
        }
        let n = if tail { self.buf.len() } else { full };
        let (elem, deflate, shuffle) = (self.elem, self.deflate, self.shuffle);
        let chunks = self.buf[..n]
            .par_chunks_exact(cb)
            .map(|c| filter_chunk(c, elem, deflate, shuffle))
            .collect::<Result<Vec<_>, _>>()?;
        self.buf.drain(..n);
        Ok((chunks, self.len - self.buf.len() / elem))
    }
}

//////////////
// H5Chunks //
//////////////

/// Filtered chunks of one HDF5 layout's `data` and `indices`, in order.
pub struct H5Chunks {
    /// Chunks of `data`
    pub(crate) data: Vec<Vec<u8>>,
    /// Length of `data` once these chunks are written
    pub(crate) data_len: usize,
    /// Chunks of `indices`
    pub(crate) indices: Vec<Vec<u8>>,
    /// Length of `indices` once these chunks are written
    pub(crate) indices_len: usize,
}

///////////////
// H5Encoder //
///////////////

/// Which HDF5 layout an encoder serves.
#[derive(Clone, Copy)]
enum H5Kind {
    /// `DGE.h5ad`
    H5ad,
    /// `matrix.h5`
    TenxH5,
}

/// Producer-side encoder of one HDF5 layout. Element types and filters come
/// from the same items the sink creates its datasets with; a mismatch would
/// store bytes that readers decode wrongly.
pub struct H5Encoder {
    /// Layout served
    kind: H5Kind,
    /// `data` stream
    data: ChunkStream,
    /// `indices` stream
    indices: ChunkStream,
}

impl H5Encoder {
    /// Encoder for `layout`.
    ///
    /// ### Params
    ///
    /// * `layout` - Output layout
    /// * `output` - Output parameters, for the compression level
    ///
    /// ### Returns
    ///
    /// The encoder, or `None` for the mtx layouts.
    pub fn new(layout: Layout, output: &OutputParams) -> Option<Self> {
        let (kind, data, indices) = match layout {
            Layout::H5ad => {
                let d = output.h5ad_compression;
                (
                    H5Kind::H5ad,
                    ChunkStream::new::<h5ad::Data>(d, h5ad::SHUFFLE),
                    ChunkStream::new::<h5ad::Indices>(d, h5ad::SHUFFLE),
                )
            }
            Layout::TenxH5 => {
                let d = output.tenx_h5_compression;
                (
                    H5Kind::TenxH5,
                    ChunkStream::new::<tenx_h5::Data>(d, tenx_h5::SHUFFLE),
                    ChunkStream::new::<tenx_h5::Indices>(d, tenx_h5::SHUFFLE),
                )
            }
            Layout::Parse | Layout::TenxMtx => return None,
        };
        Some(Self {
            kind,
            data,
            indices,
        })
    }

    /// Append a window of cell chunks and filter the HDF5 chunks it
    /// completes.
    ///
    /// ### Params
    ///
    /// * `chunks` - Cell chunks in cell order
    ///
    /// ### Returns
    ///
    /// The complete chunks; a partial one stays buffered.
    pub fn encode<'a>(
        &mut self,
        chunks: impl Iterator<Item = &'a CellChunk>,
    ) -> Result<H5Chunks, SplatErrors> {
        for c in chunks {
            match self.kind {
                H5Kind::H5ad => {
                    self.data.push(c.counts.iter().map(|&x| x as h5ad::Data));
                    self.indices
                        .push(c.indices.iter().map(|&g| g as h5ad::Indices));
                }
                H5Kind::TenxH5 => {
                    self.data.push(c.counts.iter().map(|&x| x as tenx_h5::Data));
                    self.indices
                        .push(c.indices.iter().map(|&g| g as tenx_h5::Indices));
                }
            }
        }
        self.take(false)
    }

    /// Filter the buffered partial chunks after the last window.
    ///
    /// ### Returns
    ///
    /// The final chunks.
    pub fn finish(&mut self) -> Result<H5Chunks, SplatErrors> {
        self.take(true)
    }

    /// Take the ready chunks of both streams.
    ///
    /// ### Params
    ///
    /// * `tail` - Include the zero-padded partial chunks
    ///
    /// ### Returns
    ///
    /// The chunks.
    fn take(&mut self, tail: bool) -> Result<H5Chunks, SplatErrors> {
        let (data, data_len) = self.data.take(tail)?;
        let (indices, indices_len) = self.indices.take(tail)?;
        Ok(H5Chunks {
            data,
            data_len,
            indices,
            indices_len,
        })
    }
}

/////////////////
// ChunkWriter //
/////////////////

/// Writer-side handle of one streamed dataset: stores already filtered
/// chunks with `H5Dwrite_chunk`.
pub(crate) struct ChunkWriter {
    /// The resizable dataset
    ds: Dataset,
    /// Chunks written so far
    n_chunks: usize,
}

impl ChunkWriter {
    /// Create an empty, unlimited, chunked dataset. Its filter pipeline is
    /// what readers decode with, so it must match what [`H5Encoder`]
    /// applied.
    ///
    /// ### Params
    ///
    /// * `group` - Parent group
    /// * `name` - Dataset name
    /// * `deflate` - Optional gzip level
    /// * `shuffle` - Byte-shuffle before gzip; ignored without `deflate`
    ///
    /// ### Returns
    ///
    /// The writer.
    pub fn new<T: ChunkElem>(
        group: &Group,
        name: &str,
        deflate: Option<u8>,
        shuffle: bool,
    ) -> Result<Self, SplatErrors> {
        let mut b = group.new_dataset::<T>().shape(0..).chunk(H5_CHUNK);
        if let Some(level) = deflate {
            if shuffle {
                b = b.shuffle();
            }
            b = b.deflate(level);
        }
        Ok(Self {
            ds: b.create(name)?,
            n_chunks: 0,
        })
    }

    /// Grow the dataset to `len` and store `chunks` behind the ones already
    /// written.
    ///
    /// ### Params
    ///
    /// * `chunks` - Filtered chunks in order
    /// * `len` - Dataset length once they are written
    ///
    /// ### Returns
    ///
    /// `Ok(())` or an HDF5 error.
    pub fn write(&mut self, chunks: &[Vec<u8>], len: usize) -> Result<(), SplatErrors> {
        if chunks.is_empty() {
            return Ok(());
        }
        self.ds.resize(len)?;
        let id = self.ds.id();
        for c in chunks {
            let offset = self.n_chunks * H5_CHUNK;
            let start = [offset as hsize_t];
            // Filter mask 0: every filter in the pipeline was applied.
            let status = hdf5::sync::sync(|| unsafe {
                H5Dwrite_chunk(
                    id,
                    H5P_DEFAULT,
                    0,
                    start.as_ptr(),
                    c.len(),
                    c.as_ptr().cast(),
                )
            });
            if status < 0 {
                return Err(SplatErrors::ChunkWrite { offset });
            }
            self.n_chunks += 1;
        }
        Ok(())
    }
}
