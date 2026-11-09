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

/// h5py's `CHUNK_BASE`, `CHUNK_MIN` and `CHUNK_MAX`, in bytes.
const CHUNK_BASE: f64 = 16384.0;
const CHUNK_MIN: f64 = 8192.0;
const CHUNK_MAX: f64 = 1048576.0;

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

/// h5py's `guess_chunk` for a 1-D dataset: the chunk length h5py, and so
/// anndata and Cell Ranger, pick when no chunk shape is given.
///
/// ### Params
///
/// * `n` - Dataset length; 0 counts as 1024, as in h5py
/// * `itemsize` - Bytes per element (8 for variable-length strings)
///
/// ### Returns
///
/// Elements per chunk.
pub(crate) fn guess_chunk(n: usize, itemsize: usize) -> usize {
    let n = if n == 0 { 1024 } else { n };
    let target = (CHUNK_BASE * 2f64.powf(((n * itemsize) as f64 / CHUNK_MAX).log10()))
        .clamp(CHUNK_MIN, CHUNK_MAX);
    let mut c = n;
    loop {
        let bytes = (c * itemsize) as f64;
        if ((bytes < target || (bytes - target).abs() / target < 0.5) && bytes < CHUNK_MAX)
            || c == 1
        {
            return c;
        }
        c = c.div_ceil(2);
    }
}

/// Write a 1-D dataset in one go. Contiguous without `deflate`; with it,
/// gzip over h5py's guessed chunk, as h5py writes `compression="gzip"`.
///
/// ### Params
///
/// * `group` - Parent group
/// * `name` - Dataset name
/// * `data` - Values
/// * `deflate` - Optional gzip level
///
/// ### Returns
///
/// The dataset.
pub(crate) fn write_1d<T: H5Type>(
    group: &Group,
    name: &str,
    data: &[T],
    deflate: Option<u8>,
) -> Result<Dataset, SplatErrors> {
    let mut b = group.new_dataset_builder().with_data(data);
    if let Some(level) = deflate {
        b = b
            .chunk(guess_chunk(data.len(), size_of::<T>()))
            .deflate(level);
    }
    Ok(b.create(name)?)
}

/// Write a 1-D dataset in one go as resizable (`maxshape=(None,)`) and
/// chunked, as anndata writes `X/indptr` and Cell Ranger `indptr`/`shape`.
///
/// ### Params
///
/// * `group` - Parent group
/// * `name` - Dataset name
/// * `data` - Values
/// * `chunk` - Elements per chunk
/// * `deflate` - Optional gzip level
/// * `shuffle` - Byte-shuffle before gzip; ignored without `deflate`
///
/// ### Returns
///
/// `Ok(())` or an HDF5 error.
pub(crate) fn write_1d_resizable<T: H5Type>(
    group: &Group,
    name: &str,
    data: &[T],
    chunk: usize,
    deflate: Option<u8>,
    shuffle: bool,
) -> Result<(), SplatErrors> {
    let mut b = group.new_dataset::<T>().shape(data.len()..).chunk(chunk);
    if let Some(level) = deflate {
        if shuffle {
            b = b.shuffle();
        }
        b = b.deflate(level);
    }
    b.create(name)?.write(data)?;
    Ok(())
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
    /// Elements per HDF5 chunk
    chunk: usize,
}

impl ChunkStream {
    /// Empty stream for element type `T`.
    ///
    /// ### Params
    ///
    /// * `deflate` - Optional zlib level, as set on the dataset
    /// * `shuffle` - Shuffle flag, as set on the dataset
    /// * `chunk` - Elements per chunk, as set on the dataset
    ///
    /// ### Returns
    ///
    /// The stream.
    fn new<T: ChunkElem>(deflate: Option<u8>, shuffle: bool, chunk: usize) -> Self {
        Self {
            elem: size_of::<T>(),
            buf: Vec::new(),
            len: 0,
            deflate,
            shuffle,
            chunk,
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
        let cb = self.chunk * self.elem;
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
    /// * `nnz_hint` - Expected total nonzeros, for the h5ad chunk length
    ///
    /// ### Returns
    ///
    /// The encoder, or `None` for the mtx layouts.
    pub fn new(layout: Layout, output: &OutputParams, nnz_hint: usize) -> Option<Self> {
        let (kind, data, indices) = match layout {
            Layout::H5ad => {
                let (d, c) = (output.h5ad_compression, h5ad::x_chunk(nnz_hint));
                (
                    H5Kind::H5ad,
                    ChunkStream::new::<h5ad::Data>(d, h5ad::SHUFFLE, c),
                    ChunkStream::new::<h5ad::Indices>(d, h5ad::SHUFFLE, c),
                )
            }
            Layout::TenxH5 => {
                let (d, c) = (output.tenx_h5_compression, tenx_h5::CHUNK);
                (
                    H5Kind::TenxH5,
                    ChunkStream::new::<tenx_h5::Data>(d, tenx_h5::SHUFFLE, c),
                    ChunkStream::new::<tenx_h5::Indices>(d, tenx_h5::SHUFFLE, c),
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
    /// Elements per chunk
    chunk: usize,
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
    /// * `chunk` - Elements per chunk
    ///
    /// ### Returns
    ///
    /// The writer.
    pub fn new<T: ChunkElem>(
        group: &Group,
        name: &str,
        deflate: Option<u8>,
        shuffle: bool,
        chunk: usize,
    ) -> Result<Self, SplatErrors> {
        let mut b = group.new_dataset::<T>().shape(0..).chunk(chunk);
        if let Some(level) = deflate {
            if shuffle {
                b = b.shuffle();
            }
            b = b.deflate(level);
        }
        Ok(Self {
            ds: b.create(name)?,
            n_chunks: 0,
            chunk,
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
            let offset = self.n_chunks * self.chunk;
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

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_guess_chunk_matches_h5py() {
        // (n, itemsize, chunk) from h5py 3.16 `filters.guess_chunk`.
        let cases = [
            (0, 4, 1024),
            (1, 8, 1),
            (24, 8, 24),
            (10000, 1, 10000),
            (10001, 4, 2501),
            (10000, 8, 1250),
            (19273, 8, 1205),
            (100001, 8, 1563),
            (1000001, 8, 3907),
            (6000000, 4, 11719),
            (10685844, 4, 10436),
            (107279791, 4, 26192),
            (1072778635, 4, 65478),
            (119289764, 4, 29124),
            (10000000000, 4, 76294),
        ];
        for (n, s, c) in cases {
            assert_eq!(guess_chunk(n, s), c, "n = {n}, itemsize = {s}");
        }
    }
}
