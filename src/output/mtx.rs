//! MatrixMarket helpers shared by the Parse and 10x mtx layouts.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;

use super::create;
use crate::cells::CellChunk;
use crate::errors::SplatErrors;

////////////
// Consts //
////////////

/// MatrixMarket banner shared by both mtx layouts.
pub(crate) const MTX_BANNER: &str = "%%MatrixMarket matrix coordinate integer general\n";

/////////////
// Helpers //
/////////////

/// MatrixMarket lines for a chunk, 1-based.
///
/// ### Params
///
/// * `chunk` - Counts in cell-major CSR
/// * `cells_first` - `cell gene count` (Parse) if true, `gene cell count`
///   (10x) otherwise
///
/// ### Returns
///
/// The encoded, uncompressed lines.
pub(crate) fn encode_lines(chunk: &CellChunk, cells_first: bool) -> Vec<u8> {
    let mut out = Vec::with_capacity(chunk.counts.len() * 16);
    for (i, w) in chunk.indptr.windows(2).enumerate() {
        let c = chunk.start + i + 1;
        for j in w[0]..w[1] {
            let (g, x) = (chunk.indices[j] + 1, chunk.counts[j]);
            if cells_first {
                writeln!(out, "{c} {g} {x}").unwrap();
            } else {
                writeln!(out, "{g} {c} {x}").unwrap();
            }
        }
    }
    out
}

/////////////
// MtxBody //
/////////////

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
