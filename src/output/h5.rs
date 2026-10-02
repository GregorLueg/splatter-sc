//! HDF5 helpers shared by the h5ad and 10x h5 layouts.

use hdf5::types::VarLenUnicode;
use hdf5::{Dataset, Group, H5Type, Location};

use crate::errors::SplatErrors;

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

/// A 1-D dataset that grows as chunks of cells arrive.
pub(crate) struct Appender<T> {
    /// The resizable dataset
    ds: Dataset,
    /// Elements written so far
    len: usize,
    /// Element type
    _t: std::marker::PhantomData<T>,
}

impl<T: H5Type> Appender<T> {
    /// Create an empty, unlimited, chunked dataset.
    ///
    /// ### Params
    ///
    /// * `group` - Parent group
    /// * `name` - Dataset name
    /// * `deflate` - Optional gzip level
    ///
    /// ### Returns
    ///
    /// The appender.
    pub fn new(group: &Group, name: &str, deflate: Option<u8>) -> Result<Self, SplatErrors> {
        let mut b = group.new_dataset::<T>().shape(0..).chunk(H5_CHUNK);
        if let Some(level) = deflate {
            b = b.deflate(level);
        }
        Ok(Self {
            ds: b.create(name)?,
            len: 0,
            _t: std::marker::PhantomData,
        })
    }

    /// Append values at the end.
    ///
    /// ### Params
    ///
    /// * `data` - Values
    ///
    /// ### Returns
    ///
    /// `Ok(())` or an HDF5 error.
    pub fn append(&mut self, data: &[T]) -> Result<(), SplatErrors> {
        if data.is_empty() {
            return Ok(());
        }
        let end = self.len + data.len();
        self.ds.resize(end)?;
        self.ds.write_slice(data, self.len..end)?;
        self.len = end;
        Ok(())
    }
}
