//! AnnData `DGE.h5ad` (anndata on-disk format 0.1.0): `X` as float32 CSR
//! with cells as rows, `obs` with splatter's cell truth plus `sample` (a
//! copy of `Batch`, the benchmark's batch key), `var` with the gene truth.
//! `X/data` and `X/indices` arrive as filtered chunks from the encoder;
//! `X/indptr` is kept in memory (8 bytes per cell) and written at the end.
//! Dtypes, filters and chunk shapes follow
//! `anndata.write_h5ad(compression="gzip")`: every dataset gzipped over
//! h5py's guessed chunk, `X/*` resizable.

use std::path::Path;

use hdf5::{File, Group};

use super::Sink;
use super::h5::{ChunkWriter, H5Chunks, encoding, guess_chunk, vlu, write_1d, write_1d_resizable};
use crate::cells::{CellChunk, CellMeta};
use crate::errors::SplatErrors;
use crate::genes::GeneTruth;
use crate::output::tables::{cell_name, gene_name};
use crate::params::{Method, SplatParams};

////////////
// Consts //
////////////

/// Element type of `X/data`.
pub(crate) type Data = f32;

/// Element type of `X/indices`.
pub(crate) type Indices = i32;

/// No byte-shuffle before gzip, as scanpy and anndata write `X`.
pub(crate) const SHUFFLE: bool = false;

/////////////
// Helpers //
/////////////

/// Chunk length of `X/data` and `X/indices`: h5py's guess for the final
/// length, which a streamed write only knows as an estimate.
///
/// ### Params
///
/// * `nnz_hint` - Expected total nonzeros
///
/// ### Returns
///
/// Elements per chunk.
pub(crate) fn x_chunk(nnz_hint: usize) -> usize {
    debug_assert_eq!(size_of::<Data>(), size_of::<Indices>());
    guess_chunk(nnz_hint, size_of::<Data>())
}

/// Write a categorical column. Codes take pandas' narrowest signed type
/// for the number of categories, which is what anndata stores.
///
/// ### Params
///
/// * `parent` - Dataframe group
/// * `name` - Column name
/// * `codes` - Zero-based category per row
/// * `categories` - Category labels
/// * `deflate` - Optional gzip level
///
/// ### Returns
///
/// `Ok(())` or an HDF5 error.
fn write_categorical(
    parent: &Group,
    name: &str,
    codes: &[u32],
    categories: &[String],
    deflate: Option<u8>,
) -> Result<(), SplatErrors> {
    let g = parent.create_group(name)?;
    encoding(&g, "categorical", "0.2.0")?;
    g.new_attr::<bool>()
        .shape(())
        .create("ordered")?
        .write_scalar(&false)?;
    let k = categories.len();
    if k < i8::MAX as usize {
        let c: Vec<i8> = codes.iter().map(|&c| c as i8).collect();
        write_1d(&g, "codes", &c, deflate)?;
    } else if k < i16::MAX as usize {
        let c: Vec<i16> = codes.iter().map(|&c| c as i16).collect();
        write_1d(&g, "codes", &c, deflate)?;
    } else {
        let c: Vec<i32> = codes.iter().map(|&c| c as i32).collect();
        write_1d(&g, "codes", &c, deflate)?;
    }
    write_string_array(&g, "categories", categories, deflate)
}

/// Write a string-array dataset.
///
/// ### Params
///
/// * `parent` - Parent group
/// * `name` - Dataset name
/// * `values` - Strings
/// * `deflate` - Optional gzip level
///
/// ### Returns
///
/// `Ok(())` or an HDF5 error.
fn write_string_array(
    parent: &Group,
    name: &str,
    values: &[String],
    deflate: Option<u8>,
) -> Result<(), SplatErrors> {
    let v: Vec<_> = values.iter().map(|s| vlu(s)).collect();
    let ds = write_1d(parent, name, &v, deflate)?;
    encoding(&ds, "string-array", "0.2.0")
}

/// Write a float column.
///
/// ### Params
///
/// * `parent` - Dataframe group
/// * `name` - Column name
/// * `values` - Values
/// * `deflate` - Optional gzip level
///
/// ### Returns
///
/// `Ok(())` or an HDF5 error.
fn write_numeric(
    parent: &Group,
    name: &str,
    values: &[f64],
    deflate: Option<u8>,
) -> Result<(), SplatErrors> {
    let ds = write_1d(parent, name, values, deflate)?;
    encoding(&ds, "array", "0.2.0")
}

/// Create a dataframe group with its index.
///
/// ### Params
///
/// * `file` - Open file
/// * `name` - `obs` or `var`
/// * `index` - Row names
/// * `columns` - Column names in order
/// * `deflate` - Optional gzip level
///
/// ### Returns
///
/// The group, ready for columns.
fn dataframe(
    file: &File,
    name: &str,
    index: &[String],
    columns: &[String],
    deflate: Option<u8>,
) -> Result<Group, SplatErrors> {
    let g = file.create_group(name)?;
    encoding(&g, "dataframe", "0.2.0")?;
    super::h5::attr_str(&g, "_index", "_index")?;
    let cols: Vec<_> = columns.iter().map(|s| vlu(s)).collect();
    g.new_attr::<hdf5::types::VarLenUnicode>()
        .shape(cols.len())
        .create("column-order")?
        .write_raw(&cols)?;
    write_string_array(&g, "_index", index, deflate)?;
    Ok(g)
}

//////////////
// H5adSink //
//////////////

/// Writer for the h5ad layout.
pub struct H5adSink {
    /// Open file; kept so it closes after the last write
    _file: File,
    /// `X` group, for `indptr` at the end
    x: Group,
    /// `X/data`
    data: ChunkWriter,
    /// `X/indices`
    indices: ChunkWriter,
    /// `X/indptr`, accumulated
    indptr: Vec<i64>,
    /// Optional gzip level
    deflate: Option<u8>,
}

impl H5adSink {
    /// Write `obs`, `var` and the empty slots, and open `X`.
    ///
    /// ### Params
    ///
    /// * `dir` - Output directory
    /// * `params` - Resolved parameters
    /// * `cells` - Cell metadata
    /// * `genes` - Gene truth
    /// * `deflate` - Optional gzip level for every dataset
    /// * `nnz_hint` - Expected total nonzeros, for the `X` chunk length
    ///
    /// ### Returns
    ///
    /// The sink.
    pub fn new(
        dir: &Path,
        params: &SplatParams,
        cells: &CellMeta,
        genes: &GeneTruth,
        deflate: Option<u8>,
        nnz_hint: usize,
    ) -> Result<Self, SplatErrors> {
        let file = File::create(dir.join("DGE.h5ad"))?;
        encoding(&file, "anndata", "0.1.0")?;

        let n_cells = cells.n_cells();
        let groups = params.method == Method::Groups;
        let mut obs_cols = vec!["Batch".to_string()];
        if groups {
            obs_cols.push("Group".into());
        }
        obs_cols.extend(["ExpLibSize".into(), "sample".into()]);
        let cell_names: Vec<String> = (0..n_cells).map(cell_name).collect();
        let obs = dataframe(&file, "obs", &cell_names, &obs_cols, deflate)?;
        let batches: Vec<String> = (1..=params.n_batches())
            .map(|b| format!("Batch{b}"))
            .collect();
        write_categorical(&obs, "Batch", &cells.batch, &batches, deflate)?;
        write_categorical(&obs, "sample", &cells.batch, &batches, deflate)?;
        if groups {
            let labels: Vec<String> = (1..=params.n_groups())
                .map(|k| format!("Group{k}"))
                .collect();
            write_categorical(&obs, "Group", &cells.group, &labels, deflate)?;
        }
        write_numeric(&obs, "ExpLibSize", &cells.exp_lib_size, deflate)?;

        let mut var_cols: Vec<(String, &[f64])> = vec![
            ("BaseGeneMean".into(), &genes.base_gene_mean),
            ("OutlierFactor".into(), &genes.outlier_factor),
            ("GeneMean".into(), &genes.gene_mean),
        ];
        for (b, v) in genes.batch_fac.iter().enumerate() {
            var_cols.push((format!("BatchFacBatch{}", b + 1), v));
        }
        for (k, v) in genes.de_fac.iter().enumerate() {
            var_cols.push((format!("DEFacGroup{}", k + 1), v));
        }
        var_cols.push(("BCVChiFac".into(), &genes.bcv_chi_fac));
        let gene_names: Vec<String> = (0..genes.n_genes()).map(gene_name).collect();
        let names: Vec<String> = var_cols.iter().map(|(n, _)| n.clone()).collect();
        let var = dataframe(&file, "var", &gene_names, &names, deflate)?;
        for (n, v) in &var_cols {
            write_numeric(&var, n, v, deflate)?;
        }

        for slot in ["obsm", "varm", "obsp", "varp", "layers", "uns"] {
            let g = file.create_group(slot)?;
            encoding(&g, "dict", "0.1.0")?;
        }

        let x = file.create_group("X")?;
        encoding(&x, "csr_matrix", "0.1.0")?;
        x.new_attr::<i64>()
            .shape(2)
            .create("shape")?
            .write_raw(&[n_cells as i64, genes.n_genes() as i64])?;
        let mut indptr = Vec::with_capacity(n_cells + 1);
        indptr.push(0);

        let chunk = x_chunk(nnz_hint);
        Ok(Self {
            data: ChunkWriter::new::<Data>(&x, "data", deflate, SHUFFLE, chunk)?,
            indices: ChunkWriter::new::<Indices>(&x, "indices", deflate, SHUFFLE, chunk)?,
            x,
            indptr,
            deflate,
            _file: file,
        })
    }
}

impl Sink for H5adSink {
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
        // scipy, and so anndata, keeps int32 indptr while nnz fits.
        let nnz = *self.indptr.last().expect("indptr starts with 0");
        if nnz <= i32::MAX as i64 {
            let p: Vec<i32> = self.indptr.iter().map(|&p| p as i32).collect();
            let c = guess_chunk(p.len(), size_of::<i32>());
            write_1d_resizable(&self.x, "indptr", &p, c, self.deflate, SHUFFLE)
        } else {
            let c = guess_chunk(self.indptr.len(), size_of::<i64>());
            write_1d_resizable(&self.x, "indptr", &self.indptr, c, self.deflate, SHUFFLE)
        }
    }
}
