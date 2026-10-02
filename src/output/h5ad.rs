//! AnnData `DGE.h5ad` (anndata on-disk format 0.1.0): `X` as float32 CSR
//! with cells as rows, `obs` with splatter's cell truth plus `sample` (a
//! copy of `Batch`, the benchmark's batch key), `var` with the gene truth.
//! `X/data` and `X/indices` stream into resizable datasets; `X/indptr` is
//! kept in memory (8 bytes per cell) and written at the end.

use std::path::Path;

use hdf5::{File, Group};

use super::Sink;
use super::h5::{Appender, encoding, vlu, write_1d};
use crate::cells::{CellChunk, CellMeta};
use crate::errors::SplatErrors;
use crate::genes::GeneTruth;
use crate::output::tables::{cell_name, gene_name};
use crate::params::{Method, SplatParams};

/////////////
// Helpers //
/////////////

/// Write a categorical column.
///
/// ### Params
///
/// * `parent` - Dataframe group
/// * `name` - Column name
/// * `codes` - Zero-based category per row
/// * `categories` - Category labels
///
/// ### Returns
///
/// `Ok(())` or an HDF5 error.
fn write_categorical(
    parent: &Group,
    name: &str,
    codes: &[u32],
    categories: &[String],
) -> Result<(), SplatErrors> {
    let g = parent.create_group(name)?;
    encoding(&g, "categorical", "0.2.0")?;
    g.new_attr::<bool>()
        .shape(())
        .create("ordered")?
        .write_scalar(&false)?;
    let codes: Vec<i32> = codes.iter().map(|&c| c as i32).collect();
    write_1d(&g, "codes", &codes)?;
    write_string_array(&g, "categories", categories)
}

/// Write a string-array dataset.
///
/// ### Params
///
/// * `parent` - Parent group
/// * `name` - Dataset name
/// * `values` - Strings
///
/// ### Returns
///
/// `Ok(())` or an HDF5 error.
fn write_string_array(parent: &Group, name: &str, values: &[String]) -> Result<(), SplatErrors> {
    let v: Vec<_> = values.iter().map(|s| vlu(s)).collect();
    let ds = write_1d(parent, name, &v)?;
    encoding(&ds, "string-array", "0.2.0")
}

/// Write a float column.
///
/// ### Params
///
/// * `parent` - Dataframe group
/// * `name` - Column name
/// * `values` - Values
///
/// ### Returns
///
/// `Ok(())` or an HDF5 error.
fn write_numeric(parent: &Group, name: &str, values: &[f64]) -> Result<(), SplatErrors> {
    let ds = write_1d(parent, name, values)?;
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
///
/// ### Returns
///
/// The group, ready for columns.
fn dataframe(
    file: &File,
    name: &str,
    index: &[String],
    columns: &[String],
) -> Result<Group, SplatErrors> {
    let g = file.create_group(name)?;
    encoding(&g, "dataframe", "0.2.0")?;
    super::h5::attr_str(&g, "_index", "_index")?;
    let cols: Vec<_> = columns.iter().map(|s| vlu(s)).collect();
    g.new_attr::<hdf5::types::VarLenUnicode>()
        .shape(cols.len())
        .create("column-order")?
        .write_raw(&cols)?;
    write_string_array(&g, "_index", index)?;
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
    data: Appender<f32>,
    /// `X/indices`
    indices: Appender<i32>,
    /// `X/indptr`, accumulated
    indptr: Vec<i64>,
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
    /// * `deflate` - Optional gzip level for `X`
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
        let obs = dataframe(&file, "obs", &cell_names, &obs_cols)?;
        let batches: Vec<String> = (1..=params.n_batches())
            .map(|b| format!("Batch{b}"))
            .collect();
        write_categorical(&obs, "Batch", &cells.batch, &batches)?;
        write_categorical(&obs, "sample", &cells.batch, &batches)?;
        if groups {
            let labels: Vec<String> = (1..=params.n_groups())
                .map(|k| format!("Group{k}"))
                .collect();
            write_categorical(&obs, "Group", &cells.group, &labels)?;
        }
        write_numeric(&obs, "ExpLibSize", &cells.exp_lib_size)?;

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
        let var = dataframe(&file, "var", &gene_names, &names)?;
        for (n, v) in &var_cols {
            write_numeric(&var, n, v)?;
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

        Ok(Self {
            data: Appender::new(&x, "data", deflate)?,
            indices: Appender::new(&x, "indices", deflate)?,
            x,
            indptr,
            _file: file,
        })
    }
}

impl Sink for H5adSink {
    fn write_chunk(&mut self, chunk: &CellChunk, _encoded: &[u8]) -> Result<(), SplatErrors> {
        let data: Vec<f32> = chunk.counts.iter().map(|&x| x as f32).collect();
        let indices: Vec<i32> = chunk.indices.iter().map(|&g| g as i32).collect();
        self.data.append(&data)?;
        self.indices.append(&indices)?;
        let base = *self.indptr.last().expect("indptr starts with 0");
        self.indptr
            .extend(chunk.indptr[1..].iter().map(|&p| base + p as i64));
        Ok(())
    }

    fn finish(self: Box<Self>) -> Result<(), SplatErrors> {
        write_1d(&self.x, "indptr", &self.indptr)?;
        Ok(())
    }
}
