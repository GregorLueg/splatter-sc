//! Simulation parameters, read from JSON with splatter's own names.
//!
//! The JSON is a superset of what `jsonlite::toJSON(getParams(p,
//! slotNames(p)), auto_unbox = TRUE, digits = NA)` writes for a `SplatParams`
//! object, so a fit from `splatEstimate` feeds straight in. The extra keys are
//! `method` and `output`. Path parameters are accepted and ignored. Defaults
//! follow the `SplatParams` prototype in splatter's `R/AllClasses.R`.

use std::path::PathBuf;

use serde::de::IgnoredAny;
use serde::{Deserialize, Deserializer, Serialize};

use crate::errors::{SplatErrors, invalid};

////////////
// Consts //
////////////

/// Tolerance for `group.prob` summing to one, as R's `all.equal`.
const GROUP_PROB_TOL: f64 = 1.5e-8;

/////////////
// Helpers //
/////////////

/// A JSON value that is either one item or an array of them. R's
/// `auto_unbox` turns length-one vectors into scalars.
#[derive(Deserialize)]
#[serde(untagged)]
enum OneOrMany<T> {
    /// A bare scalar
    One(T),
    /// An array
    Many(Vec<T>),
}

/// Deserialise a scalar or an array into a vector.
///
/// ### Params
///
/// * `de` - Serde deserialiser
///
/// ### Returns
///
/// The values as a vector.
fn one_or_many<'de, D, T>(de: D) -> Result<Vec<T>, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de>,
{
    Ok(match OneOrMany::<T>::deserialize(de)? {
        OneOrMany::One(x) => vec![x],
        OneOrMany::Many(v) => v,
    })
}

/// A number, or a string spelling of infinity as jsonlite writes it.
#[derive(Deserialize)]
#[serde(untagged)]
enum NumOrStr {
    /// Plain JSON number
    Num(f64),
    /// "Inf" and friends
    Str(String),
}

/// Deserialise a float that may be written as `"Inf"`.
///
/// ### Params
///
/// * `de` - Serde deserialiser
///
/// ### Returns
///
/// The float, with `"Inf"` / `"inf"` mapped to `f64::INFINITY`.
fn f64_or_inf<'de, D: Deserializer<'de>>(de: D) -> Result<f64, D::Error> {
    match NumOrStr::deserialize(de)? {
        NumOrStr::Num(x) => Ok(x),
        NumOrStr::Str(s) if s.eq_ignore_ascii_case("inf") => Ok(f64::INFINITY),
        NumOrStr::Str(s) => Err(serde::de::Error::custom(format!(
            "expected a number or \"Inf\", got \"{s}\""
        ))),
    }
}

/// Expand a length-one vector to `n`, as splatter's `paramsExpander`, and
/// check the length otherwise.
///
/// ### Params
///
/// * `v` - Vector to expand in place
/// * `n` - Target length
/// * `name` - Parameter name for the error
///
/// ### Returns
///
/// `Ok(())`, or an error when the length is neither 1 nor `n`.
fn expand(v: &mut Vec<f64>, n: usize, name: &str) -> Result<(), SplatErrors> {
    if v.len() == 1 && n > 1 {
        *v = vec![v[0]; n];
    }
    if v.len() != n {
        return Err(invalid(
            name,
            format!("length {} but {n} expected", v.len()),
        ));
    }
    Ok(())
}

/// Check every value of a vector against a closed range.
///
/// ### Params
///
/// * `v` - Values
/// * `lo` - Lower bound
/// * `hi` - Upper bound
/// * `name` - Parameter name for the error
///
/// ### Returns
///
/// `Ok(())`, or an error naming the first offending value.
fn check_range(v: &[f64], lo: f64, hi: f64, name: &str) -> Result<(), SplatErrors> {
    match v.iter().find(|x| !(**x >= lo && **x <= hi)) {
        Some(x) => Err(invalid(name, format!("{x} is outside [{lo}, {hi}]"))),
        None => Ok(()),
    }
}

///////////
// Enums //
///////////

/// Which Splat flavour to simulate.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Method {
    /// One population
    #[default]
    Single,
    /// Discrete groups (cell types) with DE
    Groups,
    /// Trajectories; rejected by `resolve`
    Paths,
}

/// How dropout parameters are shared across cells.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DropoutType {
    /// No dropout
    #[default]
    None,
    /// One midpoint and shape for all cells
    Experiment,
    /// One pair per batch
    Batch,
    /// One pair per group
    Group,
    /// One pair per cell; rejected by `resolve`
    Cell,
}

/// On-disk layouts the simulator can write.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Layout {
    /// `DGE.mtx` (cells x genes), `cell_metadata.csv`, `all_genes.csv`
    Parse,
    /// `tenx/matrix.mtx.gz` (genes x cells), `barcodes.tsv.gz`, `features.tsv.gz`
    TenxMtx,
    /// `DGE.h5ad`, float32 CSR with cells as rows
    H5ad,
    /// `matrix.h5`, Cell Ranger v3 layout
    TenxH5,
}

//////////////////
// OutputParams //
//////////////////

/// Where and how to write the output.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct OutputParams {
    /// Output directory, created if missing
    pub dir: PathBuf,
    /// Layouts to write; all go into `dir`
    pub layouts: Vec<Layout>,
    /// gzip level (0-9) for the HDF5 count datasets; `None` writes them
    /// uncompressed. HDF5 compresses on the single writer thread.
    pub h5_compression: Option<u8>,
}

impl Default for OutputParams {
    fn default() -> Self {
        Self {
            dir: PathBuf::from("splat_out"),
            layouts: vec![Layout::Parse],
            h5_compression: None,
        }
    }
}

/////////////////
// SplatParams //
/////////////////

/// Splat parameters in splatter's naming. Call [`SplatParams::resolve`]
/// before use: it expands length-one vectors, applies splatter's validity
/// checks and fills in derived counts.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SplatParams {
    /// Number of genes
    #[serde(rename = "nGenes")]
    pub n_genes: usize,
    /// Number of cells; derived from `batchCells`, checked if given
    #[serde(rename = "nCells")]
    pub n_cells: Option<usize>,
    /// Number of batches; derived from `batchCells`, checked if given
    #[serde(rename = "nBatches")]
    pub n_batches: Option<usize>,
    /// Number of groups; derived from `group.prob`, checked if given
    #[serde(rename = "nGroups")]
    pub n_groups: Option<usize>,
    /// Random seed; drawn at random when absent
    pub seed: Option<u64>,
    /// Simulation flavour
    pub method: Method,

    /// Cells per batch, in order; batches are contiguous blocks of cells
    #[serde(rename = "batchCells", deserialize_with = "one_or_many")]
    pub batch_cells: Vec<usize>,
    /// Log-normal location of batch factors, per batch
    #[serde(rename = "batch.facLoc", deserialize_with = "one_or_many")]
    pub batch_fac_loc: Vec<f64>,
    /// Log-normal scale of batch factors, per batch
    #[serde(rename = "batch.facScale", deserialize_with = "one_or_many")]
    pub batch_fac_scale: Vec<f64>,
    /// Draw batch factors but set them all to 1
    #[serde(rename = "batch.rmEffect")]
    pub batch_rm_effect: bool,

    /// Gamma shape of the base gene means
    #[serde(rename = "mean.shape")]
    pub mean_shape: f64,
    /// Gamma rate of the base gene means
    #[serde(rename = "mean.rate")]
    pub mean_rate: f64,

    /// Location of the library size distribution
    #[serde(rename = "lib.loc")]
    pub lib_loc: f64,
    /// Scale of the library size distribution
    #[serde(rename = "lib.scale")]
    pub lib_scale: f64,
    /// Normal instead of log-normal library sizes
    #[serde(rename = "lib.norm")]
    pub lib_norm: bool,

    /// Probability that a gene is an expression outlier
    #[serde(rename = "out.prob")]
    pub out_prob: f64,
    /// Log-normal location of outlier factors
    #[serde(rename = "out.facLoc")]
    pub out_fac_loc: f64,
    /// Log-normal scale of outlier factors
    #[serde(rename = "out.facScale")]
    pub out_fac_scale: f64,

    /// Group probabilities; rescaled to sum to one
    #[serde(rename = "group.prob", deserialize_with = "one_or_many")]
    pub group_prob: Vec<f64>,
    /// Group probabilities per batch, one row of `nGroups` per batch; rows
    /// are rescaled to sum to one and may hold zeros. Overrides `group.prob`
    /// for the group draw. Not in splatter.
    #[serde(rename = "batch.groupProb")]
    pub batch_group_prob: Option<Vec<Vec<f64>>>,
    /// Probability that a gene is DE, per group
    #[serde(rename = "de.prob", deserialize_with = "one_or_many")]
    pub de_prob: Vec<f64>,
    /// Probability that a DE gene is down-regulated, per group
    #[serde(rename = "de.downProb", deserialize_with = "one_or_many")]
    pub de_down_prob: Vec<f64>,
    /// Log-normal location of DE factors, per group
    #[serde(rename = "de.facLoc", deserialize_with = "one_or_many")]
    pub de_fac_loc: Vec<f64>,
    /// Log-normal scale of DE factors, per group
    #[serde(rename = "de.facScale", deserialize_with = "one_or_many")]
    pub de_fac_scale: Vec<f64>,

    /// Common biological coefficient of variation
    #[serde(rename = "bcv.common")]
    pub bcv_common: f64,
    /// Degrees of freedom of the BCV inverse chi-squared; `"Inf"` disables it
    #[serde(rename = "bcv.df", deserialize_with = "f64_or_inf")]
    pub bcv_df: f64,

    /// How dropout parameters are shared
    #[serde(rename = "dropout.type")]
    pub dropout_type: DropoutType,
    /// Logistic midpoint(s) of dropout in log mean expression
    #[serde(rename = "dropout.mid", deserialize_with = "one_or_many")]
    pub dropout_mid: Vec<f64>,
    /// Logistic shape(s) of dropout
    #[serde(rename = "dropout.shape", deserialize_with = "one_or_many")]
    pub dropout_shape: Vec<f64>,

    /// Ignored (paths are not ported)
    #[serde(rename = "path.from", skip_serializing)]
    pub path_from: Option<IgnoredAny>,
    /// Ignored (paths are not ported)
    #[serde(rename = "path.nSteps", skip_serializing)]
    pub path_n_steps: Option<IgnoredAny>,
    /// Ignored (paths are not ported)
    #[serde(rename = "path.skew", skip_serializing)]
    pub path_skew: Option<IgnoredAny>,
    /// Ignored (paths are not ported)
    #[serde(rename = "path.nonlinearProb", skip_serializing)]
    pub path_nonlinear_prob: Option<IgnoredAny>,
    /// Ignored (paths are not ported)
    #[serde(rename = "path.sigmaFac", skip_serializing)]
    pub path_sigma_fac: Option<IgnoredAny>,

    /// Output location and layouts
    pub output: OutputParams,
}

impl Default for SplatParams {
    fn default() -> Self {
        Self {
            n_genes: 10_000,
            n_cells: None,
            n_batches: None,
            n_groups: None,
            seed: None,
            method: Method::Single,
            batch_cells: vec![100],
            batch_fac_loc: vec![0.1],
            batch_fac_scale: vec![0.1],
            batch_rm_effect: false,
            mean_shape: 0.6,
            mean_rate: 0.3,
            lib_loc: 11.0,
            lib_scale: 0.2,
            lib_norm: false,
            out_prob: 0.05,
            out_fac_loc: 4.0,
            out_fac_scale: 0.5,
            group_prob: vec![1.0],
            batch_group_prob: None,
            de_prob: vec![0.1],
            de_down_prob: vec![0.5],
            de_fac_loc: vec![0.1],
            de_fac_scale: vec![0.4],
            bcv_common: 0.1,
            bcv_df: 60.0,
            dropout_type: DropoutType::None,
            dropout_mid: vec![0.0],
            dropout_shape: vec![-1.0],
            path_from: None,
            path_n_steps: None,
            path_skew: None,
            path_nonlinear_prob: None,
            path_sigma_fac: None,
            output: OutputParams::default(),
        }
    }
}

impl SplatParams {
    /// Read parameters from a JSON file.
    ///
    /// ### Params
    ///
    /// * `path` - JSON file
    ///
    /// ### Returns
    ///
    /// Unresolved parameters.
    pub fn from_json_file(path: &std::path::Path) -> Result<Self, SplatErrors> {
        let file = std::io::BufReader::new(std::fs::File::open(path)?);
        Ok(serde_json::from_reader(file)?)
    }

    /// Number of cells (sum of `batchCells`).
    ///
    /// ### Returns
    ///
    /// Total cell count.
    pub fn n_cells(&self) -> usize {
        self.batch_cells.iter().sum()
    }

    /// Number of batches (length of `batchCells`).
    ///
    /// ### Returns
    ///
    /// Batch count.
    pub fn n_batches(&self) -> usize {
        self.batch_cells.len()
    }

    /// Number of groups (length of `group.prob`).
    ///
    /// ### Returns
    ///
    /// Group count.
    pub fn n_groups(&self) -> usize {
        self.group_prob.len()
    }

    /// Validate, expand and fill in derived values, following splatter's
    /// `setParams`, `expandParams` and `setValidity` for `SplatParams`.
    ///
    /// `group.prob` that does not sum to one is rescaled with a warning, and
    /// `groups` with a single group falls back to `single`, both as in R. A
    /// missing seed is drawn at random and stored, so the resolved parameters
    /// reproduce the run.
    ///
    /// ### Returns
    ///
    /// The resolved parameters, or the first failed check.
    pub fn resolve(mut self) -> Result<Self, SplatErrors> {
        if self.method == Method::Paths {
            return Err(SplatErrors::Unsupported {
                name: "method",
                value: "paths".into(),
            });
        }
        if self.dropout_type == DropoutType::Cell {
            return Err(SplatErrors::Unsupported {
                name: "dropout.type",
                value: "cell".into(),
            });
        }
        if self.n_genes < 1 {
            return Err(invalid("nGenes", "must be at least 1"));
        }
        if self.batch_cells.is_empty() || self.batch_cells.contains(&0) {
            return Err(invalid("batchCells", "every batch needs at least 1 cell"));
        }
        if self.n_cells.is_some_and(|n| n != self.n_cells())
            || self.n_batches.is_some_and(|n| n != self.n_batches())
        {
            return Err(invalid(
                "batchCells",
                "nCells, nBatches and batchCells are not consistent",
            ));
        }

        let n_batches = self.n_batches();
        expand(&mut self.batch_fac_loc, n_batches, "batch.facLoc")?;
        expand(&mut self.batch_fac_scale, n_batches, "batch.facScale")?;
        check_range(&self.batch_fac_scale, 0.0, f64::INFINITY, "batch.facScale")?;

        // Strictly positive, unlike R's `lower = 0`: a zero gamma parameter
        // gives degenerate means in R and a constructor error here.
        if !(self.mean_shape > 0.0 && self.mean_shape.is_finite()) {
            return Err(invalid("mean.shape", "must be positive and finite"));
        }
        if !(self.mean_rate > 0.0 && self.mean_rate.is_finite()) {
            return Err(invalid("mean.rate", "must be positive and finite"));
        }
        if !self.lib_loc.is_finite() {
            return Err(invalid("lib.loc", "must be finite"));
        }
        check_range(&[self.lib_scale], 0.0, f64::INFINITY, "lib.scale")?;
        check_range(&[self.out_prob], 0.0, 1.0, "out.prob")?;
        if !self.out_fac_loc.is_finite() {
            return Err(invalid("out.facLoc", "must be finite"));
        }
        check_range(&[self.out_fac_scale], 0.0, f64::INFINITY, "out.facScale")?;

        if self.group_prob.is_empty() {
            return Err(invalid("group.prob", "must not be empty"));
        }
        check_range(&self.group_prob, 0.0, 1.0, "group.prob")?;
        let total: f64 = self.group_prob.iter().sum();
        if (total - 1.0).abs() > GROUP_PROB_TOL {
            if total <= 0.0 {
                return Err(invalid("group.prob", "must have a positive sum"));
            }
            eprintln!("Warning: group.prob does not sum to 1 and will be rescaled");
            self.group_prob.iter_mut().for_each(|p| *p /= total);
        }
        if self.n_groups.is_some_and(|n| n != self.n_groups()) {
            return Err(invalid("nGroups", "not consistent with group.prob"));
        }

        let n_groups = self.n_groups();
        if let Some(rows) = &mut self.batch_group_prob {
            if rows.len() != n_batches {
                return Err(invalid(
                    "batch.groupProb",
                    format!("{} rows but {n_batches} batches", rows.len()),
                ));
            }
            for row in rows.iter_mut() {
                if row.len() != n_groups {
                    return Err(invalid(
                        "batch.groupProb",
                        format!("row of length {} but {n_groups} groups", row.len()),
                    ));
                }
                check_range(row, 0.0, 1.0, "batch.groupProb")?;
                let total: f64 = row.iter().sum();
                if !(total > 0.0) {
                    return Err(invalid("batch.groupProb", "every row needs a positive sum"));
                }
                row.iter_mut().for_each(|p| *p /= total);
            }
        }
        expand(&mut self.de_prob, n_groups, "de.prob")?;
        expand(&mut self.de_down_prob, n_groups, "de.downProb")?;
        expand(&mut self.de_fac_loc, n_groups, "de.facLoc")?;
        expand(&mut self.de_fac_scale, n_groups, "de.facScale")?;
        check_range(&self.de_prob, 0.0, 1.0, "de.prob")?;
        check_range(&self.de_down_prob, 0.0, 1.0, "de.downProb")?;
        check_range(&self.de_fac_scale, 0.0, f64::INFINITY, "de.facScale")?;

        check_range(&[self.bcv_common], 0.0, f64::INFINITY, "bcv.common")?;
        // R allows 0; rchisq(df = 0) is all zeros there, so every BCV is Inf.
        if !(self.bcv_df > 0.0) {
            return Err(invalid("bcv.df", "must be positive (or \"Inf\")"));
        }

        let n_dropout = match self.dropout_type {
            DropoutType::None => None,
            DropoutType::Experiment => Some(1),
            DropoutType::Batch => Some(n_batches),
            DropoutType::Group => Some(n_groups),
            DropoutType::Cell => unreachable!("rejected above"),
        };
        if let Some(n) = n_dropout
            && (self.dropout_mid.len() != n || self.dropout_shape.len() != n)
        {
            return Err(invalid(
                "dropout.mid",
                format!("dropout.mid and dropout.shape must have length {n}"),
            ));
        }
        if self
            .dropout_mid
            .iter()
            .chain(&self.dropout_shape)
            .any(|x| !x.is_finite())
        {
            return Err(invalid("dropout.mid", "dropout parameters must be finite"));
        }

        if self.method == Method::Groups && n_groups == 1 {
            eprintln!("Warning: nGroups is 1, switching to single mode");
            self.method = Method::Single;
        }
        if self.dropout_type == DropoutType::Group && self.method == Method::Single {
            return Err(invalid(
                "dropout.type",
                "'group' but groups have not been simulated",
            ));
        }
        if self.output.layouts.is_empty() {
            return Err(invalid("output.layouts", "must name at least one layout"));
        }
        if self.output.h5_compression.is_some_and(|l| l > 9) {
            return Err(invalid("output.h5_compression", "gzip level must be 0-9"));
        }

        self.n_cells = Some(self.n_cells());
        self.n_batches = Some(n_batches);
        self.n_groups = Some(n_groups);
        self.seed = Some(self.seed.unwrap_or_else(rand::random));
        Ok(self)
    }

    /// The seed of a resolved parameter set.
    ///
    /// ### Returns
    ///
    /// The seed. Panics if called before [`SplatParams::resolve`].
    pub fn seed(&self) -> u64 {
        self.seed.expect("seed is set by resolve()")
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_params_r_dump_loads() {
        let json = r#"{"nBatches":2,"batchCells":[50,70],"batch.facLoc":0.1,
            "batch.facScale":0.1,"batch.rmEffect":false,"mean.shape":0.6,
            "mean.rate":0.3,"lib.loc":11,"lib.scale":0.2,"lib.norm":false,
            "out.prob":0.05,"out.facLoc":4,"out.facScale":0.5,"nGroups":2,
            "group.prob":[0.3,0.7],"de.prob":0.1,"de.downProb":0.5,
            "de.facLoc":[0.1,0.5],"de.facScale":0.4,"bcv.common":0.1,
            "bcv.df":"Inf","dropout.type":"none","dropout.mid":0,
            "dropout.shape":-1,"path.from":0,"path.nSteps":100,"path.skew":0.5,
            "path.nonlinearProb":0.1,"path.sigmaFac":0.8,"nGenes":10000,
            "nCells":120,"seed":62600}"#;
        let p: SplatParams = serde_json::from_str(json).unwrap();
        let p = p.resolve().unwrap();
        assert_eq!(p.n_cells(), 120);
        assert_eq!(p.batch_fac_loc, vec![0.1, 0.1]);
        assert_eq!(p.de_fac_loc, vec![0.1, 0.5]);
        assert!(p.bcv_df.is_infinite());
        assert_eq!(p.seed(), 62600);
    }

    #[test]
    fn test_params_unknown_key_rejected() {
        assert!(serde_json::from_str::<SplatParams>(r#"{"mean.shap":1}"#).is_err());
    }

    #[test]
    fn test_params_single_group_falls_back_to_single() {
        let p = SplatParams {
            method: Method::Groups,
            ..Default::default()
        };
        assert_eq!(p.resolve().unwrap().method, Method::Single);
    }

    #[test]
    fn test_params_group_prob_rescaled() {
        let p = SplatParams {
            group_prob: vec![0.2, 0.2],
            ..Default::default()
        };
        assert_eq!(p.resolve().unwrap().group_prob, vec![0.5, 0.5]);
    }

    #[test]
    fn test_params_batch_group_prob_checked() {
        let base = SplatParams {
            batch_cells: vec![10, 10],
            group_prob: vec![0.5, 0.5],
            ..Default::default()
        };
        let p = SplatParams {
            batch_group_prob: Some(vec![vec![2.0, 2.0]]),
            ..base.clone()
        };
        assert!(p.resolve().is_err());
        let p = SplatParams {
            batch_group_prob: Some(vec![vec![0.2, 0.2], vec![0.0, 1.0]]),
            ..base
        };
        assert_eq!(
            p.resolve().unwrap().batch_group_prob,
            Some(vec![vec![0.5, 0.5], vec![0.0, 1.0]])
        );
    }

    #[test]
    fn test_params_inconsistent_ncells_rejected() {
        let p = SplatParams {
            n_cells: Some(10),
            ..Default::default()
        };
        assert!(p.resolve().is_err());
    }

    #[test]
    fn test_params_wrong_vector_length_rejected() {
        let p = SplatParams {
            group_prob: vec![0.5, 0.5],
            de_prob: vec![0.1, 0.1, 0.1],
            ..Default::default()
        };
        assert!(p.resolve().is_err());
    }

    #[test]
    fn test_params_dropout_lengths_checked() {
        let p = SplatParams {
            batch_cells: vec![10, 10],
            dropout_type: DropoutType::Batch,
            ..Default::default()
        };
        assert!(p.resolve().is_err());
    }

    #[test]
    fn test_params_paths_unsupported() {
        let p = SplatParams {
            method: Method::Paths,
            ..Default::default()
        };
        assert!(matches!(p.resolve(), Err(SplatErrors::Unsupported { .. })));
    }
}
