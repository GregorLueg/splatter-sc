//! Per-cell metadata and the per-entry count simulation.
//!
//! Metadata (batch, group, expected library size) is drawn for all cells up
//! front: it is a few bytes per cell, and `lib.norm` needs the global minimum
//! positive library size. Counts are then simulated in fixed-size chunks of
//! cells, each with its own RNG stream, so output does not depend on the
//! thread count. Mirrors `splatSimLibSizes`, the cell half of
//! `splatSimGroupCellMeans`, `splatSimBCVMeans`, `splatSimTrueCounts` and
//! `splatSimDropout` in splatter's `R/splat-simulate.R`.

use rand::Rng;
use rand_distr::{Distribution, Gamma, LogNormal, Normal, Poisson, weighted::WeightedIndex};

use crate::errors::{SplatErrors, invalid};
use crate::genes::{GeneTruth, Profiles};
use crate::params::{DropoutType, Method, SplatParams};

/// Per-cell ground truth.
#[derive(Clone, Debug)]
pub struct CellMeta {
    /// Batch index per cell; batches are contiguous blocks of `batchCells`
    pub batch: Vec<u32>,
    /// Group index per cell; all zero in single mode
    pub group: Vec<u32>,
    /// `ExpLibSize`: expected library size per cell
    pub exp_lib_size: Vec<f64>,
}

impl CellMeta {
    /// Draw batches, groups and expected library sizes for all cells.
    ///
    /// ### Params
    ///
    /// * `params` - Resolved parameters
    /// * `rng` - The cell RNG stream
    ///
    /// ### Returns
    ///
    /// The cell metadata.
    pub fn simulate<R: Rng>(params: &SplatParams, rng: &mut R) -> Result<Self, SplatErrors> {
        let n = params.n_cells();
        let batch: Vec<u32> = params
            .batch_cells
            .iter()
            .enumerate()
            .flat_map(|(b, &k)| std::iter::repeat_n(b as u32, k))
            .collect();

        let group = if params.method == Method::Groups {
            let w = WeightedIndex::new(&params.group_prob)
                .map_err(|e| invalid("group.prob", e.to_string()))?;
            (0..n).map(|_| w.sample(rng) as u32).collect()
        } else {
            vec![0; n]
        };

        let exp_lib_size = if params.lib_norm {
            let d = Normal::new(params.lib_loc, params.lib_scale)
                .map_err(|e| invalid("lib.scale", e.to_string()))?;
            let mut lib: Vec<f64> = (0..n).map(|_| d.sample(rng)).collect();
            let min_pos = lib
                .iter()
                .copied()
                .filter(|&x| x > 0.0)
                .fold(f64::INFINITY, f64::min);
            lib.iter_mut()
                .filter(|x| **x < 0.0)
                .for_each(|x| *x = min_pos / 2.0);
            lib
        } else {
            let d = LogNormal::new(params.lib_loc, params.lib_scale)
                .map_err(|e| invalid("lib.scale", e.to_string()))?;
            (0..n).map(|_| d.sample(rng)).collect()
        };

        Ok(Self {
            batch,
            group,
            exp_lib_size,
        })
    }

    /// Number of cells.
    ///
    /// ### Returns
    ///
    /// Cell count.
    pub fn n_cells(&self) -> usize {
        self.batch.len()
    }
}

/// Simulated counts for a contiguous range of cells, cell-major CSR.
#[derive(Clone, Debug, Default)]
pub struct CellChunk {
    /// Index of the first cell
    pub start: usize,
    /// Row pointers into `indices`/`counts`, length `n_cells + 1`, local to
    /// the chunk
    pub indptr: Vec<usize>,
    /// Zero-based gene index of each nonzero, ascending within a cell
    pub indices: Vec<u32>,
    /// Count of each nonzero
    pub counts: Vec<u32>,
}

impl CellChunk {
    /// Number of cells in the chunk.
    ///
    /// ### Returns
    ///
    /// Cell count.
    pub fn n_cells(&self) -> usize {
        self.indptr.len() - 1
    }
}

/// Logistic dropout midpoint and shape for one cell, or `None` without
/// dropout.
///
/// ### Params
///
/// * `params` - Resolved parameters
/// * `cells` - Cell metadata
/// * `c` - Cell index
///
/// ### Returns
///
/// `(dropout.mid, dropout.shape)` for the cell.
#[inline]
fn dropout_for(params: &SplatParams, cells: &CellMeta, c: usize) -> Option<(f64, f64)> {
    let i = match params.dropout_type {
        DropoutType::None => return None,
        DropoutType::Experiment => 0,
        DropoutType::Batch => cells.batch[c] as usize,
        DropoutType::Group => cells.group[c] as usize,
        DropoutType::Cell => unreachable!("rejected by resolve()"),
    };
    Some((params.dropout_mid[i], params.dropout_shape[i]))
}

/// BCV of one entry, as in `splatSimBCVMeans`.
///
/// ### Params
///
/// * `bcv_common` - `bcv.common`
/// * `base` - `BaseCellMeans` entry
/// * `chi_fac` - The gene's `sqrt(bcv.df / chisq)`
///
/// ### Returns
///
/// `(bcv.common + 1 / sqrt(base)) * chi_fac`.
#[inline(always)]
pub fn bcv(bcv_common: f64, base: f64, chi_fac: f64) -> f64 {
    (bcv_common + 1.0 / base.sqrt()) * chi_fac
}

/// Largest base mean whose count is drawn by negative binomial inversion.
/// Inversion walks the pmf up from zero, so its cost grows with the mean,
/// while gamma + Poisson costs roughly the same at any mean. Swept 10, 30,
/// 100 at 20k genes, single thread: 10 and 30 tie, 100 is ~6% slower at
/// splatter's default depth.
const INVERSION_MAX_MEAN: f64 = 10.0;

/// Negative binomial draw by inversion of the cdf with one uniform.
///
/// `Poisson(Gamma(shape = r, scale = m / r))` is exactly NB with mean `m` and
/// size `r`: `P(0) = (r / (r + m))^r`, `P(k + 1) = P(k) (k + r) / (k + 1) q`
/// with `q = m / (r + m)`. Most entries are zero and cost one `ln_1p`, one
/// `exp` and the uniform, instead of a gamma and a Poisson draw.
///
/// ### Params
///
/// * `m` - Mean (`BaseCellMeans` entry)
/// * `r` - Size, `1 / bcv^2`
/// * `u` - Uniform on [0, 1)
///
/// ### Returns
///
/// The count.
#[inline(always)]
fn nb_invert(m: f64, r: f64, u: f64) -> u32 {
    let q = m / (r + m);
    let mut p = (-r * (m / r).ln_1p()).exp();
    let mut cdf = p;
    let mut k = 0u32;
    // `p > 0` stops the walk if rounding leaves cdf just below a u near 1.
    while u >= cdf && p > 0.0 {
        p *= (k as f64 + r) / (k as f64 + 1.0) * q;
        k += 1;
        cdf += p;
    }
    k
}

/// Simulate counts for cells `start..end`.
///
/// One fused pass per cell over all genes: base mean, BCV, count, optional
/// dropout. Only nonzeros are stored.
///
/// The count is splatter's gamma cell mean followed by a Poisson draw. For
/// base means up to [`INVERSION_MAX_MEAN`] the pair is replaced by the
/// identical negative binomial, drawn by [`nb_invert`]. Dropout needs the
/// cell mean, which is then drawn from its posterior given the count,
/// `Gamma(r + k, scale = m / (r + m))`; the joint law of (cell mean, count)
/// is unchanged. Dropout is only applied to nonzero counts, since a dropped
/// zero stays zero. Both change the random stream, not the distribution.
///
/// ### Params
///
/// * `params` - Resolved parameters
/// * `genes` - Gene truth
/// * `profiles` - Normalised (batch, group) profiles
/// * `cells` - Cell metadata
/// * `start` - First cell
/// * `end` - One past the last cell
/// * `rng` - This chunk's RNG stream
///
/// ### Returns
///
/// The chunk in cell-major CSR.
pub fn simulate_chunk<R: Rng>(
    params: &SplatParams,
    genes: &GeneTruth,
    profiles: &Profiles,
    cells: &CellMeta,
    start: usize,
    end: usize,
    rng: &mut R,
) -> Result<CellChunk, SplatErrors> {
    let mut chunk = CellChunk {
        start,
        indptr: Vec::with_capacity(end - start + 1),
        ..Default::default()
    };
    chunk.indptr.push(0);

    for c in start..end {
        let lib = cells.exp_lib_size[c];
        let prof = profiles.get(cells.batch[c] as usize, cells.group[c] as usize);
        let dropout = dropout_for(params, cells, c);

        for (g, (&p, &chi)) in prof.iter().zip(&genes.bcv_chi_fac).enumerate() {
            let base = lib * p;
            if !(base > 0.0) {
                continue;
            }
            let b = bcv(params.bcv_common, base, chi);
            let r = 1.0 / (b * b);
            let (count, mu) = if base <= INVERSION_MAX_MEAN {
                let k = nb_invert(base, r, rng.random());
                if k == 0 {
                    continue;
                }
                let mu = match dropout {
                    Some(_) => Gamma::new(r + k as f64, base / (r + base))
                        .map_err(|e| invalid("bcv", e.to_string()))?
                        .sample(rng),
                    None => f64::NAN,
                };
                (k, mu)
            } else {
                let mu: f64 = Gamma::new(r, base / r)
                    .map_err(|e| invalid("bcv", e.to_string()))?
                    .sample(rng);
                // Tiny shapes underflow to 0; Poisson rejects lambda = 0.
                if !(mu > 0.0) {
                    continue;
                }
                let k: f64 = Poisson::new(mu)
                    .map_err(|e| invalid("cell mean", e.to_string()))?
                    .sample(rng);
                if k == 0.0 {
                    continue;
                }
                (k as u32, mu)
            };
            if let Some((mid, shape)) = dropout {
                let p_drop = 1.0 / (1.0 + (-shape * (mu.ln() - mid)).exp());
                if rng.random::<f64>() < p_drop {
                    continue;
                }
            }
            chunk.indices.push(g as u32);
            chunk.counts.push(count);
        }
        chunk.indptr.push(chunk.indices.len());
    }
    Ok(chunk)
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn test_nb_invert_matches_gamma_poisson_moments() {
        // NB(mean m, size r): P(0) = (r / (r + m))^r, var = m + m^2 / r.
        let mut rng = ChaCha8Rng::seed_from_u64(5);
        let n = 400_000;
        for &(m, r) in &[(0.01, 0.5), (0.3, 2.0), (4.0, 10.0), (25.0, 3.0)] {
            let x: Vec<f64> = (0..n)
                .map(|_| nb_invert(m, r, rng.random()) as f64)
                .collect();
            let mean = x.iter().sum::<f64>() / n as f64;
            let var = x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
            let p0 = x.iter().filter(|&&v| v == 0.0).count() as f64 / n as f64;
            let (var_t, p0_t) = (m + m * m / r, (r / (r + m)).powf(r));
            let se_mean = (var_t / n as f64).sqrt();
            assert!((mean - m).abs() < 5.0 * se_mean, "m={m} r={r}: mean {mean}");
            assert!(
                (var / var_t - 1.0).abs() < 0.05,
                "m={m} r={r}: var {var} vs {var_t}"
            );
            let se_p0 = (p0_t * (1.0 - p0_t) / n as f64).sqrt();
            assert!(
                (p0 - p0_t).abs() < 5.0 * se_p0,
                "m={m} r={r}: P0 {p0} vs {p0_t}"
            );
        }
    }

    #[test]
    fn test_batches_are_contiguous_blocks() {
        let params = SplatParams {
            batch_cells: vec![3, 2],
            ..Default::default()
        }
        .resolve()
        .unwrap();
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let cells = CellMeta::simulate(&params, &mut rng).unwrap();
        assert_eq!(cells.batch, vec![0, 0, 0, 1, 1]);
        assert!(cells.group.iter().all(|&g| g == 0));
    }

    #[test]
    fn test_lib_norm_negatives_replaced() {
        let params = SplatParams {
            batch_cells: vec![2000],
            lib_norm: true,
            lib_loc: 0.0,
            lib_scale: 1.0,
            ..Default::default()
        }
        .resolve()
        .unwrap();
        let mut rng = ChaCha8Rng::seed_from_u64(2);
        let cells = CellMeta::simulate(&params, &mut rng).unwrap();
        assert!(cells.exp_lib_size.iter().all(|&x| x > 0.0));
    }

    #[test]
    fn test_chunk_tiny_means_give_zeros_not_errors() {
        // Library size near zero: BCV explodes, gamma shapes underflow and the
        // mu = 0 path must skip rather than hand Poisson a zero.
        let params = SplatParams {
            n_genes: 200,
            batch_cells: vec![20],
            lib_loc: -20.0,
            ..Default::default()
        }
        .resolve()
        .unwrap();
        let mut rng = ChaCha8Rng::seed_from_u64(3);
        let genes = GeneTruth::simulate(&params, &mut rng).unwrap();
        let prof = Profiles::new(&genes, 1, 1);
        let cells = CellMeta::simulate(&params, &mut rng).unwrap();
        let chunk = simulate_chunk(&params, &genes, &prof, &cells, 0, 20, &mut rng).unwrap();
        assert_eq!(chunk.n_cells(), 20);
        assert!(chunk.counts.len() < 20 * 200 / 10);
    }

    #[test]
    fn test_chunk_indices_ascending_within_cell() {
        let params = SplatParams {
            n_genes: 300,
            batch_cells: vec![10],
            ..Default::default()
        }
        .resolve()
        .unwrap();
        let mut rng = ChaCha8Rng::seed_from_u64(4);
        let genes = GeneTruth::simulate(&params, &mut rng).unwrap();
        let prof = Profiles::new(&genes, 1, 1);
        let cells = CellMeta::simulate(&params, &mut rng).unwrap();
        let chunk = simulate_chunk(&params, &genes, &prof, &cells, 0, 10, &mut rng).unwrap();
        for w in chunk.indptr.windows(2) {
            assert!(chunk.indices[w[0]..w[1]].windows(2).all(|p| p[0] < p[1]));
        }
        assert!(chunk.counts.iter().all(|&x| x > 0));
    }
}
