//! Per-gene state, drawn once: base means, outliers, batch and DE factors,
//! the shared BCV chi-squared factor, and the normalised expression profile
//! of every (batch, group) pair.
//!
//! Mirrors `splatSimGeneMeans`, `splatSimBatchEffects`, `splatSimGroupDE` and
//! the gene half of `splatSimBCVMeans` in splatter's `R/splat-simulate.R`.

use rand::Rng;
use rand_distr::{Bernoulli, ChiSquared, Distribution, Gamma, LogNormal};
use rayon::prelude::*;

use crate::errors::{SplatErrors, invalid};
use crate::params::{Method, SplatParams};

/// True per-gene parameters of one simulation. Written out as the gene
/// ground truth.
#[derive(Clone, Debug)]
pub struct GeneTruth {
    /// `BaseGeneMean`: gamma draw before outliers
    pub base_gene_mean: Vec<f64>,
    /// `OutlierFactor`: 1 for non-outliers
    pub outlier_factor: Vec<f64>,
    /// `GeneMean`: base mean with outliers replaced by median * factor
    pub gene_mean: Vec<f64>,
    /// `BatchFacBatch<b>`, one vector per batch; empty with a single batch,
    /// as splatter only draws them for `nBatches > 1`
    pub batch_fac: Vec<Vec<f64>>,
    /// `DEFacGroup<k>`, one vector per group; empty in single mode
    pub de_fac: Vec<Vec<f64>>,
    /// `sqrt(bcv.df / chisq[g])`, one chi-squared draw per gene shared by all
    /// cells (splatter recycles `rchisq(nGenes)` down the columns); all ones
    /// for infinite `bcv.df`
    pub bcv_chi_fac: Vec<f64>,
}

/// Port of splatter's `getLNormFactors`.
///
/// Each factor is selected with `sel_prob`. A selected factor is a log-normal
/// draw, inverted with probability `neg_prob`; a draw below one has its
/// direction flipped first, so `neg_prob` really is the share of factors
/// below one. Unselected factors are 1.
///
/// ### Params
///
/// * `rng` - Random number generator
/// * `n` - Number of factors
/// * `sel_prob` - Probability a factor differs from one
/// * `neg_prob` - Probability a selected factor is below one
/// * `loc` - Log-normal location
/// * `scale` - Log-normal scale
///
/// ### Returns
///
/// `n` factors.
pub fn get_lnorm_factors<R: Rng>(
    rng: &mut R,
    n: usize,
    sel_prob: f64,
    neg_prob: f64,
    loc: f64,
    scale: f64,
) -> Result<Vec<f64>, SplatErrors> {
    let sel = Bernoulli::new(sel_prob).map_err(|e| invalid("sel_prob", e.to_string()))?;
    let neg = Bernoulli::new(neg_prob).map_err(|e| invalid("neg_prob", e.to_string()))?;
    let lnorm = LogNormal::new(loc, scale).map_err(|e| invalid("fac.scale", e.to_string()))?;

    Ok((0..n)
        .map(|_| {
            if !sel.sample(rng) {
                return 1.0;
            }
            let down = neg.sample(rng);
            let f: f64 = lnorm.sample(rng);
            // `dir.selected[facs.selected < 1] <- -dir`: flip, then invert.
            if down != (f < 1.0) { 1.0 / f } else { f }
        })
        .collect())
}

/// Median as R computes it (mean of the two middle values for even `n`).
///
/// ### Params
///
/// * `x` - Values, not modified
///
/// ### Returns
///
/// The median.
fn median(x: &[f64]) -> f64 {
    let mut v = x.to_vec();
    v.sort_unstable_by(f64::total_cmp);
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

impl GeneTruth {
    /// Draw all per-gene parameters.
    ///
    /// ### Params
    ///
    /// * `params` - Resolved parameters
    /// * `rng` - The gene RNG stream
    ///
    /// ### Returns
    ///
    /// The gene truth.
    pub fn simulate<R: Rng>(params: &SplatParams, rng: &mut R) -> Result<Self, SplatErrors> {
        let n = params.n_genes;

        let gamma = Gamma::new(params.mean_shape, 1.0 / params.mean_rate)
            .map_err(|e| invalid("mean.shape", e.to_string()))?;
        let base_gene_mean: Vec<f64> = (0..n).map(|_| gamma.sample(rng)).collect();

        let outlier_factor = get_lnorm_factors(
            rng,
            n,
            params.out_prob,
            0.0,
            params.out_fac_loc,
            params.out_fac_scale,
        )?;
        let med = median(&base_gene_mean);
        let gene_mean = base_gene_mean
            .iter()
            .zip(&outlier_factor)
            .map(|(&m, &f)| if f != 1.0 { med * f } else { m })
            .collect();

        let mut batch_fac = Vec::new();
        if params.n_batches() > 1 {
            for b in 0..params.n_batches() {
                let facs = get_lnorm_factors(
                    rng,
                    n,
                    1.0,
                    0.5,
                    params.batch_fac_loc[b],
                    params.batch_fac_scale[b],
                )?;
                batch_fac.push(if params.batch_rm_effect {
                    vec![1.0; n]
                } else {
                    facs
                });
            }
        }

        let mut de_fac = Vec::new();
        if params.method == Method::Groups {
            for k in 0..params.n_groups() {
                de_fac.push(get_lnorm_factors(
                    rng,
                    n,
                    params.de_prob[k],
                    params.de_down_prob[k],
                    params.de_fac_loc[k],
                    params.de_fac_scale[k],
                )?);
            }
        }

        let bcv_chi_fac = if params.bcv_df.is_finite() {
            let chisq =
                ChiSquared::new(params.bcv_df).map_err(|e| invalid("bcv.df", e.to_string()))?;
            (0..n)
                .map(|_| {
                    let x: f64 = chisq.sample(rng);
                    (params.bcv_df / x).sqrt()
                })
                .collect()
        } else {
            vec![1.0; n]
        };

        Ok(Self {
            base_gene_mean,
            outlier_factor,
            gene_mean,
            batch_fac,
            de_fac,
            bcv_chi_fac,
        })
    }

    /// Number of genes.
    ///
    /// ### Returns
    ///
    /// Gene count.
    pub fn n_genes(&self) -> usize {
        self.gene_mean.len()
    }
}

/// Normalised expression profiles, one per (batch, group) pair.
///
/// In splatter, `BaseCellMeans[, c] = lib[c] * m / sum(m)` with
/// `m = GeneMean * BatchFac[batch[c]] * DEFac[group[c]]`. The normaliser only
/// depends on the pair, so it is computed once here instead of per cell.
#[derive(Clone, Debug)]
pub struct Profiles {
    /// Flat `[batch][group][gene]`, each length-`n_genes` row sums to one
    pub data: Vec<f64>,
    /// Number of genes
    pub n_genes: usize,
    /// Number of groups in the layout (1 in single mode)
    pub n_groups: usize,
}

impl Profiles {
    /// Build the profiles from the gene truth.
    ///
    /// ### Params
    ///
    /// * `genes` - Gene truth
    /// * `n_batches` - Number of batches
    /// * `n_groups` - Number of groups (1 in single mode)
    ///
    /// ### Returns
    ///
    /// The profiles, `n_batches * n_groups` rows of `n_genes`.
    pub fn new(genes: &GeneTruth, n_batches: usize, n_groups: usize) -> Self {
        let n_genes = genes.n_genes();
        let mut data = vec![0.0; n_batches * n_groups * n_genes];
        data.par_chunks_mut(n_genes)
            .enumerate()
            .for_each(|(i, row)| {
                let (b, k) = (i / n_groups, i % n_groups);
                for (g, x) in row.iter_mut().enumerate() {
                    let batch = genes.batch_fac.get(b).map_or(1.0, |v| v[g]);
                    let de = genes.de_fac.get(k).map_or(1.0, |v| v[g]);
                    *x = genes.gene_mean[g] * batch * de;
                }
                let total: f64 = row.iter().sum();
                row.iter_mut().for_each(|x| *x /= total);
            });
        Self {
            data,
            n_genes,
            n_groups,
        }
    }

    /// Profile of one (batch, group) pair.
    ///
    /// ### Params
    ///
    /// * `batch` - Batch index
    /// * `group` - Group index (0 in single mode)
    ///
    /// ### Returns
    ///
    /// The `n_genes` proportions.
    #[inline]
    pub fn get(&self, batch: usize, group: usize) -> &[f64] {
        let i = batch * self.n_groups + group;
        &self.data[i * self.n_genes..(i + 1) * self.n_genes]
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn test_lnorm_factors_none_selected_are_one() {
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let f = get_lnorm_factors(&mut rng, 1000, 0.0, 0.5, 1.0, 0.5).unwrap();
        assert!(f.iter().all(|&x| x == 1.0));
    }

    #[test]
    fn test_lnorm_factors_down_prob_sets_share_below_one() {
        let mut rng = ChaCha8Rng::seed_from_u64(2);
        // Loc near 0 and a wide scale: half the raw draws are below one, so
        // the direction flip must be in place for the share to come out right.
        let f = get_lnorm_factors(&mut rng, 100_000, 1.0, 0.2, 0.0, 1.0).unwrap();
        let below = f.iter().filter(|&&x| x < 1.0).count() as f64 / f.len() as f64;
        assert_relative_eq!(below, 0.2, epsilon = 0.01);
    }

    #[test]
    fn test_median_even_and_odd() {
        assert_eq!(median(&[3.0, 1.0, 2.0]), 2.0);
        assert_eq!(median(&[4.0, 1.0, 3.0, 2.0]), 2.5);
    }

    #[test]
    fn test_profiles_rows_sum_to_one() {
        let params = SplatParams {
            n_genes: 500,
            batch_cells: vec![10, 10],
            group_prob: vec![0.5, 0.5],
            method: Method::Groups,
            ..Default::default()
        }
        .resolve()
        .unwrap();
        let mut rng = ChaCha8Rng::seed_from_u64(3);
        let genes = GeneTruth::simulate(&params, &mut rng).unwrap();
        let prof = Profiles::new(&genes, 2, 2);
        for b in 0..2 {
            for k in 0..2 {
                assert_relative_eq!(prof.get(b, k).iter().sum::<f64>(), 1.0, epsilon = 1e-12);
            }
        }
    }
}
