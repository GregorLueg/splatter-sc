//! Parity against R splatter, using the fixtures written by
//! `tests/r_parity/make_reference.R`.
//!
//! Three tiers:
//! 1. Parameters: R's JSON dumps load and expand to R's values, and Rust
//!    accepts exactly the parameter sets R accepts.
//! 2. Exact: given R's gene truth and cell metadata, the deterministic steps
//!    (`BaseCellMeans`, `BCV`) reproduce R's assays.
//! 3. Distributional: the Rust simulator with R's parameters matches R's
//!    output in distribution (two-sample Kolmogorov-Smirnov), as the random
//!    streams differ. Runs on the `medium` fixtures; the `large` ones need
//!    `--features large-scale-tests` (best with `--release`).

use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use flate2::read::GzDecoder;
use rayon::prelude::*;
use splatter_sc::Simulation;
use splatter_sc::cells::{CellMeta, bcv};
use splatter_sc::genes::{GeneTruth, Profiles};
use splatter_sc::params::{Method, SplatParams};

/// Settings written by the R script.
const SETTINGS: &[&str] = &[
    "single",
    "groups_batches",
    "dropout",
    "lib_norm",
    "dropout_batch",
    "dropout_group",
    "bcv_inf",
];

/// Relative tolerance of the exact tier.
const EXACT_RTOL: f64 = 1e-10;

/// Significance level of every KS test. Seeds are fixed, so a test either
/// always passes or always fails; this sets how big a real difference has to
/// be to be caught.
const KS_ALPHA: f64 = 0.001;

/////////////
// Helpers //
/////////////

/// Fixture directory of a setting and size.
///
/// ### Params
///
/// * `setting` - Setting name
/// * `size` - `small`, `medium` or `large`
///
/// ### Returns
///
/// The directory.
fn fixture(setting: &str, size: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/r_parity")
        .join(setting)
        .join(size)
}

/// Lines of a gzipped text file.
///
/// ### Params
///
/// * `path` - File
///
/// ### Returns
///
/// All lines.
fn gz_lines(path: &Path) -> Vec<String> {
    let f = std::fs::File::open(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    BufReader::new(GzDecoder::new(f))
        .lines()
        .map(Result::unwrap)
        .collect()
}

/// A tab-separated table with a header.
struct Table {
    /// Column names
    header: Vec<String>,
    /// Rows of fields
    rows: Vec<Vec<String>>,
}

impl Table {
    /// Read a gzipped TSV.
    ///
    /// ### Params
    ///
    /// * `path` - File
    ///
    /// ### Returns
    ///
    /// The table.
    fn read(path: &Path) -> Self {
        let lines = gz_lines(path);
        let split = |l: &String| l.split('\t').map(str::to_string).collect::<Vec<_>>();
        Self {
            header: split(&lines[0]),
            rows: lines[1..].iter().map(split).collect(),
        }
    }

    /// Whether a column exists.
    fn has(&self, name: &str) -> bool {
        self.header.iter().any(|h| h == name)
    }

    /// A column as strings.
    fn str_col(&self, name: &str) -> Vec<String> {
        let j = self
            .header
            .iter()
            .position(|h| h == name)
            .unwrap_or_else(|| panic!("no column {name}"));
        self.rows.iter().map(|r| r[j].clone()).collect()
    }

    /// A column parsed as floats.
    fn col(&self, name: &str) -> Vec<f64> {
        self.str_col(name)
            .iter()
            .map(|x| x.parse().unwrap())
            .collect()
    }

    /// Columns `prefix1`, `prefix2`, ... while they exist.
    fn numbered(&self, prefix: &str) -> Vec<Vec<f64>> {
        (1..)
            .map(|i| format!("{prefix}{i}"))
            .take_while(|n| self.has(n))
            .map(|n| self.col(&n))
            .collect()
    }
}

/// A gzipped space-separated dense matrix, genes x cells.
///
/// ### Params
///
/// * `path` - File
///
/// ### Returns
///
/// Rows of values.
fn read_matrix(path: &Path) -> Vec<Vec<f64>> {
    gz_lines(path)
        .iter()
        .map(|l| l.split(' ').map(|x| x.parse().unwrap()).collect())
        .collect()
}

/// Zero-based index from a splatter label such as `Batch2`.
///
/// ### Params
///
/// * `label` - Label
/// * `prefix` - `Batch` or `Group`
///
/// ### Returns
///
/// The index.
fn label_index(label: &str, prefix: &str) -> u32 {
    label.strip_prefix(prefix).unwrap().parse::<u32>().unwrap() - 1
}

/// Load and resolve R's parameter dump of a fixture.
///
/// ### Params
///
/// * `dir` - Fixture directory
///
/// ### Returns
///
/// Resolved parameters.
fn load_params(dir: &Path) -> SplatParams {
    SplatParams::from_json_file(&dir.join("params.json"))
        .unwrap()
        .resolve()
        .unwrap()
}

/// Two-sample Kolmogorov-Smirnov statistic.
///
/// ### Params
///
/// * `a` - First sample
/// * `b` - Second sample
///
/// ### Returns
///
/// `sup |F_a - F_b|`.
fn ks_stat(a: &[f64], b: &[f64]) -> f64 {
    let mut a = a.to_vec();
    let mut b = b.to_vec();
    a.sort_unstable_by(f64::total_cmp);
    b.sort_unstable_by(f64::total_cmp);
    let (n, m) = (a.len() as f64, b.len() as f64);
    let (mut i, mut j, mut d) = (0, 0, 0.0f64);
    while i < a.len() && j < b.len() {
        let x = a[i].min(b[j]);
        while i < a.len() && a[i] <= x {
            i += 1;
        }
        while j < b.len() && b[j] <= x {
            j += 1;
        }
        d = d.max((i as f64 / n - j as f64 / m).abs());
    }
    d
}

/// Assert two samples pass a KS test at [`KS_ALPHA`].
///
/// ### Params
///
/// * `what` - Label for the message
/// * `r` - R's sample
/// * `rust` - The port's sample
fn assert_ks(what: &str, r: &[f64], rust: &[f64]) {
    let (n, m) = (r.len() as f64, rust.len() as f64);
    let crit = (-(KS_ALPHA / 2.0).ln() / 2.0).sqrt() * ((n + m) / (n * m)).sqrt();
    let d = ks_stat(r, rust);
    eprintln!("  KS {what}: D = {d:.4} (crit {crit:.4}, n = {n}, m = {m})");
    assert!(d < crit, "{what}: KS D = {d:.4} >= {crit:.4}");
}

/// Assert two proportions agree within four standard errors.
///
/// ### Params
///
/// * `what` - Label for the message
/// * `r` - R's sample of booleans
/// * `rust` - The port's sample of booleans
fn assert_share(what: &str, r: &[bool], rust: &[bool]) {
    let p = |v: &[bool]| v.iter().filter(|&&x| x).count() as f64 / v.len() as f64;
    let (pr, pu) = (p(r), p(rust));
    let pool = 0.5 * (pr + pu);
    let se = (pool * (1.0 - pool) * (1.0 / r.len() as f64 + 1.0 / rust.len() as f64)).sqrt();
    eprintln!(
        "  share {what}: R {pr:.4}, Rust {pu:.4}, 4 se {:.4}",
        4.0 * se
    );
    assert!(
        (pr - pu).abs() <= 4.0 * se + 1e-12,
        "{what}: R {pr:.4} vs Rust {pu:.4}"
    );
}

/// Log of the factors that differ from one.
///
/// ### Params
///
/// * `v` - Factors
///
/// ### Returns
///
/// `ln(f)` for every `f != 1`.
fn log_selected(v: &[f64]) -> Vec<f64> {
    v.iter().filter(|&&x| x != 1.0).map(|x| x.ln()).collect()
}

/////////////////////
// Tier 1: params  //
/////////////////////

#[test]
fn test_r_parity_params_expand_like_r() {
    for setting in SETTINGS {
        for size in ["small", "medium", "large"] {
            let dir = fixture(setting, size);
            let p = load_params(&dir);
            let r: serde_json::Value = serde_json::from_reader(
                std::fs::File::open(dir.join("params_expanded.json")).unwrap(),
            )
            .unwrap();
            // jsonlite writes Inf as the string "Inf".
            let num = |x: &serde_json::Value| {
                x.as_f64().unwrap_or_else(|| {
                    assert_eq!(x.as_str(), Some("Inf"));
                    f64::INFINITY
                })
            };
            let vec = |k: &str| -> Vec<f64> {
                match &r[k] {
                    serde_json::Value::Array(a) => a.iter().map(num).collect(),
                    x => vec![num(x)],
                }
            };
            assert_eq!(p.n_genes as f64, r["nGenes"].as_f64().unwrap());
            assert_eq!(p.n_cells() as f64, r["nCells"].as_f64().unwrap());
            assert_eq!(p.n_batches() as f64, r["nBatches"].as_f64().unwrap());
            assert_eq!(p.n_groups() as f64, r["nGroups"].as_f64().unwrap());
            assert_eq!(p.seed() as f64, r["seed"].as_f64().unwrap());
            let bc: Vec<f64> = p.batch_cells.iter().map(|&x| x as f64).collect();
            assert_eq!(bc, vec("batchCells"));
            assert_eq!(p.batch_fac_loc, vec("batch.facLoc"));
            assert_eq!(p.batch_fac_scale, vec("batch.facScale"));
            assert_eq!(p.group_prob, vec("group.prob"));
            assert_eq!(p.de_prob, vec("de.prob"));
            assert_eq!(p.de_down_prob, vec("de.downProb"));
            assert_eq!(p.de_fac_loc, vec("de.facLoc"));
            assert_eq!(p.de_fac_scale, vec("de.facScale"));
            assert_eq!(p.dropout_mid, vec("dropout.mid"));
            assert_eq!(p.dropout_shape, vec("dropout.shape"));
            assert_eq!(
                vec![p.lib_loc, p.lib_scale],
                [vec("lib.loc"), vec("lib.scale")].concat()
            );
            assert_eq!(
                vec![p.bcv_common, p.bcv_df],
                [vec("bcv.common"), vec("bcv.df")].concat()
            );
            assert_eq!(p.lib_norm, r["lib.norm"].as_bool().unwrap());
        }
    }
}

#[test]
fn test_r_parity_validity_matches_r() {
    let path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/r_parity/validity_cases.json");
    let cases: Vec<serde_json::Value> =
        serde_json::from_reader(std::fs::File::open(path).unwrap()).unwrap();
    for case in cases {
        let name = case["name"].as_str().unwrap();
        let r_valid = case["valid"].as_bool().unwrap();
        let rust_valid = serde_json::from_value::<SplatParams>(case["params"].clone())
            .map_err(|e| e.to_string())
            .and_then(|p| p.resolve().map_err(|e| e.to_string()));
        eprintln!("  {name}: R valid = {r_valid}, Rust = {rust_valid:?}");
        assert_eq!(rust_valid.is_ok(), r_valid, "{name}");
    }
}

////////////////////
// Tier 2: exact  //
////////////////////

/// Gene truth and cell metadata rebuilt from R's `rowData` / `colData`.
///
/// ### Params
///
/// * `dir` - Fixture directory
/// * `params` - Resolved parameters
///
/// ### Returns
///
/// `(genes, cells)`; `bcv_chi_fac` is the `BCVChiFac` column the R script
/// recovers from cell 1.
fn truth_from_r(dir: &Path, params: &SplatParams) -> (GeneTruth, CellMeta) {
    let rd = Table::read(&dir.join("rowData.tsv.gz"));
    let cd = Table::read(&dir.join("colData.tsv.gz"));
    let genes = GeneTruth {
        base_gene_mean: rd.col("BaseGeneMean"),
        outlier_factor: rd.col("OutlierFactor"),
        gene_mean: rd.col("GeneMean"),
        batch_fac: rd.numbered("BatchFacBatch"),
        de_fac: rd.numbered("DEFacGroup"),
        bcv_chi_fac: rd.col("BCVChiFac"),
    };
    let cells = CellMeta {
        batch: cd
            .str_col("Batch")
            .iter()
            .map(|b| label_index(b, "Batch"))
            .collect(),
        group: if cd.has("Group") {
            cd.str_col("Group")
                .iter()
                .map(|g| label_index(g, "Group"))
                .collect()
        } else {
            vec![0; params.n_cells()]
        },
        exp_lib_size: cd.col("ExpLibSize"),
    };
    (genes, cells)
}

#[test]
fn test_r_parity_base_cell_means_and_bcv_exact() {
    for setting in SETTINGS {
        let dir = fixture(setting, "small");
        let params = load_params(&dir);
        let (genes, cells) = truth_from_r(&dir, &params);
        let n_groups = if params.method == Method::Groups {
            params.n_groups()
        } else {
            1
        };
        let profiles = Profiles::new(&genes, params.n_batches(), n_groups);

        let r_base = read_matrix(&dir.join("BaseCellMeans.txt.gz"));
        let r_bcv = read_matrix(&dir.join("BCV.txt.gz"));
        let mut worst = (0.0f64, 0.0f64, 0.0f64);

        for g in 0..params.n_genes {
            // One chi-squared draw per gene in R: recover it from cell 0,
            // then every other cell must agree.
            let base0 = r_base[g][0];
            let chi = r_bcv[g][0] / (params.bcv_common + 1.0 / base0.sqrt());
            for c in 0..params.n_cells() {
                let prof = profiles.get(cells.batch[c] as usize, cells.group[c] as usize);
                let base = cells.exp_lib_size[c] * prof[g];
                let rel_base = (base - r_base[g][c]).abs() / r_base[g][c];
                let chi_c = r_bcv[g][c] / (params.bcv_common + 1.0 / r_base[g][c].sqrt());
                let rel_chi = (chi_c - chi).abs() / chi;
                let b = bcv(params.bcv_common, base, chi);
                let rel_bcv = (b - r_bcv[g][c]).abs() / r_bcv[g][c];
                worst = (
                    worst.0.max(rel_base),
                    worst.1.max(rel_chi),
                    worst.2.max(rel_bcv),
                );
            }
        }
        eprintln!(
            "  {setting}: max rel diff BaseCellMeans {:.2e}, chi per gene {:.2e}, BCV {:.2e}",
            worst.0, worst.1, worst.2
        );
        assert!(worst.0 < EXACT_RTOL, "{setting}: BaseCellMeans");
        assert!(
            worst.1 < EXACT_RTOL,
            "{setting}: chi-squared factor not constant per gene"
        );
        assert!(worst.2 < EXACT_RTOL, "{setting}: BCV");
    }
}

////////////////////////////
// Tier 3: distributional //
////////////////////////////

/// The same summaries the R script writes, from a Rust simulation.
struct Summaries {
    /// Per gene mean count
    mean: Vec<f64>,
    /// Per gene sample variance (n - 1)
    var: Vec<f64>,
    /// Per gene zero fraction
    zero_frac: Vec<f64>,
    /// Per group, per gene mean count
    group_mean: Vec<Vec<f64>>,
    /// Per group, per gene sample variance
    group_var: Vec<Vec<f64>>,
    /// Number of cells per group
    group_n: Vec<f64>,
    /// Per cell total count
    total: Vec<f64>,
    /// Per cell detected genes
    detected: Vec<f64>,
}

/// Mean and sample variance from sums.
///
/// ### Params
///
/// * `s` - Sum
/// * `q` - Sum of squares
/// * `n` - Count
///
/// ### Returns
///
/// `(mean, variance)`.
fn mean_var(s: f64, q: f64, n: f64) -> (f64, f64) {
    (s / n, (q - s * s / n) / (n - 1.0))
}

/// Simulate in memory and summarise.
///
/// ### Params
///
/// * `sim` - The simulation
///
/// ### Returns
///
/// The summaries.
fn summarise(sim: &Simulation) -> Summaries {
    let n_genes = sim.params.n_genes;
    let n_groups = sim.params.n_groups();
    let mut sum = vec![vec![0.0; n_genes]; n_groups];
    let mut sumsq = vec![vec![0.0; n_genes]; n_groups];
    let mut nz = vec![0.0; n_genes];
    let mut group_n = vec![0.0; n_groups];
    let mut total = Vec::new();
    let mut detected = Vec::new();

    let chunks: Vec<_> = (0..sim.n_chunks())
        .into_par_iter()
        .map(|i| sim.chunk(i).unwrap())
        .collect();
    for chunk in &chunks {
        for (j, w) in chunk.indptr.windows(2).enumerate() {
            let k = sim.cells.group[chunk.start + j] as usize;
            group_n[k] += 1.0;
            let mut t = 0.0;
            for e in w[0]..w[1] {
                let (g, x) = (chunk.indices[e] as usize, chunk.counts[e] as f64);
                sum[k][g] += x;
                sumsq[k][g] += x * x;
                nz[g] += 1.0;
                t += x;
            }
            total.push(t);
            detected.push((w[1] - w[0]) as f64);
        }
    }
    let n: f64 = group_n.iter().sum();
    let (mut mean, mut var) = (Vec::new(), Vec::new());
    for g in 0..n_genes {
        let s: f64 = sum.iter().map(|v| v[g]).sum();
        let q: f64 = sumsq.iter().map(|v| v[g]).sum();
        let (m, v) = mean_var(s, q, n);
        mean.push(m);
        var.push(v);
    }
    let per_group = |pick: fn((f64, f64)) -> f64| -> Vec<Vec<f64>> {
        (0..n_groups)
            .map(|k| {
                (0..n_genes)
                    .map(|g| pick(mean_var(sum[k][g], sumsq[k][g], group_n[k])))
                    .collect()
            })
            .collect()
    };
    Summaries {
        group_mean: per_group(|x| x.0),
        group_var: per_group(|x| x.1),
        mean,
        var,
        zero_frac: nz.iter().map(|z| 1.0 - z / n).collect(),
        group_n,
        total,
        detected,
    }
}

/// Mean and standard deviation.
///
/// ### Params
///
/// * `z` - Values
///
/// ### Returns
///
/// `(mean, sd)`.
fn mean_sd(z: &[f64]) -> (f64, f64) {
    let n = z.len() as f64;
    let mean = z.iter().sum::<f64>() / n;
    let sd = (z.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();
    (mean, sd)
}

/// Assert paired z-scores of R against Rust behave like those of two Rust
/// runs on the same truth: mean within four standard errors of zero, and
/// standard deviation within four standard errors of the Rust-vs-Rust one,
/// taking `sd / sqrt(2 (n - 1))` as the standard error of a sample sd. The null
/// sd is not one, since studentised means of skewed counts are
/// under-dispersed.
///
/// ### Params
///
/// * `what` - Label for the message
/// * `z` - R vs Rust z-scores, one per gene
/// * `z_null` - Rust vs Rust (different seed) z-scores
fn assert_z(what: &str, z: &[f64], z_null: &[f64]) {
    let (mean, sd) = mean_sd(z);
    let (_, sd_null) = mean_sd(z_null);
    let bound = 4.0 / (z.len() as f64).sqrt();
    let sd_bound = 4.0 * sd_null / (z_null.len() as f64 - 1.0).sqrt();
    eprintln!(
        "  paired z {what}: mean {mean:.4} (bound {bound:.4}), sd {sd:.4}, null sd {sd_null:.4} (bound {sd_bound:.4}), n = {}",
        z.len()
    );
    assert!(mean.abs() < bound, "{what}: mean z = {mean:.4}");
    assert!(
        (sd - sd_null).abs() < sd_bound,
        "{what}: sd z = {sd:.4} vs null {sd_null:.4}"
    );
}

/// Paired z-scores of two sets of per-gene means.
///
/// ### Params
///
/// * `m_r` - R means
/// * `v_r` - R variances
/// * `m_u` - Rust means
/// * `v_u` - Rust variances
/// * `n` - Cells per side
///
/// ### Returns
///
/// `(m_u - m_r) / sqrt((v_r + v_u) / n)` for genes with nonzero variance.
fn z_means(m_r: &[f64], v_r: &[f64], m_u: &[f64], v_u: &[f64], n: f64) -> Vec<f64> {
    (0..m_r.len())
        .filter(|&g| v_r[g] + v_u[g] > 0.0)
        .map(|g| (m_u[g] - m_r[g]) / ((v_r[g] + v_u[g]) / n).sqrt())
        .collect()
}

/// Factors that differ from one, as booleans.
///
/// ### Params
///
/// * `v` - Factors
///
/// ### Returns
///
/// `f != 1` per factor.
fn selected(v: &[f64]) -> Vec<bool> {
    v.iter().map(|&x| x != 1.0).collect()
}

/// Generative tier: quantities that are independent across genes or cells
/// within a simulation, compared by KS between R's and Rust's own draws.
/// Per-gene count summaries are left out here: all genes of a run share the
/// normaliser and outlier median, so two runs differ by more than KS allows
/// even Rust against Rust.
///
/// ### Params
///
/// * `size` - Fixture size
fn generative_distributions(size: &str) {
    for setting in SETTINGS {
        eprintln!("{setting}:");
        let dir = fixture(setting, size);
        let sim = Simulation::new(load_params(&dir)).unwrap();
        let rd = Table::read(&dir.join("rowData.tsv.gz"));
        let cd = Table::read(&dir.join("colData.tsv.gz"));
        let cs = Table::read(&dir.join("cell_summary.tsv.gz"));

        let g = &sim.genes;
        assert_ks("BaseGeneMean", &rd.col("BaseGeneMean"), &g.base_gene_mean);
        assert_ks("GeneMean", &rd.col("GeneMean"), &g.gene_mean);
        assert_ks("BCVChiFac", &rd.col("BCVChiFac"), &g.bcv_chi_fac);
        let r_out = rd.col("OutlierFactor");
        assert_share("outliers", &selected(&r_out), &selected(&g.outlier_factor));
        assert_ks(
            "log OutlierFactor",
            &log_selected(&r_out),
            &log_selected(&g.outlier_factor),
        );
        assert_ks("ExpLibSize", &cd.col("ExpLibSize"), &sim.cells.exp_lib_size);
        assert_ks("cell total", &cs.col("Total"), &summarise(&sim).total);

        for (b, r_fac) in rd.numbered("BatchFacBatch").iter().enumerate() {
            let what = format!("log BatchFacBatch{}", b + 1);
            assert_ks(&what, &log_selected(r_fac), &log_selected(&g.batch_fac[b]));
        }
        for (k, r_fac) in rd.numbered("DEFacGroup").iter().enumerate() {
            let u_fac = &g.de_fac[k];
            let what = format!("DE genes Group{}", k + 1);
            assert_share(&what, &selected(r_fac), &selected(u_fac));
            let what = format!("log DEFacGroup{}", k + 1);
            assert_ks(&what, &log_selected(r_fac), &log_selected(u_fac));
        }
        if cd.has("Group") {
            let groups = |v: Vec<u32>| v.into_iter().map(f64::from).collect::<Vec<_>>();
            let r: Vec<u32> = cd
                .str_col("Group")
                .iter()
                .map(|x| label_index(x, "Group"))
                .collect();
            assert_ks(
                "group assignment",
                &groups(r),
                &groups(sim.cells.group.clone()),
            );
        }
    }
}

/// Per-gene count statistics of one run, from R's table or a Rust run.
struct GeneStats {
    /// Mean count
    mean: Vec<f64>,
    /// Sample variance
    var: Vec<f64>,
    /// Zero fraction
    zero_frac: Vec<f64>,
    /// Per group `(mean, variance)`
    groups: Vec<(Vec<f64>, Vec<f64>)>,
}

impl GeneStats {
    /// From R's `gene_summary.tsv.gz`.
    fn from_r(gs: &Table) -> Self {
        let groups = (1..)
            .take_while(|k| gs.has(&format!("MeanGroup{k}")))
            .map(|k| {
                (
                    gs.col(&format!("MeanGroup{k}")),
                    gs.col(&format!("VarGroup{k}")),
                )
            })
            .collect();
        Self {
            mean: gs.col("Mean"),
            var: gs.col("Var"),
            zero_frac: gs.col("ZeroFrac"),
            groups,
        }
    }

    /// From a Rust run.
    fn from_rust(s: &Summaries, with_groups: bool) -> Self {
        let groups = if with_groups {
            s.group_mean
                .iter()
                .cloned()
                .zip(s.group_var.iter().cloned())
                .collect()
        } else {
            Vec::new()
        };
        Self {
            mean: s.mean.clone(),
            var: s.var.clone(),
            zero_frac: s.zero_frac.clone(),
            groups,
        }
    }
}

/// Paired z-scores of zero fractions.
///
/// ### Params
///
/// * `a` - First run
/// * `b` - Second run
/// * `n` - Cells per run
///
/// ### Returns
///
/// `(b - a) / sqrt(2 p (1 - p) / n)` with pooled `p`, for genes with
/// `0 < p < 1`.
fn z_zero_frac(a: &[f64], b: &[f64], n: f64) -> Vec<f64> {
    (0..a.len())
        .filter_map(|g| {
            let p = 0.5 * (a[g] + b[g]);
            (p > 0.0 && p < 1.0).then(|| (b[g] - a[g]) / (2.0 * p * (1.0 - p) / n).sqrt())
        })
        .collect()
}

/// Count tier: Rust runs on R's exact gene truth (including the per-gene
/// chi-squared factor) and cell metadata, so only the per-entry gamma,
/// Poisson and dropout draws differ. Per-gene summaries are compared gene by
/// gene with paired z-scores against a Rust-vs-Rust null on the same truth;
/// per-cell ones by KS.
///
/// ### Params
///
/// * `size` - Fixture size
fn counts_on_r_truth(size: &str) {
    for setting in SETTINGS {
        eprintln!("{setting}:");
        let dir = fixture(setting, size);
        let params = load_params(&dir);
        let (genes, cells) = truth_from_r(&dir, &params);
        let with_groups = params.method == Method::Groups;
        let mut params2 = params.clone();
        params2.seed = Some(params.seed() + 1);
        let sim = Simulation::from_parts(params, genes.clone(), cells.clone());
        let sim2 = Simulation::from_parts(params2, genes, cells);

        let gs = Table::read(&dir.join("gene_summary.tsv.gz"));
        let cs = Table::read(&dir.join("cell_summary.tsv.gz"));
        let ours = summarise(&sim);
        let r = GeneStats::from_r(&gs);
        let u = GeneStats::from_rust(&ours, with_groups);
        let u2 = GeneStats::from_rust(&summarise(&sim2), with_groups);
        let n = sim.cells.n_cells() as f64;

        assert_z(
            "gene mean",
            &z_means(&r.mean, &r.var, &u.mean, &u.var, n),
            &z_means(&u2.mean, &u2.var, &u.mean, &u.var, n),
        );
        assert_z(
            "gene zero fraction",
            &z_zero_frac(&r.zero_frac, &u.zero_frac, n),
            &z_zero_frac(&u2.zero_frac, &u.zero_frac, n),
        );
        assert_eq!(r.groups.len(), u.groups.len());
        for k in 0..r.groups.len() {
            let ((rm, rv), (um, uv), (um2, uv2)) = (&r.groups[k], &u.groups[k], &u2.groups[k]);
            let n_k = ours.group_n[k];
            assert_z(
                &format!("gene mean Group{}", k + 1),
                &z_means(rm, rv, um, uv, n_k),
                &z_means(um2, uv2, um, uv, n_k),
            );
        }

        assert_ks("gene variance", &r.var, &u.var);
        assert_ks("cell total", &cs.col("Total"), &ours.total);
        assert_ks("cell detected", &cs.col("Detected"), &ours.detected);
    }
}

#[test]
fn test_r_parity_generative_distributions() {
    generative_distributions("medium");
}

#[test]
fn test_r_parity_counts_on_r_truth() {
    counts_on_r_truth("medium");
}

#[cfg(feature = "large-scale-tests")]
#[test]
fn test_r_parity_generative_distributions_large() {
    generative_distributions("large");
}

#[cfg(feature = "large-scale-tests")]
#[test]
fn test_r_parity_counts_on_r_truth_large() {
    counts_on_r_truth("large");
}
