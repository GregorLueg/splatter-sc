# Changelog

## 0.1.0

First release. A fresh Rust port of splatter's Splat simulator, shipped as the
`splatter-sc` binary.

### Simulation

- Splat `single` and `groups` modes with batches, outlier genes, DE, BCV,
  `lib.norm` and dropout (`none`, `experiment`, `batch`, `group`).
- Parameters read from JSON with splatter's names and defaults. A `SplatParams`
  dump from `jsonlite::toJSON(getParams(...))` loads as is, and validation
  follows splatter's `setValidity`.
- `batch.groupProb`: per-batch group probabilities, so batch composition can
  differ. Not in splatter.
- Cells are simulated in 256-cell chunks across rayon threads and streamed to
  disk. Peak memory doesn't grow with the cell count.
- ChaCha8 streams per seed: same seed, same output, whatever the thread count.
  Does not reproduce R's random stream.
- Low-mean counts are drawn by negative binomial inversion instead of gamma
  plus Poisson.

### Output

- Parse layout (`DGE.mtx`, `cell_metadata.csv`, `all_genes.csv`).
- 10x mtx directory (`tenx/matrix.mtx.gz`, `barcodes.tsv.gz`,
  `features.tsv.gz`).
- `DGE.h5ad` (float32 CSR, truth in `obs` and `var`) and Cell Ranger v3
  `matrix.h5`, with optional gzip via `output.h5_compression`.
- Ground truth sidecars on every run: `params_used.json`,
  `cells_truth.tsv.gz`, `genes_truth.tsv.gz`.

### Not ported (yet)

- `method = "paths"` and `dropout.type = "cell"`; both are rejected.
