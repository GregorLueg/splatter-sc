# Changelog

## 0.1.1

### Output

- `output.h5_compression` is split in two, each defaulting to what the
  original tool writes. Configs using the old key are rejected.
- `tenx_h5_compression`: gzip level 4 by default, applied to every dataset in
  `matrix.h5` but `features/_all_tag_keys`. `null` for uncompressed.
- `h5ad_compression`: gzip level 4 by default, applied to every dataset in
  `DGE.h5ad`, as scanpy's `sc.write`. `null` for uncompressed.
- `DGE.h5ad` mirrors `anndata.write_h5ad(compression="gzip")`: no shuffle,
  h5py's guessed chunk length per dataset, `X/*` resizable, `X/indptr` int32
  while nnz fits, categorical codes int8/int16/int32 as pandas picks them
  (were int64 `indptr`, int32 codes, uncompressed `obs`/`var`).
- The `X/data` and `X/indices` chunk length comes from an nnz estimate over
  the first 256 cells, so it can differ from anndata's by a few percent
  (1.5% at 10k cells). Values are unaffected.
- `matrix.h5` mirrors Cell Ranger's h5py writer (as in Space Ranger 2.1):
  `data`, `indices`, `indptr` and `shape` resizable in 80000-element chunks
  with shuffle plus gzip; strings NUL-padded and as wide as their longest
  entry (were `|S31`), gzipped without shuffle; root attributes `filetype`,
  `version`, `software_version`, `chemistry_description`, `library_ids`,
  `original_gem_groups` added. Names are no longer capped at 31 characters.
- HDF5 compression moved off the writer thread: the encoder shuffles and
  zlib-compresses whole HDF5 chunks in parallel on the simulation threads,
  and the writer stores them with `H5Dwrite_chunk`.
- `output.parse_gzip` writes the Parse matrix as `DGE.mtx.gz`, compressed in
  parallel on the simulation threads. Off by default.

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
