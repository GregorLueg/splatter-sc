# splatter-sc

A streaming Rust port of the Splat simulator from
[splatter](https://github.com/Oshlack/splatter), as a command line binary.

splatter is great for small benchmarks, but it builds the whole count matrix
(plus every intermediate assay) in memory. Ask it for a few hundred thousand
cells and that gets painful fast. `splatter-sc` draws the per-gene state once,
then simulates cells in chunks in parallel and streams them straight to disk.
Peak memory is the per-gene state plus a window of chunks per thread. Cells
cost a few bytes of metadata each, nothing more.

On an M1 Max (10 threads), 10k cells x 20k genes in three groups and two
batches, writing the Parse mtx and an h5ad:

```
Done in 2.55s: 82553277 nonzeros (setup 121.41ms, simulate 1.76s, write 267.48ms, finish 530.67ms)
```

## Installation

Needs Rust 1.88+ (edition 2024, let chains). HDF5 is either linked from the system or
built from source:

```sh
# links an existing libhdf5 (found via HDF5_DIR or pkg-config)
cargo install --git https://github.com/GregorLueg/splatter-sc

# no system HDF5? build it from source
cargo install --git https://github.com/GregorLueg/splatter-sc --features hdf5-static
```

## Usage

```sh
splatter-sc params.json
```

Thread count follows `RAYON_NUM_THREADS`. A minimal parameter file:

```json
{
  "nGenes": 20000,
  "batchCells": [5000, 5000],
  "method": "groups",
  "group.prob": [0.5, 0.3, 0.2],
  "de.prob": 0.1,
  "dropout.type": "experiment",
  "seed": 42,
  "output": {
    "dir": "splat_out",
    "layouts": ["parse", "tenx_mtx", "h5ad", "tenx_h5"],
    "h5ad_compression": 4,
    "tenx_h5_compression": 4,
    "parse_gzip": false
  }
}
```

Parameters use splatter's own names and defaults (`mean.shape`, `lib.loc`,
`bcv.df`, ...). Anything you leave out takes the value from splatter's
`SplatParams` prototype. Unknown keys are an error, so typos don't slip
through silently.

### Feeding in a splatter fit

Already have a `SplatParams` from `splatEstimate()`? Dump it from R and hand it
over:

```r
p <- splatter::splatEstimate(counts)
x <- splatter::getParams(p, methods::slotNames(p))
x$method <- "groups"
writeLines(jsonlite::toJSON(x, auto_unbox = TRUE, digits = NA), "params.json")
```

The `path.*` keys are accepted and ignored. Add an `output` block if you don't
want the defaults (`splat_out/`, Parse layout only).

### Output

Every run writes three sidecars into `output.dir`:

- `params_used.json`: the resolved parameters, including the seed. Rerunning
  on this file reproduces the output exactly.
- `cells_truth.tsv.gz`: per-cell truth (`Cell`, `Batch`, `Group`, `ExpLibSize`).
- `genes_truth.tsv.gz`: per-gene truth (`BaseGeneMean`, `OutlierFactor`,
  `GeneMean`, `BatchFacBatch*`, `DEFacGroup*`, `BCVChiFac`), as splatter's
  `rowData`.

Plus one set of files per layout:

| Layout | Files |
|---|---|
| `parse` | `DGE.mtx` or `DGE.mtx.gz` (cells x genes), `cell_metadata.csv`, `all_genes.csv` |
| `tenx_mtx` | `tenx/matrix.mtx.gz` (genes x cells), `barcodes.tsv.gz`, `features.tsv.gz` |
| `h5ad` | `DGE.h5ad`, float32 CSR with cells as rows, truth in `obs` and `var` |
| `tenx_h5` | `matrix.h5`, Cell Ranger v3 layout (genes x cells CSC under `/matrix`) |

Compression follows what the original tools write:

- `tenx_mtx` is always gzipped (level 6, like the `gzip` CLI).
- `tenx_h5_compression` sets the gzip level (0-9) of every dataset in
  `matrix.h5` except `features/_all_tag_keys`, which stays contiguous.
  Defaults to 4 (h5py's gzip level); `null` for uncompressed.
- `h5ad_compression` sets the gzip level of every dataset in `DGE.h5ad`.
  Defaults to 4, as scanpy's `sc.write` (anndata's `write_h5ad` itself
  defaults to uncompressed); `null` for uncompressed.
- All compression runs in parallel on the simulation threads; the writer
  thread only stores finished bytes.
- `parse_gzip: true` writes `DGE.mtx.gz` instead of `DGE.mtx` (level 6,
  compressed in parallel). Off by default, since Parse's own pipeline writes
  it plain.

The HDF5 files mirror the originals down to dtypes, filters and chunk shapes,
checked dataset by dataset with h5py:

- `DGE.h5ad` matches `anndata.write_h5ad(compression="gzip")`: no shuffle,
  chunk lengths from h5py's `guess_chunk`, `X/*` resizable, `X/indptr` int32
  while nnz fits (int64 beyond), categorical codes in pandas' narrowest int
  (int8 below 127 categories).
- `matrix.h5` matches Cell Ranger's h5py writer, as in Space Ranger 2.1 files:
  `data`, `indices`, `indptr` and `shape` resizable in 80000-element chunks
  with shuffle plus gzip; strings NUL-padded fixed-width ASCII as wide as
  their longest entry, gzipped without shuffle; root attributes `filetype`,
  `version`, `software_version`, `chemistry_description`, `library_ids` and
  `original_gem_groups`. Older Cell Ranger releases (e.g. 4.0) wrote gzip 1,
  65536-element chunks and `|S256` strings instead.

Known differences:

- **h5ad `X/data` and `X/indices` chunk length.** h5py derives it from the
  final nnz, which a streaming writer does not know up front. splatter-sc
  extrapolates nnz from the first 256 cells, so the chunk length is typically
  within a few percent of what anndata would pick (10k cells x 19k genes:
  10591 vs 10436, 1.5%). Near one of h5py's halving steps the estimate can land
  on the other side and give a chunk twice or half as long. The stored values
  are identical either way; only the chunk layout, and so read and compression
  performance, can differ slightly.
- **h5ad `X/indices` dtype.** Always int32. anndata switches it to int64
  together with `indptr` once nnz exceeds 2^31 - 1; readers accept both.
- **h5ad `_index` with pandas 3.** anndata on pandas 3 writes the index as a
  nullable string array (`mask` + `values`); splatter-sc writes the plain
  string array that anndata on pandas 2 writes.

## How close is this to splatter?

Close in distribution, not bit for bit. The random stream is ChaCha8, not R's
Mersenne Twister, so the same seed gives different counts than in R. Same seed
gives the same output here, whatever the thread count.

The test suite checks this against fixtures generated with splatter 1.37.1:

1. R's parameter dumps load and expand to R's values, and the binary accepts
   exactly the parameter sets R accepts.
2. Given R's gene and cell draws, `BaseCellMeans` and `BCV` match R to a
   relative tolerance of 1e-10.
3. With R's parameters, gene means, BCV factors, library sizes, batch and DE
   factors, per-gene variance and per-cell totals and detected genes match
   R's output under two-sample KS tests.

Run them with `cargo test --release`. The large fixtures sit behind
`--features large-scale-tests`.

## What's not ported (yet)

- `method = "paths"` (trajectories).
- `dropout.type = "cell"`.
- Anything outside `splatSimulate`: no `splatEstimate`, no other simulators.

A couple of things go the other way. `batch.groupProb` takes one row of group
probabilities per batch, so batches can differ in composition (zeros allowed),
which splatter can't do. Validation is a touch stricter in places where R would
happily produce garbage: `mean.shape`, `mean.rate` and `bcv.df` must be
strictly positive.

## Reference

Zappia L, Phipson B, Oshlack A. Splatter: simulation of single-cell RNA
sequencing data. *Genome Biology* 18, 174 (2017).
[doi:10.1186/s13059-017-1305-0](https://doi.org/10.1186/s13059-017-1305-0)

## Licence

GPL-3.0-or-later, as splatter.
