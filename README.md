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
    "h5_compression": 4
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
| `parse` | `DGE.mtx` (cells x genes), `cell_metadata.csv`, `all_genes.csv` |
| `tenx_mtx` | `tenx/matrix.mtx.gz` (genes x cells), `barcodes.tsv.gz`, `features.tsv.gz` |
| `h5ad` | `DGE.h5ad`, float32 CSR with cells as rows, truth in `obs` and `var` |
| `tenx_h5` | `matrix.h5`, Cell Ranger v3 layout |

`h5_compression` sets the gzip level (0-9) of the HDF5 count datasets. Leave
it out for uncompressed. HDF5 compresses on the single writer thread, not the
simulation pool.

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

## What's not ported

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
