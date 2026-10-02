# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A streaming Rust port of splatter's Splat simulator (`splatSimulate`), shipped as a binary: `splatter-sc <params.json>`. The library target exists for the binary and the tests; it is not a stable API. Paths (`method = "paths"`) and `dropout.type = "cell"` are not ported and are rejected in `resolve()`.

## Commands

```sh
cargo build --release                       # links system libhdf5 (HDF5_DIR or pkg-config)
cargo build --release --features hdf5-static  # builds HDF5 from source, as CI does
cargo test --release                        # unit, pipeline and medium R parity tiers
cargo test --release --features large-scale-tests  # adds the large R parity fixtures
cargo test --release --test r_parity test_r_parity_validity_matches_r  # one test
cargo clippy --all-targets
RAYON_NUM_THREADS=4 ./target/release/splatter-sc params.json
Rscript tests/r_parity/make_reference.R [extra R lib path]  # regenerate fixtures, from repo root
```

CI calls the reusable workflows in `GregorLueg/personal-actions` (`rust-test.yml`, `rust-release.yml`, `rust-binary-release.yml`) with `--features hdf5-static`. No Windows lane.

## Architecture

Pipeline in `src/lib.rs::run`:

1. `params.rs`: JSON with splatter's own key names (`mean.shape`, `batchCells`, ...). It is a superset of `jsonlite::toJSON(getParams(p, slotNames(p)), auto_unbox = TRUE, digits = NA)`, so an R `SplatParams` dump loads directly. Extra keys: `method`, `output`, and `batch.groupProb` (not in splatter). `deny_unknown_fields` is on. `resolve()` mirrors splatter's `setParams`/`expandParams`/`setValidity`: expands length-one vectors, rescales `group.prob`, falls back from `groups` to `single` with one group, and draws and stores a seed if absent. Everything downstream assumes resolved params.
2. `genes.rs`: per-gene truth drawn once (base means, outliers, batch and DE factors, BCV chi-squared factor). `Profiles` precomputes the normalised mean profile per (batch, group) pair so the per-cell normaliser of `BaseCellMeans` is never recomputed.
3. `cells.rs`: per-cell metadata (batch, group, library size) drawn up front for all cells; `lib.norm` needs the global min. `simulate_chunk` then produces counts for a cell range as cell-major CSR. Low means (`<= INVERSION_MAX_MEAN`) draw the gamma-Poisson count by negative binomial inversion; higher ones use gamma plus Poisson.
4. `lib.rs::run`: windows of `threads * CHUNKS_PER_THREAD` chunks of `CELL_CHUNK = 256` cells are simulated and encoded in parallel, then sent over a depth-one channel to a single writer thread. Peak memory is independent of the cell count.
5. `output/`: one `Sink` per layout (`parse`, `tenx_mtx`, `h5ad`, `tenx_h5`). `encode` runs on worker threads (mtx text, gzip members for 10x); HDF5 sinks take the CSR directly. MatrixMarket headers need the final nnz, so mtx sinks write the body to a temp file and copy it behind the header in `finish`. Every run also writes `params_used.json`, `cells_truth.tsv.gz` and `genes_truth.tsv.gz`, using splatter's `colData`/`rowData` column names.

## Invariants

- RNG: ChaCha8 with fixed streams per seed (`STREAM_GENES`, `STREAM_CELLS`, `STREAM_CHUNK0 + i`). Output is identical across thread counts. Changing `CELL_CHUNK`, stream indices or the order of draws inside a stream changes every simulated count for a given seed.
- The random stream does not match R. Parity with splatter is distributional, not bitwise.
- `!(x > 0.0)` style comparisons are deliberate (catch NaN); the clippy lint is allowed crate-wide.
- Where validation deliberately differs from R (e.g. `mean.shape`/`mean.rate` strictly positive, `bcv.df > 0`), the reason sits in a comment next to the check.

## Tests

- `tests/pipeline.rs`: end-to-end reproducibility and thread-count independence across all four layouts.
- `tests/r_parity.rs`: three tiers against fixtures in `tests/fixtures/r_parity/` (generated with splatter 1.37.1, see `VERSION`):
  1. params expand like R, and Rust accepts exactly the parameter sets R accepts (`validity_cases.json`);
  2. exact: given R's gene truth and cell metadata, `BaseCellMeans` and `BCV` match to `1e-10` (small fixtures);
  3. distributional: two-sample KS at `alpha = 0.001` on medium fixtures, large behind `large-scale-tests`.
- New settings go into `make_reference.R` and the `SETTINGS` list in `r_parity.rs` together; commit the regenerated fixtures.
