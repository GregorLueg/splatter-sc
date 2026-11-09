//! A streaming Rust port of the Splat simulator from splatter.
//!
//! After a handful of per-gene vectors, every cell is independent, so counts
//! are generated in chunks of cells in parallel and streamed to disk. Peak
//! memory is the per-gene state plus a window of chunks, independent of the
//! number of cells. Same seed, same output, whatever the thread count; the
//! random stream does not match R's.
//!
//! The library target exists for the binary and the tests. It is not a
//! stable API.
//!
//! ### References
//!
//! Zappia L, Phipson B, Oshlack A. Splatter: simulation of single-cell RNA
//! sequencing data. Genome Biology 18, 174 (2017). doi:10.1186/s13059-017-1305-0
//!
//! Ported from splatter (GPL-3), <https://github.com/Oshlack/splatter>.

// `!(x > 0.0)` is deliberate throughout: it also catches NaN.
#![allow(clippy::neg_cmp_op_on_partial_ord)]

pub mod cells;
pub mod errors;
pub mod genes;
pub mod output;
pub mod params;

use std::io::Write;
use std::sync::mpsc::sync_channel;
use std::time::{Duration, Instant};

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

use crate::cells::{CellChunk, CellMeta, simulate_chunk};
use crate::errors::SplatErrors;
use crate::genes::{GeneTruth, Profiles};
use crate::output::{H5Chunks, H5Encoder, encode, write_gz};
use crate::params::{Method, SplatParams};

////////////
// Consts //
////////////

/// Cells per chunk. Each chunk has its own RNG stream, so this fixes the
/// output for a seed; changing it changes every simulated count. Memory in
/// flight is about `2 * threads * CHUNKS_PER_THREAD * CELL_CHUNK` cells of
/// CSR plus encoded text, ~20 bytes per nonzero. At splatter's default
/// `lib.loc = 11` (~8k nonzeros per cell over 20k genes) one 256-cell chunk
/// is ~45 MB. The HDF5 encoders add, for the window being encoded only, its
/// raw values (8 bytes per nonzero for h5ad, 12 for 10x h5) before they are
/// cut into chunks; between windows each keeps under one chunk per dataset,
/// and the compressed chunks in the channel are smaller than the raw values.
pub const CELL_CHUNK: usize = 256;

/// Chunks per thread in one window. Two windows are alive at most, one
/// being simulated and one being written.
const CHUNKS_PER_THREAD: usize = 2;

/// RNG stream of the per-gene draws.
const STREAM_GENES: u64 = 0;

/// RNG stream of the per-cell metadata.
const STREAM_CELLS: u64 = 1;

/// RNG stream of chunk 0; chunk `i` uses `STREAM_CHUNK0 + i`.
const STREAM_CHUNK0: u64 = 2;

/////////////
// Helpers //
/////////////

/// RNG for one stream of a seed.
///
/// ### Params
///
/// * `seed` - Simulation seed
/// * `stream` - Stream index
///
/// ### Returns
///
/// An independent ChaCha8 stream.
fn stream_rng(seed: u64, stream: u64) -> ChaCha8Rng {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    rng.set_stream(stream);
    rng
}

////////////////
// Simulation //
////////////////

/// Everything drawn before the counts: gene truth, profiles, cell metadata.
pub struct Simulation {
    /// Resolved parameters
    pub params: SplatParams,
    /// Per-gene truth
    pub genes: GeneTruth,
    /// Normalised (batch, group) profiles
    pub profiles: Profiles,
    /// Per-cell truth
    pub cells: CellMeta,
}

impl Simulation {
    /// Draw the gene and cell state for resolved parameters.
    ///
    /// ### Params
    ///
    /// * `params` - Output of [`SplatParams::resolve`]
    ///
    /// ### Returns
    ///
    /// The simulation, ready to generate chunks.
    pub fn new(params: SplatParams) -> Result<Self, SplatErrors> {
        let seed = params.seed();
        let genes = GeneTruth::simulate(&params, &mut stream_rng(seed, STREAM_GENES))?;
        let cells = CellMeta::simulate(&params, &mut stream_rng(seed, STREAM_CELLS))?;
        Ok(Self::from_parts(params, genes, cells))
    }

    /// Build a simulation from given gene truth and cell metadata, e.g. to
    /// run on another implementation's draws.
    ///
    /// ### Params
    ///
    /// * `params` - Output of [`SplatParams::resolve`]
    /// * `genes` - Gene truth matching `params`
    /// * `cells` - Cell metadata matching `params`
    ///
    /// ### Returns
    ///
    /// The simulation, ready to generate chunks.
    pub fn from_parts(params: SplatParams, genes: GeneTruth, cells: CellMeta) -> Self {
        let n_groups = if params.method == Method::Groups {
            params.n_groups()
        } else {
            1
        };
        let profiles = Profiles::new(&genes, params.n_batches(), n_groups);
        Self {
            params,
            genes,
            profiles,
            cells,
        }
    }

    /// Number of chunks.
    ///
    /// ### Returns
    ///
    /// `ceil(n_cells / CELL_CHUNK)`.
    pub fn n_chunks(&self) -> usize {
        self.cells.n_cells().div_ceil(CELL_CHUNK)
    }

    /// Simulate chunk `i`.
    ///
    /// ### Params
    ///
    /// * `i` - Chunk index
    ///
    /// ### Returns
    ///
    /// Counts for cells `i * CELL_CHUNK ..` in cell-major CSR.
    pub fn chunk(&self, i: usize) -> Result<CellChunk, SplatErrors> {
        let start = i * CELL_CHUNK;
        let end = (start + CELL_CHUNK).min(self.cells.n_cells());
        let mut rng = stream_rng(self.params.seed(), STREAM_CHUNK0 + i as u64);
        simulate_chunk(
            &self.params,
            &self.genes,
            &self.profiles,
            &self.cells,
            start,
            end,
            &mut rng,
        )
    }
}

////////////////
// RunSummary //
////////////////

/// Timings and size of a finished run.
#[derive(Clone, Debug)]
pub struct RunSummary {
    /// Gene and cell setup
    pub setup: Duration,
    /// Main thread time in simulation and encoding windows
    pub simulate: Duration,
    /// Writer thread time appending chunks
    pub write: Duration,
    /// Finalising the files (mtx header plus body copy)
    pub finish: Duration,
    /// Total nonzeros
    pub nnz: u64,
}

/////////
// Run //
/////////

/// Run a full simulation and write all requested layouts plus the truth
/// sidecars (`params_used.json`, `cells_truth.tsv.gz`, `genes_truth.tsv.gz`).
///
/// Windows of chunks are simulated and encoded in parallel, then handed to
/// one writer thread over a channel of depth one, so writing overlaps with
/// the next window.
///
/// ### Params
///
/// * `params` - Output of [`SplatParams::resolve`]
///
/// ### Returns
///
/// Timings and the nonzero count.
pub fn run(params: SplatParams) -> Result<RunSummary, SplatErrors> {
    let t0 = Instant::now();
    let dir = params.output.dir.clone();
    std::fs::create_dir_all(&dir)?;
    let mut f = output::create(&dir.join("params_used.json"))?;
    serde_json::to_writer_pretty(&mut f, &params)?;
    f.flush()?;

    let sim = Simulation::new(params)?;
    write_gz(
        &dir.join("cells_truth.tsv.gz"),
        &output::tables::cell_table(&sim.params, &sim.cells, '\t', "Cell", "Batch"),
    )?;
    write_gz(
        &dir.join("genes_truth.tsv.gz"),
        &output::tables::gene_table(&sim.genes, '\t'),
    )?;
    let sinks = output::open_sinks(&sim)?;
    let setup = t0.elapsed();

    let layouts = sim.params.output.layouts.clone();
    let parse_gzip = sim.params.output.parse_gzip;
    let n_chunks = sim.n_chunks();
    let n_cells = sim.cells.n_cells();
    let window = rayon::current_num_threads() * CHUNKS_PER_THREAD;
    // Built here, not in the writer: their rayon work must run on the pool
    // `run` was called in, and the writer thread would use the global one.
    let mut encoders: Vec<Option<H5Encoder>> = layouts
        .iter()
        .map(|&l| H5Encoder::new(l, &sim.params.output))
        .collect();

    // Cell chunks with their per-layout text, plus per-layout HDF5 chunks.
    type Window = (Vec<(CellChunk, Vec<Vec<u8>>)>, Vec<Option<H5Chunks>>);
    let (simulate, write, sinks, nnz) = std::thread::scope(|s| {
        let (tx, rx) = sync_channel::<Window>(1);
        let writer = s.spawn(move || -> Result<_, SplatErrors> {
            let mut sinks = sinks;
            let (mut busy, mut nnz) = (Duration::ZERO, 0u64);
            for (batch, h5) in rx {
                let t = Instant::now();
                for (chunk, encoded) in &batch {
                    for (sink, e) in sinks.iter_mut().zip(encoded) {
                        sink.write_chunk(chunk, e)?;
                    }
                    nnz += chunk.counts.len() as u64;
                }
                for (sink, h) in sinks.iter_mut().zip(&h5) {
                    if let Some(h) = h {
                        sink.write_h5(h)?;
                    }
                }
                busy += t.elapsed();
            }
            Ok((sinks, busy, nnz))
        });

        let mut simulate = Duration::ZERO;
        let mut result = Ok(());
        for w0 in (0..n_chunks).step_by(window) {
            let t = Instant::now();
            let batch: Result<Window, SplatErrors> = (w0..(w0 + window).min(n_chunks))
                .into_par_iter()
                .map(|i| {
                    let chunk = sim.chunk(i)?;
                    let encoded = layouts
                        .iter()
                        .map(|&l| encode(l, parse_gzip, &chunk))
                        .collect::<Result<_, _>>()?;
                    Ok((chunk, encoded))
                })
                .collect::<Result<Vec<_>, _>>()
                .and_then(|b| {
                    let h5 = encoders
                        .iter_mut()
                        .map(|e| {
                            e.as_mut()
                                .map(|e| e.encode(b.iter().map(|(c, _)| c)))
                                .transpose()
                        })
                        .collect::<Result<_, _>>()?;
                    Ok((b, h5))
                });
            simulate += t.elapsed();
            match batch {
                // A send error means the writer died; its error surfaces on join.
                Ok(b) => {
                    if tx.send(b).is_err() {
                        break;
                    }
                }
                Err(e) => {
                    result = Err(e);
                    break;
                }
            }
            let done = ((w0 + window) * CELL_CHUNK).min(n_cells);
            eprint!("\rSimulated {done} / {n_cells} cells");
        }
        eprintln!();
        if result.is_ok() {
            let t = Instant::now();
            let tails = encoders
                .iter_mut()
                .map(|e| e.as_mut().map(H5Encoder::finish).transpose())
                .collect::<Result<_, _>>();
            simulate += t.elapsed();
            match tails {
                // A send error means the writer died; its error surfaces on join.
                Ok(h5) => {
                    let _ = tx.send((Vec::new(), h5));
                }
                Err(e) => result = Err(e),
            }
        }
        drop(tx);
        let (sinks, write, nnz) = writer.join().expect("writer thread panicked")?;
        result.map(|_| (simulate, write, sinks, nnz))
    })?;

    let t = Instant::now();
    for sink in sinks {
        sink.finish()?;
    }
    Ok(RunSummary {
        setup,
        simulate,
        write,
        finish: t.elapsed(),
        nnz,
    })
}
