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
use crate::output::{Sink, encode, write_gz};
use crate::params::{Layout, Method, SplatParams};

/// Cells per chunk. Each chunk has its own RNG stream, so this fixes the
/// output for a seed; changing it changes every simulated count. 4096 cells
/// keeps a chunk's CSR in the low MB at typical sparsity.
pub const CELL_CHUNK: usize = 4096;

/// Chunks per thread in one window. Bounds memory to two windows (one being
/// simulated, one being written) while keeping all threads busy.
const CHUNKS_PER_THREAD: usize = 4;

/// RNG stream of the per-gene draws.
const STREAM_GENES: u64 = 0;

/// RNG stream of the per-cell metadata.
const STREAM_CELLS: u64 = 1;

/// RNG stream of chunk 0; chunk `i` uses `STREAM_CHUNK0 + i`.
const STREAM_CHUNK0: u64 = 2;

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
        let n_groups = if params.method == Method::Groups {
            params.n_groups()
        } else {
            1
        };
        let profiles = Profiles::new(&genes, params.n_batches(), n_groups);
        let cells = CellMeta::simulate(&params, &mut stream_rng(seed, STREAM_CELLS))?;
        Ok(Self {
            params,
            genes,
            profiles,
            cells,
        })
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

/// Open the sink of every requested layout.
///
/// ### Params
///
/// * `sim` - The simulation
///
/// ### Returns
///
/// One sink per layout, in `output.layouts` order.
fn open_sinks(sim: &Simulation) -> Result<Vec<Box<dyn Sink>>, SplatErrors> {
    let dir = &sim.params.output.dir;
    sim.params
        .output
        .layouts
        .iter()
        .map(|l| -> Result<Box<dyn Sink>, SplatErrors> {
            Ok(match l {
                Layout::Parse => {
                    Box::new(output::parse::ParseSink::new(dir, &sim.params, &sim.cells)?)
                }
                Layout::TenxMtx => Box::new(output::tenx_mtx::TenxMtxSink::new(dir, &sim.params)?),
            })
        })
        .collect()
}

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
    let sinks = open_sinks(&sim)?;
    let setup = t0.elapsed();

    let layouts = sim.params.output.layouts.clone();
    let n_chunks = sim.n_chunks();
    let n_cells = sim.cells.n_cells();
    let window = rayon::current_num_threads() * CHUNKS_PER_THREAD;

    type Window = Vec<(CellChunk, Vec<Vec<u8>>)>;
    let (simulate, write, sinks, nnz) = std::thread::scope(|s| {
        let (tx, rx) = sync_channel::<Window>(1);
        let writer = s.spawn(move || -> Result<_, SplatErrors> {
            let mut sinks = sinks;
            let (mut busy, mut nnz) = (Duration::ZERO, 0u64);
            for batch in rx {
                let t = Instant::now();
                for (chunk, encoded) in &batch {
                    for (sink, e) in sinks.iter_mut().zip(encoded) {
                        sink.write_chunk(chunk, e)?;
                    }
                    nnz += chunk.counts.len() as u64;
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
                        .map(|&l| encode(l, &chunk))
                        .collect::<Result<_, _>>()?;
                    Ok((chunk, encoded))
                })
                .collect();
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
