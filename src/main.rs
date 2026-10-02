//! `splatter-sc <params.json>`: simulate a Splat data set and write it to
//! the layouts named in the JSON. Threads follow `RAYON_NUM_THREADS`.

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use splatter_sc::params::SplatParams;

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let (Some(path), None) = (args.next(), args.next()) else {
        eprintln!("usage: splatter-sc <params.json>");
        return ExitCode::FAILURE;
    };
    let t = Instant::now();
    let result = SplatParams::from_json_file(&PathBuf::from(path))
        .and_then(|p| p.resolve())
        .and_then(splatter_sc::run);
    match result {
        Ok(s) => {
            eprintln!(
                "Done in {:.2?}: {} nonzeros (setup {:.2?}, simulate {:.2?}, write {:.2?}, finish {:.2?})",
                t.elapsed(),
                s.nnz,
                s.setup,
                s.simulate,
                s.write,
                s.finish
            );
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("Error: {e}");
            ExitCode::FAILURE
        }
    }
}
