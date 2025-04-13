use clap::Parser;
use mpi::{
    ffi::{MPI_Finalize, MPI_Init},
    scope::MpiGlobalScope,
};
use rayon::ThreadPoolBuilder;
use std::{
    cmp::min,
    env,
    fmt::Arguments,
    io::{Write, stdout},
    ptr::{null_mut, write},
    sync::Mutex,
};
use timers::GlobalClock;

mod compute_context;
mod dictionary_reader;
pub mod gpu;
mod mpi;
pub mod sequence_generator;
mod timers;
/// Program options parsed from command-line arguments
#[derive(Parser, Debug, Default, Clone)]
pub struct ProgramOptions {
    /// Path to the configuration file
    #[clap(long, default_value = "config.txt")]
    pub config_file: String,

    /// Use GPU for computation
    #[clap(long, default_value_t = false)]
    pub use_gpu: bool,

    /// Target MD5 hash to crack
    #[clap(long)]
    pub target_md5: String,

    /// Number of threads to use
    #[clap(long, default_value_t = 1000)]
    pub num_threads: i32,

    /// Size of each chunk to process
    #[clap(long, default_value_t = 1000)]
    pub chunk_size: i32,

    /// Verbosity level
    #[clap(long, default_value_t = 0)]
    pub verbosity: i32,

    /// Path to the save file
    #[clap(long, default_value = "savefile.txt")]
    pub savefile: String,

    /// Use MPI for distributed computation
    #[clap(long, default_value_t = false)]
    pub ismpi: bool,

    /// Restore from a previously saved file
    #[clap(long, default_value_t = false)]
    pub restore_from_file: bool,

    /// Use MPI for communication
    #[clap(long, default_value_t = false)]
    pub use_mpi: bool,

    /// Path to the dictionary file
    #[clap(long, default_value = "NONE")]
    pub dictionary: String,

    /// Starting point for brute force
    #[clap(long, default_value_t = 4)]
    pub brutestart: i32,
}

use lazy_static::lazy_static;

lazy_static! {
    static ref ARGS: Mutex<ProgramOptions> = Mutex::new(ProgramOptions::default());
}

impl ProgramOptions {
    /// Check if a dictionary is being used
    fn use_dictionary(&self) -> bool {
        self.dictionary != "NONE"
    }
}

#[unsafe(no_mangle)]
pub fn rust_main() {
    let options = ProgramOptions::parse();
    simple_logger::init_with_level(log::Level::Warn).unwrap();
    ARGS.lock().unwrap().clone_from(&options);
    ThreadPoolBuilder::new()
        .num_threads(min(ARGS.lock().unwrap().num_threads as usize, 32))
        .build_global()
        .unwrap();
    if ARGS.lock().unwrap().use_mpi {
        // Initialize MPI
        let mut argc = 0;
        let mut argv: *mut *mut i8 = null_mut();
        let mut global_scope = MpiGlobalScope::new();
        run_mpi_work(&mut global_scope);
    }

    GlobalClock::instance().report_stats();
}

pub fn run_mpi_work(global_scope: &mut MpiGlobalScope) -> Option<String> {
    let world = global_scope.world();
    let rank = world.rank();
    if rank == 0 {
        let mut res = None;
        GlobalClock::instance().with_context("wallclock", || {
            res = Some(mpi::routines::generator_process(&world));
            1
        });
        res
    } else {
        mpi::routines::worker_process(&world);
        None
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use crate::mpi::scope::init;

    use super::*;

    #[test]
    fn test_brute_attack_with_gpu() {
        let options = ProgramOptions::parse_from(
            "--use-mpi --use-gpu --target-md5 4a7d1ed414474e4033ac29ccb8653d9b --num-threads 1000 --chunk-size 1000"
                .split_whitespace(),
        );
        ARGS.lock().unwrap().clone_from(&options);
        let mut universe = init();
        let result = run_mpi_work(&mut universe);
        if universe.world().rank() == 0 {
            assert_eq!(result.unwrap(), "0000");
        }
    }

    #[test]
    fn test_brute_attack() {
        let options = ProgramOptions::parse_from(
            "--use-mpi  --target-md5 4a7d1ed414474e4033ac29ccb8653d9b --num-threads 1000 --chunk-size 1000"
                .split_whitespace(),
        );
        ARGS.lock().unwrap().clone_from(&options);
        let mut universe = init();
        let result = run_mpi_work(&mut universe);
        if universe.world().rank() == 0 {
            assert_eq!(result.unwrap().replace('\0', " ").trim(), "0000");
        }
    }

    #[test]
    fn test_chunked_attack() {
        let options = ProgramOptions::parse_from(format!(
            "--use-mpi --target-md5 45ed7216d59ce25d2ce05470c6bf52d0 --num-threads 1000 --chunk-size 1000 --dictionary {}/dictionary.txt", Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().display())
                .split_whitespace(),
        );
        ARGS.lock().unwrap().clone_from(&options);
        let mut universe = init();
        let result = run_mpi_work(&mut universe);
        if universe.world().rank() == 0 {
            assert_eq!(result.unwrap(), "#name?");
        }
    }
}
