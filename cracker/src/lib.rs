use clap::Parser;
use mpi::ffi::{MPI_Finalize, MPI_Init};

use std::{
    env,
    fmt::Arguments,
    ptr::{null_mut, write},
    sync::Mutex,
};

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
    ARGS.lock().unwrap().clone_from(&options);
    if ARGS.lock().unwrap().use_mpi {
        // Initialize MPI
        unsafe {
            let mut argc = 0;
            let mut argv: *mut *mut i8 = null_mut();
            MPI_Init(&mut argc, &mut argv);
            run_mpi_work();
            MPI_Finalize();
        }
    }
}

pub fn run_mpi_work() {
    println!("Running MPI work");
}
