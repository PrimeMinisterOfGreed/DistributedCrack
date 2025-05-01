/// Program options parsed from command-line arguments
#[derive(Parser, Debug, Default, Clone, Serialize, Deserialize, PartialEq)]
pub struct ProgramOptions {
    #[clap(long, default_value = "NONE")]
    pub config_file: String,
    /// Use GPU for computation
    #[clap(long, default_value_t = false)]
    pub use_gpu: bool,

    /// Target MD5 hash to crack
    #[clap(long, default_value = "NONE")]
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
    #[clap(long, default_value = "savefile.dat")]
    pub savefile: String,

    #[clap(long, default_value = "results.csv")]
    pub result_file: String,

    /// Use MPI for communication
    #[clap(long, default_value_t = true)]
    pub use_mpi: bool,

    /// Path to the dictionary file
    #[clap(long, default_value = "NONE")]
    pub dictionary: String,

    /// Starting point for brute force
    #[clap(long, default_value_t = 4)]
    pub brutestart: i32,
}

use std::{str::FromStr, sync::Mutex};

use clap::Parser;
use lazy_static::lazy_static;
use serde::{Deserialize, Serialize};

use crate::{
    mpi::communicator::{Communicator, MPI_UINT8_T},
    state::StateFile,
};

lazy_static! {
    pub static ref ARGS: Mutex<ProgramOptions> = Mutex::new(ProgramOptions::default());
    pub static ref STORE: Mutex<StateFile> = Mutex::new(StateFile::default());
}

impl ProgramOptions {
    /// Check if a dictionary is being used
    pub fn use_dictionary(&self) -> bool {
        self.dictionary != "NONE"
    }

    pub fn valid_target_md5(&self) -> bool {
        if self.target_md5 == "NONE" {
            return false;
        }
        true
    }
}

pub fn load_options() {
    let mut args = ProgramOptions::parse();
    ARGS.lock().unwrap().clone_from(&args);
    if args.config_file != "NONE" {
        let config =
            std::fs::read_to_string(&args.config_file).expect("Failed to read config file");
        let options: ProgramOptions = toml::from_str(&config).expect("Failed to parse config file");
        ARGS.lock().unwrap().clone_from(&options);
    }
    args = ARGS.lock().unwrap().clone();
    if args.savefile != "NONE" {
        let state_file = StateFile::load_from_file(&args.savefile);
        if let Some(state) = state_file {
            STORE.lock().unwrap().clone_from(&state);
        }
    }
}

pub fn save_options() {
    let options = ARGS.lock().unwrap().clone();
    let state = STORE.lock().unwrap().clone();
    if options.savefile != "NONE" {
        state.save_to_file(&options.savefile).unwrap();
    }
}

pub fn distribute_config_mpi(comm: &Communicator) {
    if comm.rank() == 0 {
        let mut stream = String::new();
        let mut serializer = toml::Serializer::new(&mut stream);
        ARGS.lock().unwrap().serialize(serializer).unwrap();
        let mut buffer = [0u8; 1024];
        buffer[..stream.len()].copy_from_slice(stream.as_bytes());
        comm.broadcast(&mut buffer, comm.rank(), MPI_UINT8_T)
            .unwrap();
    } else {
        let mut buffer = [0u8; 1024];
        comm.broadcast(&mut buffer, 0, MPI_UINT8_T).unwrap();
        let len = buffer.iter().position(|&x| x == 0).unwrap_or(buffer.len());
        let stream = String::from_utf8_lossy(&buffer[..len]).into_owned();
        let mut deserializer = toml::de::Deserializer::new(stream.as_str());
        let options: ProgramOptions = toml::de::from_str(stream.as_str()).unwrap();
        ARGS.lock().unwrap().clone_from(&options);
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use crate::mpi::scope::init;

    use super::*;

    #[test]
    fn test_options_exchange() {
        let mut universe = init();
        let comm = universe.world();
        if comm.rank() == 0 {
            let mut options = ProgramOptions::parse_from(vec![format!(
                "--config-file {}/launch.toml",
                Path::new(env!("CARGO_MANIFEST_DIR"))
                    .parent()
                    .unwrap()
                    .display(),
            )]);
            ARGS.lock().unwrap().clone_from(&options);
            println!("Options: {:?}", options);
            let config =
                std::fs::read_to_string(options.config_file).expect("Failed to read config file");
            let options: ProgramOptions =
                toml::from_str(&config).expect("Failed to parse config file");
            ARGS.lock().unwrap().clone_from(&options);
            distribute_config_mpi(&comm);
        } else {
            distribute_config_mpi(&comm);
            assert_eq!(
                ARGS.lock().unwrap().target_md5,
                "98abe3a28383501f4bfd2d9077820f11"
            );
            assert_eq!(ARGS.lock().unwrap().num_threads, 100000);
            assert_eq!(ARGS.lock().unwrap().chunk_size, 100000);
            assert_eq!(ARGS.lock().unwrap().brutestart, 4);
            assert_eq!(ARGS.lock().unwrap().use_gpu, true);
            assert_eq!(ARGS.lock().unwrap().use_mpi, true);
        }
    }
}
