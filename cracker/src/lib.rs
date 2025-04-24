use clap::Parser;
use log::logger;
use mpi::{
    ffi::{MPI_Finalize, MPI_Init},
    scope::MpiGlobalScope,
};
use options::{ARGS, load_options};
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
pub mod options;
pub mod result_collector;
pub mod sequence_generator;
pub mod single_node;
pub mod state;
mod timers;

#[unsafe(no_mangle)]
pub fn rust_main() {
    load_options();
    simple_logger::init_with_level({
        match ARGS.lock().unwrap().verbosity {
            0 => log::Level::Error,
            1 => log::Level::Warn,
            2 => log::Level::Info,
            3 => log::Level::Debug,
            _ => log::Level::Trace,
        }
    })
    .unwrap();
    ThreadPoolBuilder::new()
        .num_threads(min(ARGS.lock().unwrap().num_threads as usize, 32))
        .build_global()
        .unwrap();
    if ARGS.lock().unwrap().use_mpi {
        let mut global_scope = MpiGlobalScope::new();
        run_mpi_work(&mut global_scope);
    } else {
        single_node_routine();
    }
    GlobalClock::instance().report_stats();
}

pub fn run_mpi_work(global_scope: &mut MpiGlobalScope) -> Option<String> {
    let world = global_scope.world();
    let rank = world.rank();
    if rank == 0 {
        let res = Some(
            mpi::generators::generator_process(&world)
                .trim_end_matches('\0')
                .to_string(),
        );
        res
    } else {
        mpi::workers::worker_process(&world);
        None
    }
}

pub fn single_node_routine() {}

#[cfg(test)]
mod tests {
    use std::{path::Path, process::Termination};

    use crate::{mpi::scope::init, options::ProgramOptions};

    use super::*;

    #[test]
    fn test_brute_attack_with_gpu() {
        let options = ProgramOptions::parse_from(
            "--use-mpi --use-gpu --brutestart 4b --target-md5 98abe3a28383501f4bfd2d9077820f11 --num-threads 100000 --chunk-size 100000"
                .split_whitespace(),
        );
        ARGS.lock().unwrap().clone_from(&options);
        let mut universe = init();
        let result = run_mpi_work(&mut universe);
        if universe.world().rank() == 0 {
            assert_eq!(result.unwrap(), "!!!!");
        }
        GlobalClock::instance().report_stats();
    }

    #[test]
    fn test_brute_attack() {
        let options = ProgramOptions::parse_from(
            "--use-mpi  --target-md5 98abe3a28383501f4bfd2d9077820f11 --num-threads 1000 --chunk-size 1000"
                .split_whitespace(),
        );
        ARGS.lock().unwrap().clone_from(&options);
        let mut universe = init();
        let result = run_mpi_work(&mut universe);
        if universe.world().rank() == 0 {
            assert_eq!(result.unwrap().replace('\0', " ").trim(), "!!!!");
        }
        if universe.world().rank() == 0 {
            GlobalClock::instance().report_stats();
        }
    }

    #[test]
    fn test_chunked_attack() {
        let options = ProgramOptions::parse_from(format!(
            "--use-mpi --target-md5 c4eaf0c0b43f2efcefa870ddbab7950c --num-threads 1000 --chunk-size 1000 --dictionary {}/dictionary.txt", Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().display())
                .split_whitespace(),
        );
        ARGS.lock().unwrap().clone_from(&options);
        let mut universe = init();
        let result = run_mpi_work(&mut universe);
        if universe.world().rank() == 0 {
            assert_eq!(result.unwrap(), "#name?");
        }
        if universe.world().rank() == 0 {
            GlobalClock::instance().report_stats();
        }
    }

    #[test]

    fn bench_brute_attack() {
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
        if universe.world().rank() == 0 {
            GlobalClock::instance().report_stats();
        }
    }
}
