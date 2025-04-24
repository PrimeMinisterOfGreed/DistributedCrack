use std::{
    ffi::CString,
    io::{Write, stdout},
    process::exit,
};

use allocator_api::Global;
use log::{debug, trace};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use crate::{
    ARGS,
    compute_context::{ComputeContext, compute},
    dictionary_reader::DictionaryReader,
    gpu::md5_transform,
    sequence_generator::ChunkGenerator,
    timers::{ClockStats, Context, GlobalClock},
};

use super::{
    communicator::{Communicator, MPI_UINT8_T, MpiStatus},
    ffi::{MPI_ANY_SOURCE, MPI_Status},
    promise::{MpiFuture, waitany},
};

#[repr(i32)]
pub(crate) enum MpiTags {
    DATA,
    RESULT,
    TERMINATE,
    REQUEST,
    BRUTE,
    SIZES,
}

impl Into<i32> for MpiTags {
    fn into(self) -> i32 {
        self as i32
    }
}

impl From<i32> for MpiTags {
    fn from(value: i32) -> Self {
        match value {
            0 => MpiTags::DATA,
            1 => MpiTags::RESULT,
            2 => MpiTags::TERMINATE,
            3 => MpiTags::REQUEST,
            4 => MpiTags::BRUTE,
            5 => MpiTags::SIZES,
            _ => panic!("Invalid MPI tag"),
        }
    }
}

pub struct MpiProcess<'a> {
    pub(crate) comm: &'a Communicator,
    pub(crate) futures: Vec<Box<dyn MpiFuture>>,
}

impl<'a> MpiProcess<'a> {
    pub(crate) fn new(comm: &'a Communicator) -> Self {
        Self {
            comm,
            futures: Vec::new(),
        }
    }

    pub(crate) fn add_future(&mut self, future: Box<dyn MpiFuture>) {
        self.futures.push(future);
    }
    pub(crate) fn remove_future(&mut self, index: usize) {
        if index < self.futures.len() {
            self.futures.remove(index);
        }
    }

    pub(crate) fn wait_any(&mut self) -> (usize, MpiStatus) {
        let res = waitany(self.futures.as_mut_slice());
        let index = res.index;
        let status = res.status;
        (index, status)
    }

    pub(crate) fn stop_workers(&mut self) {
        for i in 0..self.comm.size() {
            if i == self.comm.rank() {
                continue;
            }
            self.comm
                .send(&[1], MPI_UINT8_T, i as i32, MpiTags::TERMINATE.into());
        }
    }

    pub(crate) fn send_result(&mut self, result: &[u8]) {
        self.comm
            .send_vector(&result, MPI_UINT8_T, 0, MpiTags::RESULT.into());
    }

    pub(crate) fn exchange_timers(&mut self) {
        let mut clock = GlobalClock::instance();
        if self.comm.rank() == 0 {
            /* --------------------------- Receive clock stats -------------------------- */
            stdout().flush().unwrap();
            for i in 1..self.comm.size() {
                let data: Vec<Context> = self.comm.recv_object_vector(i, 99);
                for ctx in data {
                    clock.add_context(ctx);
                }
            }
        } else {
            /* -------------------------- Send out clock stats -------------------------- */
            let data = clock.get_contexts().map(|t| t.clone()).collect::<Vec<_>>();
            self.comm.send_object_vector(data.as_slice(), 0, 99);
        }
    }
}

#[cfg(test)]
mod tests {

    use std::path::Path;

    use clap::Parser;

    use crate::{
        mpi::{
            communicator::MPI_UINT64_T,
            generators::generator_process,
            scope::init,
            workers::{chunked_worker_process, worker_process},
        },
        options::ProgramOptions,
    };

    use super::*;

    #[test]
    fn test_generator_routine() {
        let universe = init();
        let comm = universe.world();
        simple_logger::init_with_level(log::Level::Trace).unwrap();
        debug!("Rank: {}", comm.rank());
        {
            let mut args = ARGS.lock();
            let bind = args.as_mut().unwrap();
            bind.chunk_size = 100;
            bind.use_mpi = true;
            bind.dictionary = "NONE".to_string();
            bind.brutestart = 4;
        }
        let rank = comm.rank();
        if rank == 0 {
            generator_process(&comm);
        } else {
            trace!("Sending message");
            comm.send(&[1], MPI_UINT8_T, 0, MpiTags::REQUEST.into());
            let sizes = comm.recv_vector::<u64>(MPI_UINT64_T, 0, MpiTags::SIZES.into());
            assert_eq!(sizes.len(), 2);
            comm.send(&"hello world", MPI_UINT8_T, 0, MpiTags::RESULT.into());
        }
    }

    #[test]
    fn test_worker_routine() {
        let universe = init();
        let comm = universe.world();
        if comm.rank() == 0 {
            comm.recv::<u8>(MPI_UINT8_T, MPI_ANY_SOURCE, MpiTags::REQUEST.into());
            let buffer = [0u64, 10000];
            comm.send_vector(&buffer, MPI_UINT64_T, 1, MpiTags::BRUTE.into());
            comm.send(&[0], MPI_UINT8_T, 1, MpiTags::TERMINATE.into());
        } else {
            worker_process(&comm);
        }
    }

    #[test]
    fn test_chunked_worker_routine() {
        let universe = init();
        let comm = universe.world();

        let options = ProgramOptions::parse_from(
            "--use-mpi  --target-md5 c4eaf0c0b43f2efcefa870ddbab7950c --num-threads 10 --chunk-size 10"
            .split_whitespace()
        );
        ARGS.lock().unwrap().clone_from(&options);
        if comm.rank() == 0 {
            let filepath = Path::new(format!("{}", env!("CARGO_MANIFEST_DIR")).as_str())
                .parent()
                .unwrap()
                .join("dictionary.txt");
            let mut generator = DictionaryReader::new(filepath.as_os_str().to_str().unwrap())
                .unwrap_or_else(|_| panic!("Failed to open dictionary"));
            let mut data = generator.generate_flatten_chunk(10 as usize);
            comm.recv::<u8>(MPI_UINT8_T, 1, MpiTags::REQUEST.into());
            comm.send_vector(
                &data.sizes.as_slice(),
                MPI_UINT8_T,
                1,
                MpiTags::SIZES.into(),
            );
            comm.send_vector(
                &data.strings.as_slice(),
                MPI_UINT8_T,
                1,
                MpiTags::DATA.into(),
            );
            let res = comm.recv_vector::<u8>(MPI_UINT8_T, 1, MpiTags::RESULT.into());
            println!("Received result:{}", String::from_utf8_lossy(&res));
        } else {
            chunked_worker_process(&mut MpiProcess::new(&comm));
        }
    }
}
