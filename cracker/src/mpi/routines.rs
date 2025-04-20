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
    communicator::Communicator,
    ffi::{MPI_ANY_SOURCE, MPI_Status, MPI_UINT8_T, MPI_UINT64_T},
    promise::{MpiFuture, waitany},
};

#[repr(i32)]
enum MpiTags {
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

struct MpiProcess<'a> {
    comm: &'a Communicator,
    futures: Vec<Box<dyn MpiFuture>>,
}

impl<'a> MpiProcess<'a> {
    fn new(comm: &'a Communicator) -> Self {
        Self {
            comm,
            futures: Vec::new(),
        }
    }

    fn add_future(&mut self, future: Box<dyn MpiFuture>) {
        self.futures.push(future);
    }
    fn remove_future(&mut self, index: usize) {
        if index < self.futures.len() {
            self.futures.remove(index);
        }
    }

    fn wait_any(&mut self) -> (usize, MPI_Status) {
        let res = waitany(self.futures.as_mut_slice());
        let index = res.index;
        let status = res.status;
        (index, status)
    }

    fn stop_workers(&mut self) {
        for i in 0..self.comm.size() {
            if i == self.comm.rank() {
                continue;
            }
            self.comm
                .send(&[1], MPI_UINT8_T, i as i32, MpiTags::TERMINATE.into());
        }
    }

    fn send_result(&mut self, result: &[u8]) {
        self.comm
            .send_vector(&result, MPI_UINT8_T, 0, MpiTags::RESULT.into());
    }

    fn exchange_timers(&mut self) {
        let mut clock = GlobalClock::instance();
        println!("Clock locked");
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

/* -------------------------------------------------------------------------- */
/*                               Generator Nodes                              */
/* -------------------------------------------------------------------------- */

pub fn generator_process(communicator: &Communicator) -> String {
    let use_dict = {
        let args = ARGS.lock().unwrap();
        args.use_dictionary()
    };
    debug!("Generator process started");
    let mut process = MpiProcess::new(communicator);
    let stop_future: Box<dyn MpiFuture> =
        communicator.irecv::<u8>(100, MPI_UINT8_T, MPI_ANY_SOURCE, MpiTags::RESULT.into());
    process.add_future(stop_future);
    let res = if use_dict {
        chunked_generator_process(&mut process)
    } else {
        brute_generator_process(&mut process)
    };
    process.exchange_timers();
    res
}

fn chunked_generator_process(process: &mut MpiProcess) -> String {
    let filepath = {
        let args = ARGS.lock().unwrap();
        args.dictionary.clone()
    };
    let chunk = {
        let args = ARGS.lock().unwrap();
        args.chunk_size
    };

    let mut request =
        process
            .comm
            .recv_init::<u8>(1, MPI_UINT8_T, MPI_ANY_SOURCE, MpiTags::REQUEST.into());
    request.start();
    process.add_future(request);
    let mut reader = DictionaryReader::new(filepath.as_str()).unwrap_or_else(|_| exit(-1));
    loop {
        let res = process.wait_any();
        match MpiTags::from(res.1.MPI_TAG) {
            MpiTags::REQUEST => {
                let data = reader.generate_flatten_chunk(chunk as usize);
                if data.strings.is_empty() {
                    println!("Empty chunk, stopping");
                    process.stop_workers();
                    return String::new();
                }
                process.comm.send_vector(
                    &data.sizes.as_slice(),
                    MPI_UINT8_T,
                    res.1.MPI_SOURCE,
                    MpiTags::SIZES.into(),
                );
                process.comm.send_vector(
                    &data.strings.as_slice(),
                    MPI_UINT8_T,
                    res.1.MPI_SOURCE,
                    MpiTags::DATA.into(),
                );
                process.futures[res.0]
                    .as_mut_persistent_promise::<u8>()
                    .start();
            }
            MpiTags::RESULT => {
                let result = process.futures[res.0].as_mut_promise::<u8>();
                println!(
                    "Received result: {}",
                    String::from_utf8_lossy(&result.data())
                );
                let res = String::from_utf8_lossy(&result.data()).to_string();
                process.stop_workers();
                return res;
            }
            _ => {
                panic!("Unexpected message");
            }
        }
    }
}

fn brute_generator_process(process: &mut MpiProcess) -> String {
    let mut address: [usize; 2] = [0; 2];
    let chunks = {
        let args = ARGS.lock().unwrap();
        args.chunk_size
    };
    address[1] = chunks as usize;

    let request = process
        .comm
        .irecv::<u8>(1, MPI_UINT8_T, MPI_ANY_SOURCE, MpiTags::REQUEST.into());
    process.add_future(request);
    loop {
        trace!("Waiting for request");
        let (index, status) = process.wait_any();
        trace!(
            "Promise completed, message from rank {} , with tag {}, index {}",
            status.MPI_SOURCE, status.MPI_TAG, index
        );
        match MpiTags::from(status.MPI_TAG) {
            MpiTags::REQUEST => {
                debug!("sending sizes to rank {}", status.MPI_SOURCE);
                process.comm.send_vector(
                    &address,
                    MPI_UINT64_T,
                    status.MPI_SOURCE,
                    MpiTags::BRUTE.into(),
                );

                address[0] += chunks as usize;
                address[1] += chunks as usize;
                process.add_future(process.comm.irecv::<u8>(
                    1,
                    MPI_UINT8_T,
                    MPI_ANY_SOURCE,
                    MpiTags::REQUEST.into(),
                ));
            }
            MpiTags::RESULT => {
                let future = process.futures[index].as_mut();
                let result = future.as_mut_promise::<u8>();
                println!(
                    "Received result: {}",
                    String::from_utf8_lossy(&result.data())
                );
                let ret = String::from_utf8_lossy(&result.data()).to_string();
                process.stop_workers();
                return ret;
            }
            _ => {
                panic!("Unexpected message");
            }
        }
        process.remove_future(index);
    }
}

/* -------------------------------------------------------------------------- */
/*                              Worker processes                              */
/* -------------------------------------------------------------------------- */

pub fn worker_process(communicator: &Communicator) {
    let use_dict = {
        let args = ARGS.lock().unwrap();
        args.use_dictionary()
    };
    debug!("Worker process started");
    let mut process = MpiProcess::new(communicator);
    let stop_future: Box<dyn MpiFuture> =
        communicator.irecv::<u8>(1, MPI_UINT8_T, MPI_ANY_SOURCE, MpiTags::TERMINATE.into());
    process.add_future(stop_future);
    if use_dict {
        chunked_worker_process(&mut process);
    } else {
        brute_worker_process(&mut process);
    }
    process.exchange_timers();
}

fn receive_size_or_stop(process: &mut MpiProcess, chunks: i32) -> Option<Vec<u8>> {
    process.add_future(process.comm.irecv::<u8>(
        chunks as usize,
        MPI_UINT8_T,
        0,
        MpiTags::SIZES.into(),
    ));
    let (index, result) = process.wait_any();
    return match MpiTags::from(result.MPI_TAG) {
        MpiTags::SIZES => {
            let future = process.futures[index].as_mut();
            let sizes = future.as_promise::<u8>();
            let res = Some(sizes.data().to_vec());
            process.remove_future(index);
            res
        }
        MpiTags::TERMINATE => None,
        _ => panic!("Unexpected message"),
    };
}

fn receive_string_or_stop(process: &mut MpiProcess, sizes: &Vec<u8>) -> Option<Vec<u8>> {
    let total_size: usize = sizes.iter().map(|f| *f as usize).sum();
    process.add_future(process.comm.irecv::<u8>(
        total_size as usize,
        MPI_UINT8_T,
        0,
        MpiTags::DATA.into(),
    ));
    let (index, result) = process.wait_any();
    return match MpiTags::from(result.MPI_TAG) {
        MpiTags::DATA => {
            let future = process.futures[index].as_mut();
            let data = future.as_promise::<u8>();
            let res = Some(data.data().to_vec());
            process.remove_future(index);
            res
        }

        MpiTags::TERMINATE => None,
        _ => panic!("Unexpected message"),
    };
}

fn chunked_worker_process(process: &mut MpiProcess) {
    let chunks = {
        let args = ARGS.lock().unwrap();
        args.chunk_size
    };
    let threads = {
        let args = ARGS.lock().unwrap();
        args.num_threads
    };
    let target = {
        let args = ARGS.lock().unwrap();
        CString::new(args.target_md5.clone()).unwrap()
    };
    loop {
        process
            .comm
            .send::<u8>(&0u8, MPI_UINT8_T, 0, MpiTags::REQUEST.into());
        if let Some(mut size) = receive_size_or_stop(process, chunks) {
            if let Some(chunks) = receive_string_or_stop(process, &mut size) {
                let ctx = ComputeContext::Chunked(chunks, size, &target);
                let result = compute(ctx);
                if let Some(res) = result {
                    process.send_result(res.as_bytes());
                }
            } else {
                debug!("Stop requested");
                return;
            }
        } else {
            return;
        }
    }
}

fn brute_worker_process(process: &mut MpiProcess) {
    let target = {
        let args = ARGS.lock().unwrap();
        CString::new(args.target_md5.as_bytes()).unwrap()
    };
    let context_compute_name = format!("Node:{}, brute_compute", process.comm.rank());
    loop {
        process.add_future(
            process
                .comm
                .irecv::<u64>(2, MPI_UINT64_T, 0, MpiTags::BRUTE.into()),
        );
        process
            .comm
            .send(&[0], MPI_UINT8_T, 0, MpiTags::REQUEST.into());

        let (index, status) = process.wait_any();
        match MpiTags::from(status.MPI_TAG) {
            MpiTags::BRUTE => {
                let promise = &process.futures[index];
                let mut sizes = [0u64; 2];
                sizes[0..2].copy_from_slice(&promise.as_promise().data()[0..2]);
                process.remove_future(index);
                let mut result = None;
                GlobalClock::instance().with_context(&context_compute_name, || {
                    let context =
                        ComputeContext::Brute(sizes[0] as usize, sizes[1] as usize, &target);
                    result = compute(context);
                    1
                });
                if let Some(res) = result {
                    process.send_result(res.as_bytes());
                }
            }
            MpiTags::TERMINATE => {
                return;
            }
            _ => {
                panic!("Unexpected mpi tag")
            }
        }
    }
}

#[cfg(test)]
mod tests {

    use std::path::Path;

    use clap::Parser;

    use crate::{ProgramOptions, mpi::scope::init};

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
            bind.ismpi = true;
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
