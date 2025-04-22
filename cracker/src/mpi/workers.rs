/* -------------------------------------------------------------------------- */
/*                              Worker processes                              */
/* -------------------------------------------------------------------------- */

use std::ffi::CString;

use log::debug;

use crate::{
    compute_context::{ComputeContext, compute},
    mpi::{
        ffi::*,
        promise::MpiFuture,
        routines::{MpiProcess, MpiTags},
    },
    options::ARGS,
    timers::GlobalClock,
};

use super::communicator::Communicator;

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

pub fn chunked_worker_process(process: &mut MpiProcess) {
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

pub fn brute_worker_process(process: &mut MpiProcess) {
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
