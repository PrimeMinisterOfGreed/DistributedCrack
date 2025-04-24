/* -------------------------------------------------------------------------- */
/*                               Generator Nodes                              */
/* -------------------------------------------------------------------------- */

use std::process::exit;

use log::{debug, trace};

use crate::{
    dictionary_reader::DictionaryReader,
    mpi::{
        communicator::MPI_UINT64_T,
        ffi::*,
        promise::MpiFuture,
        routines::{MpiProcess, MpiTags},
    },
    options::{ARGS, STORE},
    sequence_generator::ChunkGenerator,
    state::State,
};

use super::communicator::{Communicator, MPI_UINT8_T};

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
//TODO update status
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
    let mut reader =
        DictionaryReader::new(filepath.as_str()).unwrap_or_else(|_| panic!("Failed to open file"));
    loop {
        let res = process.wait_any();
        match MpiTags::from(res.1.status.MPI_TAG) {
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
                    res.1.status.MPI_SOURCE,
                    MpiTags::SIZES.into(),
                );
                process.comm.send_vector(
                    &data.strings.as_slice(),
                    MPI_UINT8_T,
                    res.1.status.MPI_SOURCE,
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

//TODO update status
fn brute_generator_process(process: &mut MpiProcess) -> String {
    let mut address: [usize; 2] = [0; 2];
    let chunks = {
        let args = ARGS.lock().unwrap();
        args.chunk_size
    };
    address[0] = {
        let store = STORE.lock().unwrap();
        store.last_state().last_address
    };
    debug!("Starting address: {}", address[0]);
    address[1] = address[0] + chunks as usize;

    let request = process
        .comm
        .irecv::<u8>(1, MPI_UINT8_T, MPI_ANY_SOURCE, MpiTags::REQUEST.into());
    process.add_future(request);
    loop {
        trace!("Waiting for request");
        let (index, status) = process.wait_any();
        trace!(
            "Promise completed, message from rank {} , with tag {}, index {}",
            status.status.MPI_SOURCE, status.status.MPI_TAG, index
        );
        match MpiTags::from(status.status.MPI_TAG) {
            MpiTags::REQUEST => {
                debug!("sending sizes to rank {}", status.status.MPI_SOURCE);
                process.comm.send_vector(
                    &address,
                    MPI_UINT64_T,
                    status.status.MPI_SOURCE,
                    MpiTags::BRUTE.into(),
                );
                address[0] += chunks as usize;
                address[1] += chunks as usize;
                //TODO evakuate if it is a bottleneck
                {
                    let mut store = STORE.lock().unwrap();
                    store.update_address(address[0]);
                    store.save_default().unwrap();
                }
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
