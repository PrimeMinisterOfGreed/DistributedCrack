use std::vec;

use mpi::{
    Rank,
    ffi::{MPI_F_STATUS_IGNORE, MPI_Finalize, MPI_Test, MPI_Wait, ompi_status_public_t},
    point_to_point::ReceiveFuture,
    raw::AsRaw,
    request::{self, CancelGuard, LocalScope, Request, Scope, WaitGuard, scope, wait_any},
    topology::SimpleCommunicator,
    traits::{
        AsDatatype, Buffer, Communicator, CommunicatorCollectives, Destination, Root, Source,
    },
};

use crate::{ARGS, sequence_generator::SequenceGenerator};

#[repr(i32)]
enum MpiTags {
    DATA,
    RESULT,
    TERMINATE,
    ALIVE,
}

impl Into<i32> for MpiTags {
    fn into(self) -> i32 {
        self as i32
    }
}

pub fn run_mpi() {}

pub fn completed<'a, T>(request: &Request<'a, T, &LocalScope<'a>>) -> bool {
    let mut ptr = request.as_raw();
    unsafe {
        let mut flag = 0;

        MPI_Test(
            &mut ptr,
            &mut flag,
            MPI_F_STATUS_IGNORE as *mut ompi_status_public_t,
        );
        flag != 0
    }
}

pub fn root(communicator: SimpleCommunicator) {
    let mut result = [0i8; 32];

    mpi::request::scope(|f| {
        let mut stopreq = communicator.this_process().immediate_receive_into_with_tag(
            f,
            &mut result,
            MpiTags::RESULT.into(),
        );
        while !completed(&stopreq) {
            if ARGS.lock().unwrap().use_dictionary() {}
        }
    });
}

pub fn root_bruter(communicator: SimpleCommunicator) {}

pub fn root_chunked(communicator: SimpleCommunicator) {
    let req = communicator
        .this_process()
        .receive_with_tag::<u8>(MpiTags::ALIVE.into());
}

pub fn worker(communicator: SimpleCommunicator) {}

pub fn worker_bruter(communicator: SimpleCommunicator) {}

pub fn worker_chunked(communicator: SimpleCommunicator) {}
