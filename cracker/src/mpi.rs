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

use crate::ARGS;

enum MpiTags {
    DATA,
    RESULT,
    TERMINATE,
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
        let stopreq = communicator.this_process().immediate_receive_into_with_tag(
            f,
            &mut result,
            MpiTags::RESULT as i32,
        );
        while !completed(&stopreq) {}
    });
}

pub fn root_bruter(communicator: SimpleCommunicator) {}

pub fn root_chunked(communicator: SimpleCommunicator) {}

pub fn worker(communicator: SimpleCommunicator) {}

pub fn worker_bruter(communicator: SimpleCommunicator) {}

pub fn worker_chunked(communicator: SimpleCommunicator) {}
