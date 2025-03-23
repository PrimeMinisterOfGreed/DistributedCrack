use mpi::{
    Rank,
    ffi::MPI_Finalize,
    request::{self, CancelGuard, WaitGuard, wait_any},
    topology::SimpleCommunicator,
    traits::{
        AsDatatype, Buffer, Communicator, CommunicatorCollectives, Destination, Root, Source,
    },
};

use crate::ARGS;

pub fn run_mpi() {}

pub fn root_routine(communicator: SimpleCommunicator) {}

pub fn worker_routine(communicator: SimpleCommunicator) {}
