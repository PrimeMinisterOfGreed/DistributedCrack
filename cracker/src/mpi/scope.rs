use super::{
    communicator::Communicator,
    ffi::{MPI_COMM_WORLD, MPI_Comm, MPI_Comm_rank, MPI_Finalize, MPI_Init},
    promise::MpiFuture,
};

pub struct MpiGlobalScope {}

impl MpiGlobalScope {
    pub fn new() -> Self {
        unsafe {
            MPI_Init(std::ptr::null_mut(), std::ptr::null_mut());
        }
        MpiGlobalScope {}
    }

    pub fn world(&self) -> Communicator {
        let comm = MPI_COMM_WORLD;
        let mut rank = 0;
        unsafe {
            MPI_Comm_rank(comm as i32, &raw mut rank);
        }
        Communicator::new(comm)
    }
}

impl Drop for MpiGlobalScope {
    fn drop(&mut self) {
        unsafe {
            MPI_Finalize();
        }
    }
}

pub fn init() -> MpiGlobalScope {
    MpiGlobalScope::new()
}

pub struct MpiScope {
    futures: Vec<Box<dyn MpiFuture>>,
}
