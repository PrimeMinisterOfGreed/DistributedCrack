use super::{
    communicator::Communicator,
    ffi::{MPI_COMM_WORLD, MPI_Comm, MPI_Comm_rank, MPI_Finalize, MPI_Init},
};

pub struct MpiScope {}

impl MpiScope {
    pub fn new() -> Self {
        unsafe {
            MPI_Init(std::ptr::null_mut(), std::ptr::null_mut());
        }
        MpiScope {}
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

impl Drop for MpiScope {
    fn drop(&mut self) {
        unsafe {
            MPI_Finalize();
        }
    }
}

pub fn init() -> MpiScope {
    MpiScope::new()
}
