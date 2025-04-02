use std::mem::MaybeUninit;

use crate::mpi::ffi::MPI_Comm;

use super::ffi::{
    MPI_Comm_rank, MPI_Comm_size, MPI_Datatype, MPI_Finalize, MPI_Get_count, MPI_INT32_T, MPI_Init,
    MPI_Probe, MPI_Recv, MPI_Send, MPI_Status, MpiType,
};

pub struct Communicator {
    comm: MPI_Comm,
    rank: i32,
    size: i32,
}

impl Communicator {
    pub fn new(comm: MPI_Comm) -> Self {
        let mut rank = 0;
        let mut size = 0;
        unsafe {
            MPI_Comm_rank(comm, &mut rank);
            MPI_Comm_size(comm, &mut size);
        }
        Communicator { comm, rank, size }
    }
    pub fn rank(&self) -> i32 {
        self.rank
    }

    pub fn comm(&self) -> MPI_Comm {
        self.comm
    }

    pub fn size(&self) -> i32 {
        self.size
    }

    pub fn send<T>(&self, buf: &T, mpi_type: i32, dest: i32, tag: i32)
    where
        T: Sized,
    {
        unsafe {
            MPI_Send(
                buf as *const T as *const std::ffi::c_void,
                1,
                mpi_type,
                dest,
                tag,
                self.comm,
            );
        }
    }

    pub fn recv<T>(&self, mpi_type: i32, source: i32, tag: i32) -> T {
        let mut result = MaybeUninit::<T>::uninit();
        unsafe {
            let mut status = MaybeUninit::<MPI_Status>::uninit();
            MPI_Recv(
                result.as_mut_ptr() as *mut std::ffi::c_void,
                1,
                mpi_type,
                source,
                tag,
                self.comm,
                status.as_mut_ptr(),
            );
            result.assume_init()
        }
    }

    pub fn send_vector<T>(&self, buf: &[T], mpi_type: i32, dest: i32, tag: i32)
    where
        T: Sized,
    {
        unsafe {
            MPI_Send(
                buf.as_ptr() as *const std::ffi::c_void,
                buf.len() as i32,
                mpi_type,
                dest,
                tag,
                self.comm,
            );
        }
    }

    pub fn recv_vector<T>(&self, mpi_type: i32, source: i32, tag: i32) -> Vec<T> {
        let mut count = 0;
        unsafe {
            let mut status = MaybeUninit::<MPI_Status>::uninit();
            MPI_Probe(source, tag, self.comm, status.as_mut_ptr());
            MPI_Get_count(status.as_ptr(), mpi_type, &mut count);
            let mut result = Vec::<T>::with_capacity(count as usize);
            MPI_Recv(
                result.as_mut_ptr() as *mut std::ffi::c_void,
                count,
                mpi_type,
                source,
                tag,
                self.comm,
                status.as_mut_ptr(),
            );
            result
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{mem::MaybeUninit, os::raw::c_void, ptr::null_mut};

    use crate::mpi::{
        ffi::{
            MPI_F_STATUS_IGNORE, MPI_Finalize, MPI_Get_count, MPI_Init, MPI_PROC_NULL, MPI_Probe,
            MPI_Recv, MPI_Send, MPI_Send_c, MPI_Status, MPI_UINT64_T, MpiType,
        },
        scope::init,
    };

    use super::*;

    #[test]
    fn test_init() {
        let world = init();
        println!("Rank: {}", world.world().rank);
    }
    #[test]
    fn test_send() {
        let world = init();
        let rank = world.world().rank;
        let comm = world.world();
        if rank == 0 {
            let res = comm.recv::<i32>(MPI_INT32_T, 1, 1);
            assert_eq!(res, 10);
        } else {
            let mut s = 10;
            comm.send(&s, MPI_INT32_T, 0, 1);
        }
    }
}
