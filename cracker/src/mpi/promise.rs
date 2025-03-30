use std::mem::MaybeUninit;

use super::{
    communicator::Communicator,
    ffi::{MPI_Iprobe, MPI_Irecv, MPI_Isend, MPI_Request, MPI_Status, MPI_Test},
};

pub struct MpiRequest<T> {
    buffer: MaybeUninit<T>,
    request: MaybeUninit<MPI_Request>,
    status: MaybeUninit<MPI_Status>,
    flag: i32,
}

impl<T> MpiRequest<T> {
    pub fn new() -> Self {
        MpiRequest {
            buffer: MaybeUninit::uninit(),
            request: MaybeUninit::uninit(),
            status: MaybeUninit::uninit(),
            flag: 0,
        }
    }

    pub fn test(&mut self) -> Option<MPI_Status> {
        unsafe {
            MPI_Test(
                self.request.as_mut_ptr(),
                &raw mut self.flag,
                self.status.as_mut_ptr(),
            );
        }
        if self.flag == 1 {
            Some(unsafe { self.status.assume_init() })
        } else {
            None
        }
    }
}

impl Communicator {
    pub fn irecv<T>(&self, mpi_type: i32, source: i32, tag: i32) -> MpiRequest<T> {
        let mut result = MpiRequest::<T>::new();
        //this is wrong
        unsafe {
            MPI_Iprobe(
                source,
                tag,
                self.comm(),
                &raw mut result.flag,
                result.status.as_mut_ptr(),
            );
            result
        }
    }

    pub fn isend<T>(&self, request: &mut MpiRequest<T>, mpi_type: i32, dest: i32, tag: i32) {
        unsafe {
            MPI_Isend(
                request.buffer.as_mut_ptr() as *mut std::ffi::c_void,
                1,
                mpi_type,
                dest,
                tag,
                self.comm(),
                request.request.as_mut_ptr(),
            );
        }
    }

    pub fn irecv_vector<T>(&self, mpi_type: i32, source: i32, tag: i32) -> MpiRequest<Vec<T>> {
        let mut result = MpiRequest::<Vec<T>>::new();
        unsafe {
            MPI_Iprobe(
                source,
                tag,
                self.comm(),
                &raw mut result.flag,
                result.status.as_mut_ptr(),
            );
            result
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::mpi::{
        ffi::{MPI_INT32_T, MpiType},
        scope::init,
    };

    use super::*;

    #[test]
    fn test_probe() {
        let world = init();
        let comm = world.world();
        if comm.rank() == 0 {
            let mut flag = 0;
            let mut status = MaybeUninit::<MPI_Status>::uninit();
            unsafe {
                MPI_Iprobe(1, 1, comm.comm(), &mut flag, status.as_mut_ptr());
            }
            while flag == 0 {
                unsafe {
                    MPI_Iprobe(1, 1, comm.comm(), &mut flag, status.as_mut_ptr());
                }
            }
        } else {
            let mut buf = 42;
            comm.send(&buf, MPI_INT32_T, 0, 1);
        }
    }
}
