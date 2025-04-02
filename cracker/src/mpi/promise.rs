use super::{
    communicator::Communicator,
    ffi::{
        MPI_Cancel, MPI_Comm, MPI_Get_count, MPI_Iprobe, MPI_Irecv, MPI_Isend, MPI_Request,
        MPI_Status, MPI_Test, MPI_Wait,
    },
};
use std::{any::Any, mem::MaybeUninit};

pub trait MpiFuture {
    fn wait(&mut self) -> i32;
    fn test(&mut self) -> bool;
    fn cancel(&mut self) -> i32;
    fn status(&self) -> MPI_Status;
    fn request(&self) -> MPI_Request;
}

#[derive(PartialEq, Clone, Copy)]
#[repr(i8)]
enum PromiseState {
    Idle,
    Receiving,
    Sending,
    Completed,
    Canceled,
}

pub struct MpiPromise<T> {
    buffer: Vec<T>,
    status: MPI_Status,
    request: MPI_Request,
    comm: MPI_Comm,
    flag: i32,
    mpi_type: i32,
    state: PromiseState,
    source: i32,
    tag: i32,
    dest: i32,
}

impl<T> MpiPromise<T> {
    fn as_recv(comm: MPI_Comm, count: usize) -> Self
    where
        T: Clone + Default,
    {
        Self {
            buffer: vec![T::default(); count],
            status: MPI_Status::default(),
            request: MPI_Request::default(),
            comm: comm,
            state: PromiseState::Receiving,
            flag: 0,
            mpi_type: 0,
            source: 0,
            tag: 0,
            dest: 0,
        }
    }

    fn as_send(buf: &[T], comm: MPI_Comm) -> Self
    where
        T: Clone + Default,
    {
        Self {
            buffer: Vec::from(buf),
            status: MPI_Status::default(),
            request: MPI_Request::default(),
            comm: comm,
            state: PromiseState::Sending,
            flag: 0,
            mpi_type: 0,
            source: 0,
            tag: 0,
            dest: 0,
        }
    }
    pub fn state(&self) -> PromiseState {
        self.state
    }
    pub fn request(&self) -> MPI_Request {
        self.request
    }
    pub fn cancel(&mut self) {
        if self.state() != PromiseState::Idle
            && self.state() != PromiseState::Completed
            && self.state() != PromiseState::Canceled
        {
            unsafe {
                MPI_Cancel(&mut self.request);
            }
            self.state = PromiseState::Canceled;
        }
    }

    pub fn test(&mut self) -> bool {
        if self.state == PromiseState::Idle {
            return false;
        }
        unsafe {
            MPI_Test(&mut self.request, &mut self.flag, &mut self.status);
        }
        self.flag != 0
    }

    pub fn status(&self) -> MPI_Status {
        self.status
    }

    pub fn wait(&mut self) {
        unsafe {
            MPI_Wait(&raw mut self.request, &mut self.status);
        }
    }
    pub fn probe(&mut self) {
        if self.state == PromiseState::Idle {
            return;
        }
        unsafe {
            MPI_Iprobe(
                self.source,
                self.tag,
                self.comm,
                &mut self.flag,
                &mut self.status,
            );
        }
    }
    pub fn data(&self) -> &[T] {
        &self.buffer
    }
}

impl Communicator {
    pub fn irecv<T>(&self, count: usize, mpi_type: i32, source: i32, tag: i32) -> Box<MpiPromise<T>>
    where
        T: Clone + Default,
    {
        let mut result = Box::new(MpiPromise::<T>::as_recv(self.comm(), count));
        result.mpi_type = mpi_type;
        result.source = source;
        result.tag = tag;
        result.dest = self.rank();
        unsafe {
            MPI_Irecv(
                result.buffer.as_mut_ptr() as *mut _,
                count as i32,
                mpi_type,
                source,
                tag,
                self.comm(),
                &mut result.request,
            );
        }
        result
    }

    pub fn isend<T>(&self, elems: &[T], mpi_type: i32, dest: i32, tag: i32) -> Box<MpiPromise<T>>
    where
        T: Clone + Default,
    {
        let mut result = Box::new(MpiPromise::<T>::as_send(elems, self.comm()));
        result.mpi_type = mpi_type;
        result.source = self.rank();
        result.tag = tag;
        result.dest = dest;
        unsafe {
            MPI_Isend(
                result.buffer.as_mut_ptr() as *mut _,
                result.buffer.len() as i32,
                mpi_type,
                dest,
                tag,
                self.comm(),
                &mut result.request,
            );
        }
        result
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

    #[test]
    fn test_promise() {
        let world = init();
        let comm = world.world();
        let rank = comm.rank();
        assert!(
            comm.size() >= 2,
            "This test requires at least 2 processes, provided {}",
            comm.size()
        );
        if rank == 0 {
            let mut promise = comm.irecv::<i32>(1, MPI_INT32_T, 1, 1);
            promise.wait();
            assert!(promise.test());
            assert!(promise.status().MPI_SOURCE == 1);
            assert!(promise.data()[0] == 42);
        } else {
            let mut buf = [42u8];
            let mut promise = comm.isend(&buf, MPI_INT32_T, 0, 1);
            promise.wait();
            assert!(promise.test());
            assert!(promise.status().MPI_SOURCE == 0);
        }
    }
}
