use super::{
    communicator::Communicator,
    ffi::{
        MPI_Cancel, MPI_Comm, MPI_Get_count, MPI_Iprobe, MPI_Irecv, MPI_Isend, MPI_Recv_init,
        MPI_Request, MPI_Request_free, MPI_Send_init, MPI_Start, MPI_Status, MPI_Test, MPI_Wait,
        MPI_Waitall, MPI_Waitany,
    },
};
use std::{any::Any, mem::MaybeUninit, path::Display};

/*------------------------------------------------------------------------
 *                           Mpi Future Traits
 *------------------------------------------------------------------------*/

/// MpiFuture is a trait that defines the interface for an asynchronous
/// operation in MPI. It provides methods to wait for the operation to
/// complete, test if it has completed, cancel it, and retrieve the
pub trait MpiFuture {
    fn wait(&mut self) -> i32;
    fn test(&mut self) -> bool;
    fn cancel(&mut self) -> i32;
    fn status(&self) -> MPI_Status;
    fn request(&self) -> MPI_Request;
    fn set_status(&mut self, status: MPI_Status);
}

/// MpiBufferedFuture is a trait that extends MpiFuture to support
/// buffered operations. It provides methods to create a new
/// MpiBufferedFuture instance for receiving and sending data.
pub trait MpiBufferedFuture<T>: MpiFuture
where
    T: Clone + Default,
{
    fn as_recv(comm: MPI_Comm, count: usize) -> Box<Self>;
    fn as_send(elems: &[T], comm: MPI_Comm) -> Box<Self>;
    fn data(&self) -> &[T];
}

/*------------------------------------------------------------------------------------------------
 *                                         Mpi Promise
 *------------------------------------------------------------------------------------------------*/

#[derive(PartialEq, Clone, Copy, Default, Debug)]
#[repr(i8)]
enum PromiseState {
    #[default]
    Idle,
    PreReceiving,
    PreSending,
    Receiving,
    Sending,
    Completed,
    Canceled,
}

impl std::fmt::Display for PromiseState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PromiseState::Idle => write!(f, "Idle"),
            PromiseState::PreReceiving => write!(f, "PreReceiving"),
            PromiseState::PreSending => write!(f, "PreSending"),
            PromiseState::Receiving => write!(f, "Receiving"),
            PromiseState::Sending => write!(f, "Sending"),
            PromiseState::Completed => write!(f, "Completed"),
            PromiseState::Canceled => write!(f, "Canceled"),
        }
    }
}

impl PromiseState {
    pub fn init_recv(self) -> Self {
        debug_assert!(self == PromiseState::Idle);
        PromiseState::PreReceiving
    }

    pub fn init_send(self) -> Self {
        debug_assert!(self == PromiseState::Idle);
        PromiseState::PreSending
    }

    pub fn recv(self) -> Self {
        debug_assert!(self == PromiseState::PreReceiving || self == PromiseState::Idle);
        PromiseState::Receiving
    }

    pub fn send(self) -> Self {
        debug_assert!(self == PromiseState::PreSending || self == PromiseState::Idle);
        PromiseState::Sending
    }
    pub fn complete(self) -> Self {
        debug_assert!(self == PromiseState::Receiving || self == PromiseState::Sending);
        PromiseState::Completed
    }
    pub fn cancel(self) -> Self {
        debug_assert!(self == PromiseState::Receiving || self == PromiseState::Sending);
        PromiseState::Canceled
    }
    pub fn restart(self) -> Self {
        debug_assert!(self == PromiseState::Canceled || self == PromiseState::Completed);
        PromiseState::Idle
    }

    pub fn start(self) -> Self {
        debug_assert!(
            self == PromiseState::PreReceiving || self == PromiseState::PreSending,
            "PromiseState::start() called on a non-pre state {}",
            self
        );
        if self == PromiseState::PreReceiving {
            PromiseState::Receiving
        } else {
            PromiseState::Sending
        }
    }
}

/// Represents a promise for an asynchronous operation in MPI.
/// It contains a buffer for the data, a status object,
/// a request object, and other metadata
#[derive(Debug, Default)]
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

impl<T> MpiBufferedFuture<T> for MpiPromise<T>
where
    T: Clone + Default,
{
    fn as_recv(comm: MPI_Comm, count: usize) -> Box<Self> {
        let mut this = Self {
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
        };
        Box::new(this)
    }

    fn as_send(buf: &[T], comm: MPI_Comm) -> Box<Self> {
        let mut this = Self {
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
        };
        Box::new(this)
    }
    fn data(&self) -> &[T] {
        &self.buffer.as_slice()
    }
}

impl<T> MpiFuture for MpiPromise<T> {
    fn wait(&mut self) -> i32 {
        self.wait();
        0
    }
    fn set_status(&mut self, status: MPI_Status) {
        self.status = status;
    }
    fn test(&mut self) -> bool {
        self.test()
    }

    fn cancel(&mut self) -> i32 {
        self.cancel();
        0
    }

    fn status(&self) -> MPI_Status {
        self.status
    }

    fn request(&self) -> MPI_Request {
        self.request
    }
}

impl<T> MpiPromise<T> {
    pub fn state(&self) -> PromiseState {
        self.state
    }
    pub fn request(&self) -> MPI_Request {
        self.request
    }
    pub fn cancel(&mut self) {
        self.state = self.state.cancel();
        {
            unsafe {
                MPI_Cancel(&mut self.request);
            }
        }
    }

    pub fn test(&mut self) -> bool {
        if self.state == PromiseState::Idle
            || self.state == PromiseState::Canceled
            || self.state == PromiseState::Completed
        {
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
        self.state = self.state.complete();
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

/// Communicator implementation for generate
/// MpiPromise instances for sending and receiving data
impl Communicator {
    /// Creates a new MpiPromise instance for receiving data
    pub fn irecv<T>(&self, count: usize, mpi_type: i32, source: i32, tag: i32) -> Box<MpiPromise<T>>
    where
        T: Clone + Default,
    {
        let mut result = MpiPromise::<T>::as_recv(self.comm(), count);
        result.mpi_type = mpi_type;
        result.source = source;
        result.tag = tag;
        result.dest = self.rank();
        unsafe {
            MPI_Irecv(
                result.buffer.as_mut_ptr() as *mut _,
                (count) as i32,
                mpi_type,
                source,
                tag,
                self.comm(),
                &mut result.request,
            );
        }
        result
    }

    /// Creates a new MpiPromise instance for sending data
    pub fn isend<T>(&self, elems: &[T], mpi_type: i32, dest: i32, tag: i32) -> Box<MpiPromise<T>>
    where
        T: Clone + Default,
    {
        let mut result = MpiPromise::<T>::as_send(elems, self.comm());
        result.mpi_type = mpi_type;
        result.source = self.rank();
        result.tag = tag;
        result.dest = dest;
        unsafe {
            MPI_Isend(
                result.buffer.as_mut_ptr() as *mut _,
                (result.buffer.len()) as i32,
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

/*------------------------------------------------------------------------------------------------
 *                                         Persistent Promise
 *------------------------------------------------------------------------------------------------*/

pub struct PersistentPromise<T>
where
    T: Clone + Default,
{
    innerPromise: MpiPromise<T>,
}

impl<T> MpiBufferedFuture<T> for PersistentPromise<T>
where
    T: Clone + Default,
{
    fn as_recv(comm: MPI_Comm, count: usize) -> Box<Self> {
        let mut this = Self {
            innerPromise: MpiPromise::<T>::default(),
        };
        this.innerPromise.buffer = vec![T::default(); count];
        this.innerPromise.status = MPI_Status::default();
        this.innerPromise.request = MPI_Request::default();
        this.innerPromise.comm = comm;
        this.innerPromise.state = PromiseState::PreReceiving;
        Box::new(this)
    }

    fn as_send(elems: &[T], comm: MPI_Comm) -> Box<Self> {
        let mut this = Self {
            innerPromise: MpiPromise::<T>::default(),
        };
        this.innerPromise.buffer = Vec::from(elems);
        this.innerPromise.status = MPI_Status::default();
        this.innerPromise.request = MPI_Request::default();
        this.innerPromise.comm = comm;
        this.innerPromise.state = PromiseState::PreSending;
        Box::new(this)
    }

    fn data(&self) -> &[T] {
        self.innerPromise.buffer.as_slice()
    }
}

impl<T> MpiFuture for PersistentPromise<T>
where
    T: Clone + Default,
{
    fn wait(&mut self) -> i32 {
        let prev_state = self.innerPromise.state;
        self.innerPromise.wait();
        self.innerPromise.state = match prev_state {
            PromiseState::Receiving => PromiseState::PreReceiving,
            PromiseState::Sending => PromiseState::PreSending,
            _ => self.innerPromise.state,
        };
        0
    }

    fn set_status(&mut self, status: MPI_Status) {
        self.innerPromise.status = status;
    }

    fn test(&mut self) -> bool {
        self.innerPromise.test()
    }

    fn cancel(&mut self) -> i32 {
        self.innerPromise.cancel();
        0
    }

    fn status(&self) -> MPI_Status {
        self.innerPromise.status
    }

    fn request(&self) -> MPI_Request {
        self.innerPromise.request
    }
}

impl<T> PersistentPromise<T>
where
    T: Clone + Default,
{
    pub fn start(&mut self) {
        self.innerPromise.state = self.innerPromise.state.start();
        unsafe {
            MPI_Start(&mut self.innerPromise.request);
        }
    }
}

impl<T> Drop for PersistentPromise<T>
where
    T: Clone + Default,
{
    fn drop(&mut self) {
        unsafe {
            MPI_Request_free(&mut self.innerPromise.request);
        }
    }
}

impl Communicator {
    pub fn recv_init<T>(
        &self,
        count: usize,
        mpi_type: i32,
        source: i32,
        tag: i32,
    ) -> Box<PersistentPromise<T>>
    where
        T: Clone + Default,
    {
        let mut result = PersistentPromise::<T>::as_recv(self.comm(), count);
        result.innerPromise.mpi_type = mpi_type;
        result.innerPromise.source = source;
        result.innerPromise.tag = tag;
        result.innerPromise.dest = self.rank();
        unsafe {
            MPI_Recv_init(
                result.innerPromise.buffer.as_mut_ptr() as *mut _,
                (count) as i32,
                mpi_type,
                source,
                tag,
                self.comm(),
                &mut result.innerPromise.request,
            );
        }
        result
    }

    pub fn send_init<T>(
        &self,
        elems: &[T],
        mpi_type: i32,
        dest: i32,
        tag: i32,
    ) -> Box<PersistentPromise<T>>
    where
        T: Clone + Default,
    {
        let mut result = PersistentPromise::<T>::as_send(elems, self.comm());
        result.innerPromise.mpi_type = mpi_type;
        result.innerPromise.source = self.rank();
        result.innerPromise.tag = tag;
        result.innerPromise.dest = dest;
        unsafe {
            MPI_Send_init(
                result.innerPromise.buffer.as_mut_ptr() as *mut _,
                (elems.len()) as i32,
                mpi_type,
                dest,
                tag,
                self.comm(),
                &mut result.innerPromise.request,
            );
        }
        result
    }
}

/*------------------------------------------------------------------------------------------------
 *                                         Group Functions
 *------------------------------------------------------------------------------------------------*/

pub struct WaitResult<'a> {
    pub status: MPI_Status,
    pub future: &'a mut Box<dyn MpiFuture>,
    pub index: usize,
}

pub fn waitany(futures: &mut [Box<dyn MpiFuture>]) -> WaitResult {
    let mut reqs = futures.iter().map(|f| f.request()).collect::<Vec<_>>();
    let mut status = MPI_Status::default();
    let mut index = 0;
    unsafe {
        MPI_Waitany(
            futures.len() as i32,
            reqs.as_mut_ptr(),
            &mut index,
            &mut status,
        );
    }
    futures[index as usize].set_status(status);
    WaitResult {
        status: status,
        future: &mut futures[index as usize],
        index: index as usize,
    }
}

pub fn waitall<'a>(futures: &[&'a dyn MpiFuture]) -> Vec<MPI_Status> {
    let mut reqs = futures.iter().map(|f| f.request()).collect::<Vec<_>>();
    let mut statuses = futures
        .iter()
        .map(|_| MPI_Status::default())
        .collect::<Vec<_>>();
    unsafe {
        MPI_Waitall(
            futures.len() as i32,
            reqs.as_mut_ptr(),
            statuses.as_mut_ptr(),
        );
    }
    statuses
}

impl dyn MpiFuture {
    pub fn as_promise<T>(&self) -> &MpiPromise<T>
    where
        T: Clone + Default,
    {
        unsafe {
            let future = *(&raw const self as *const *const dyn MpiFuture) as *const MpiPromise<T>;
            &*future
        }
    }

    pub fn as_persistent_promise<T>(&self) -> &PersistentPromise<T>
    where
        T: Clone + Default,
    {
        unsafe {
            let future =
                *(&raw const self as *const *const dyn MpiFuture) as *const PersistentPromise<T>;
            &*future
        }
    }

    pub fn as_mut_persistent_promise<T>(&mut self) -> &mut PersistentPromise<T>
    where
        T: Clone + Default,
    {
        unsafe {
            let future =
                *(&raw const self as *const *const dyn MpiFuture) as *const PersistentPromise<T>;
            &mut *(future as *mut PersistentPromise<T>)
        }
    }
    pub fn as_mut_promise<T>(&mut self) -> &mut MpiPromise<T>
    where
        T: Clone + Default,
    {
        unsafe {
            let future = *(&raw const self as *const *const dyn MpiFuture) as *const MpiPromise<T>;
            &mut *(future as *mut MpiPromise<T>)
        }
    }
}

/*------------------------------------------------------------------------------------------------
 *                                         Tests
 *------------------------------------------------------------------------------------------------*/

#[cfg(test)]
mod tests {
    use std::ops::Deref;

    use crate::mpi::{
        ffi::{MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_INT32_T, MPI_UINT8_T, MpiType},
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
            let mut promise = comm.irecv::<i32>(1, MPI_UINT8_T, 1, 1);
            promise.wait();
            assert!(!promise.test());
            assert!(promise.status().MPI_SOURCE == 1);
            assert!(promise.data()[0] == 42);
        } else {
            let mut buf = [42u8];
            let mut promise = comm.isend(&buf, MPI_UINT8_T, 0, 1);
            promise.wait();
            assert!(!promise.test());
        }
    }

    #[test]
    fn test_persistent_promises() {
        let universe = init();
        let comm = universe.world();
        let rank = comm.rank();
        if rank == 0 {
            let mut promise = comm.recv_init::<i32>(1, MPI_UINT8_T, MPI_ANY_TAG, MPI_ANY_SOURCE);
            for i in 0..10 {
                promise.start();
                promise.wait();
                assert!(
                    promise.innerPromise.state == PromiseState::PreReceiving,
                    "Promise state should be Receiving, but is {}",
                    promise.innerPromise.state
                );
                assert!(promise.status().MPI_SOURCE == 1);
                assert!(promise.data()[0] == i);
            }
        } else {
            let mut buf = [0u8];
            let mut promise = comm.send_init(&buf, MPI_UINT8_T, 0, 1);
            for i in 0..10 {
                promise.innerPromise.buffer[0] = i;
                promise.start();
                promise.wait();
            }
        }
    }

    #[test]
    fn test_waitany() {
        let world = init();
        let comm = world.world();
        let rank = comm.rank();
        if rank == 0 {
            let mut promise1 = comm.irecv::<i32>(1, MPI_UINT8_T, MPI_ANY_SOURCE, MPI_ANY_TAG);
            let mut promise2 = comm.irecv::<i32>(1, MPI_UINT8_T, 1, 2);
            let mut futures: [Box<dyn MpiFuture>; 2] = [promise2, promise1];
            let res = waitany(&mut futures);
            assert_eq!(res.status.MPI_SOURCE, 1);
            assert_eq!(res.status.MPI_TAG, 1);
            assert_eq!(res.future.status().MPI_SOURCE, 1);
            assert_eq!(res.future.status().MPI_TAG, 1);
            assert_eq!(res.future.as_promise::<i32>().buffer[0], 42);
            assert_eq!(res.index, 1);
        } else {
            let mut buf = [42u8];
            comm.send(&buf, MPI_UINT8_T, 0, 1);
        }
    }

    #[test]
    fn test_waitany_vector() {
        let world = init();
        let comm = world.world();
        let rank = comm.rank();
        if rank == 0 {
            let mut promise1 = comm.irecv::<i32>(2, MPI_INT32_T, MPI_ANY_SOURCE, MPI_ANY_TAG);
            let mut promise2 = comm.irecv::<i32>(2, MPI_INT32_T, 1, 2);
            let mut futures: [Box<dyn MpiFuture>; 2] = [promise2, promise1];
            let res = waitany(&mut futures);
            assert_eq!(res.status.MPI_SOURCE, 1);
            assert_eq!(res.status.MPI_TAG, 1);
            assert_eq!(res.future.status().MPI_SOURCE, 1);
            assert_eq!(res.future.status().MPI_TAG, 1);
            assert_eq!(res.future.as_promise::<i32>().buffer[0], 42);
            assert_eq!(res.future.as_promise::<i32>().buffer[1], 25);
            assert_eq!(res.index, 1);
        } else {
            let mut buf = [42, 25];
            comm.send_vector(&buf, MPI_INT32_T, 0, 1);
        }
    }
}
