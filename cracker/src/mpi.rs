use std::{any::Any, mem::MaybeUninit, vec};

use mpi::{
    Rank,
    ffi::{
        MPI_F_STATUS_IGNORE, MPI_Finalize, MPI_Request, MPI_Status, MPI_Test, MPI_Wait, MPI_Waitany,
    },
    point_to_point::ReceiveFuture,
    raw::AsRaw,
    request::{
        self, CancelGuard, LocalScope, Request, RequestCollection, Scope, WaitGuard, scope,
        wait_any,
    },
    topology::SimpleCommunicator,
    traits::{
        AsDatatype, Buffer, Communicator, CommunicatorCollectives, Destination, Root, Source,
    },
};

use crate::{
    ARGS,
    dictionary_reader::DictionaryReader,
    sequence_generator::{ChunkGenerator, SequenceGenerator},
};

#[repr(i32)]
enum MpiTags {
    DATA,
    RESULT,
    TERMINATE,
    ALIVE,
    SIZES,
}

impl Into<i32> for MpiTags {
    fn into(self) -> i32 {
        self as i32
    }
}

impl From<i32> for MpiTags{
    fn from(value: i32) -> Self {
        match value {
            0 => MpiTags::DATA,
            1 => MpiTags::RESULT,
            2 => MpiTags::TERMINATE,
            3 => MpiTags::ALIVE,
            4 => MpiTags::SIZES,
            _ => panic!("Invalid MPI tag"),
        }
    }
}

pub fn run_mpi() {}


struct MpiRequestCollection {
    requests: Vec<*mut MPI_Request>,
}

impl MpiRequestCollection {
    pub fn new() -> Self {
        Self {
            requests: vec![],
        }
    }

    pub fn wait_any(&mut self) -> Option<(usize, MPI_Status)> {
        let mut status = MaybeUninit::<MPI_Status>::zeroed();
        let mut index: i32 = 0;
        unsafe {
            MPI_Waitany(
                self.requests.len() as i32,
                *self.requests.as_mut_ptr(),
                &raw mut index,
                status.as_mut_ptr(),
            );
        }
        if index != -1 {
            None
        }
        else {
            Some((index as usize, unsafe { status.assume_init() }))
        }
    }

    pub fn add_request<'a,T>(&mut self, request: &Request<'a, T, &LocalScope<'a>>) {
        self.requests.push(request.as_raw() as *mut MPI_Request);
    }
}

pub fn completed<'a, T>(request: &Request<'a, T, &LocalScope<'a>>) -> bool {
    let ptr: *mut MPI_Request = request.as_raw() as *mut MPI_Request;
    unsafe {
        let mut flag = 0;

        MPI_Test(
            ptr,
            &mut flag,
            MPI_F_STATUS_IGNORE as *mut mpi::ffi::MPI_Status,
        );
        flag != 0
    }
}

enum SendContext{
    DATA(DictionaryReader),
    BRUTE([usize;2]),
    None
}

pub fn root(communicator: SimpleCommunicator) {
    let mut result = [0i8; 32];
    let mut requests = MpiRequestCollection::new();
    mpi::request::scope(|f| {
        let mut stopreq = communicator.this_process().immediate_receive_into_with_tag(
            f,
            &mut result,
            MpiTags::RESULT.into(),
        );

        requests.add_request(&stopreq);
        let mut generator: SendContext = SendContext::None;
        if ARGS.lock().unwrap().use_dictionary() {
            let mut loader =
                DictionaryReader::new(ARGS.lock().unwrap().dictionary.as_str()).unwrap();
            generator = SendContext::DATA(loader);
        }
        while !completed(&stopreq) {
            if ARGS.lock().unwrap().use_dictionary() {
                root_chunked(&communicator, &mut generator.into(), &mut requests);
            } else {
                root_bruter(&communicator,&mut requests);
            }
        }
    });
}

pub fn root_bruter(communicator: &SimpleCommunicator, requests: &mut MpiRequestCollection) {
    let mut buf : u8 = 0 ;
    scope(|f| {
        let mut data_request = communicator.this_process().
        immediate_receive_into_with_tag(f, &mut buf, MpiTags::DATA.into());
        requests.add_request(&data_request);
        let mut req = requests.wait_any();
        if let Some(result) = req{
            match MpiTags::from(result.1.MPI_TAG) {
                MpiTags::DATA => {
                    
                },
                _=>{}
            }
        }
        else{
            println!("Error: No request completed");
            return;
        }

    });
}

pub fn root_chunked(communicator: &SimpleCommunicator, generator: &mut impl ChunkGenerator, requests: &mut MpiRequestCollection) {
    let mut buf : u8= 0;
    scope(|f|{
        let mut data_request = communicator.this_process().
        immediate_receive_into_with_tag(f, &mut buf, MpiTags::DATA.into());
        requests.add_request(&data_request);

        let mut req = requests.wait_any();
        if let Some(result) = req{
            match MpiTags::from(result.1.MPI_TAG) {
                MpiTags::DATA => {
                    let mut chunk= generator.generate_flatten_chunk(ARGS.lock().unwrap().chunk_size as usize);
                    communicator.process_at_rank(result.1.MPI_SOURCE).send_with_tag(&chunk.strings, MpiTags::DATA.into());
                    communicator.process_at_rank(result.1.MPI_SOURCE).send_with_tag(&chunk.sizes, MpiTags::SIZES.into());
                },
                MpiTags::TERMINATE => {
                    return;
                }
                _=>{}   
            }
        }
        else{
            println!("Error: No request completed");
        }
    });
}

pub fn worker(communicator: SimpleCommunicator) {}

pub fn worker_bruter(communicator: SimpleCommunicator) {}

pub fn worker_chunked(communicator: SimpleCommunicator) {}
