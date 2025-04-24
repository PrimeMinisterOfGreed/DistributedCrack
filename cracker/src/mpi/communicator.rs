use std::{mem::MaybeUninit, os::raw::c_void, ptr::null_mut};

use log::debug;

use crate::mpi::ffi::{MPI_Bcast, MPI_Comm};

use super::ffi::{
    MPI_Comm_rank, MPI_Comm_size, MPI_Datatype, MPI_Finalize, MPI_Get_count, MPI_Init, MPI_Probe,
    MPI_Recv, MPI_Request, MPI_Send, MPI_Status, ompi_datatype_t, *,
};

#[derive(Debug, Clone, Copy)]
pub struct Communicator {
    comm: MPI_Comm,
    rank: i32,
    size: i32,
}

impl Default for Communicator {
    fn default() -> Self {
        let mut rank = 0;
        let mut size = 0;
        unsafe {
            MPI_Comm_rank(MPI_COMM_WORLD, &mut rank);
            MPI_Comm_size(MPI_COMM_WORLD, &mut size);
        }
        Communicator {
            comm: MPI_COMM_WORLD,
            rank,
            size,
        }
    }
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

    pub fn send<T>(&self, buf: &T, mpi_type: MpiDatatype, dest: i32, tag: i32)
    where
        T: Sized,
    {
        unsafe {
            MPI_Send(
                buf as *const T as *const std::ffi::c_void,
                1,
                mpi_type.into(),
                dest,
                tag,
                self.comm,
            );
        }
    }

    pub fn recv<T>(&self, mpi_type: MpiDatatype, source: i32, tag: i32) -> T {
        let mut result = MaybeUninit::<T>::uninit();
        unsafe {
            let mut status = MaybeUninit::<MPI_Status>::uninit();
            MPI_Recv(
                result.as_mut_ptr() as *mut std::ffi::c_void,
                1,
                mpi_type.into(),
                source,
                tag,
                self.comm,
                status.as_mut_ptr(),
            );
            result.assume_init()
        }
    }

    pub fn send_vector<T>(&self, buf: &[T], mpi_type: MpiDatatype, dest: i32, tag: i32)
    where
        T: Sized,
    {
        unsafe {
            MPI_Send(
                buf.as_ptr() as *const std::ffi::c_void,
                (buf.len()) as i32,
                mpi_type.into(),
                dest,
                tag,
                self.comm,
            );
        }
    }

    pub fn recv_vector<T>(&self, mpi_type: MpiDatatype, source: i32, tag: i32) -> Vec<T>
    where
        T: Default + Clone,
    {
        let mut count = 0;
        unsafe {
            let mut status = MaybeUninit::<MPI_Status>::uninit();
            MPI_Probe(source, tag, self.comm, status.as_mut_ptr());
            MPI_Get_count(status.as_ptr(), mpi_type.into(), &mut count);
            let mut result = vec![T::default(); count as usize];
            MPI_Recv(
                result.as_mut_ptr() as *mut std::ffi::c_void,
                count,
                mpi_type.into(),
                source,
                tag,
                self.comm,
                status.as_mut_ptr(),
            );
            result
        }
    }

    pub fn send_object<T>(&self, obj: &T, dest: i32, tag: i32)
    where
        T: Sized,
    {
        unsafe {
            MPI_Send(
                obj as *const T as *const std::ffi::c_void,
                size_of::<T>() as i32,
                MPI_UINT8_T.into(),
                dest,
                tag,
                self.comm,
            );
        }
    }

    pub fn recv_object<T>(&self, source: i32, tag: i32) -> T
    where
        T: Sized,
    {
        let mut result = MaybeUninit::<T>::uninit();
        unsafe {
            let mut status = MaybeUninit::<MPI_Status>::uninit();
            MPI_Recv(
                result.as_mut_ptr() as *mut std::ffi::c_void,
                size_of::<T>() as i32,
                MPI_UINT8_T.into(),
                source,
                tag,
                self.comm,
                status.as_mut_ptr(),
            );
            result.assume_init()
        }
    }

    pub fn send_object_vector<T>(&self, obj: &[T], dest: i32, tag: i32)
    where
        T: Sized,
    {
        unsafe {
            MPI_Send(
                obj.as_ptr() as *const std::ffi::c_void,
                (obj.len() * size_of::<T>()) as i32,
                MPI_UINT8_T.into(),
                dest,
                tag,
                self.comm,
            );
        }
    }
    pub fn recv_object_vector<T>(&self, source: i32, tag: i32) -> Vec<T>
    where
        T: Sized,
    {
        let mut count = 0;
        unsafe {
            let mut status = MaybeUninit::<MPI_Status>::uninit();
            MPI_Probe(source, tag, self.comm, status.as_mut_ptr());
            MPI_Get_count(status.as_ptr(), MPI_UINT8_T.into(), &mut count);
            let mut result: Vec<T> = Vec::with_capacity(count as usize);
            MPI_Recv(
                result.as_mut_ptr() as *mut std::ffi::c_void,
                (count * size_of::<T>() as i32) as i32,
                MPI_UINT8_T.into(),
                source,
                tag,
                self.comm,
                status.as_mut_ptr(),
            );
            result.set_len((count / size_of::<T>() as i32) as usize);
            result
        }
    }

    pub fn broadcast<T>(
        &self,
        buffer: &mut [T],
        root: i32,
        datatype: MpiDatatype,
    ) -> Result<(), i32> {
        unsafe {
            let res = MPI_Bcast(
                buffer.as_mut_ptr() as *mut c_void,
                buffer.len() as i32,
                datatype.into(),
                root,
                self.comm,
            );
            if res != 0 {
                return Err(res);
            }
            Ok(())
        }
    }
}

macro_rules! mpi_type {
    ($export:ident ,$name:ident) => {
        pub const $export: MpiDatatype = MpiDatatype {
            datatype: unsafe { &raw mut $name as *mut ompi_datatype_t },
        };
    };
    () => {};
}

mpi_type!(MPI_UINT8_T, ompi_mpi_uint8_t);
mpi_type!(MPI_INT, ompi_mpi_int);
mpi_type!(MPI_FLOAT, ompi_mpi_float);
mpi_type!(MPI_DOUBLE, ompi_mpi_double);
mpi_type!(MPI_LONG, ompi_mpi_long);
mpi_type!(MPI_CHAR, ompi_mpi_char);
mpi_type!(MPI_SHORT, ompi_mpi_short);
mpi_type!(MPI_UNSIGNED_CHAR, ompi_mpi_unsigned_char);
mpi_type!(MPI_UNSIGNED_LONG, ompi_mpi_unsigned_long);
mpi_type!(MPI_UNSIGNED_LONG_LONG, ompi_mpi_unsigned_long_long);
mpi_type!(MPI_UNSIGNED_SHORT, ompi_mpi_unsigned_short);
mpi_type!(MPI_DOUBLE_INT, ompi_mpi_double_int);
mpi_type!(MPI_LONG_INT, ompi_mpi_long_int);
mpi_type!(MPI_LONG_DOUBLE, ompi_mpi_long_double);
mpi_type!(MPI_LONG_LONG_INT, ompi_mpi_long_long_int);
mpi_type!(MPI_SHORT_INT, ompi_mpi_short_int);
mpi_type!(MPI_UINT16_T, ompi_mpi_uint16_t);
mpi_type!(MPI_UINT32_T, ompi_mpi_uint32_t);
mpi_type!(MPI_UINT64_T, ompi_mpi_uint64_t);
mpi_type!(MPI_INT8_T, ompi_mpi_int8_t);
mpi_type!(MPI_INT16_T, ompi_mpi_int16_t);
mpi_type!(MPI_INT32_T, ompi_mpi_int32_t);
mpi_type!(MPI_INT64_T, ompi_mpi_int64_t);
mpi_type!(MPI_FLOAT_INT, ompi_mpi_float_int);

impl Into<*mut ompi_datatype_t> for MpiDatatype {
    fn into(self) -> *mut ompi_datatype_t {
        self.datatype
    }
}

pub const MPI_COMM_WORLD: MPI_Comm = unsafe { &raw mut ompi_mpi_comm_world as MPI_Comm };

/* -------------------------------------------------------------------------- */
/*                          FFI Interface for OpenMpi                         */
/* -------------------------------------------------------------------------- */

#[derive(Debug, Clone, Copy)]
pub struct MpiRequest {
    pub request: MPI_Request,
}

impl Default for MpiRequest {
    fn default() -> Self {
        MpiRequest {
            request: null_mut(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MpiDatatype {
    pub datatype: MPI_Datatype,
}

impl Default for MpiDatatype {
    fn default() -> Self {
        MpiDatatype {
            datatype: null_mut(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MpiStatus {
    pub status: MPI_Status,
}
impl Default for MpiStatus {
    fn default() -> Self {
        MpiStatus {
            status: MPI_Status {
                _cancelled: 0,
                _ucount: 0,
                MPI_SOURCE: 0,
                MPI_TAG: 0,
                MPI_ERROR: 0,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{mem::MaybeUninit, os::raw::c_void, ptr::null_mut, str::FromStr};

    use crate::{
        mpi::{
            ffi::{
                MPI_F_STATUS_IGNORE, MPI_Finalize, MPI_Get_count, MPI_Init, MPI_PROC_NULL,
                MPI_Probe, MPI_Recv, MPI_Send, MPI_Status,
            },
            scope::init,
        },
        timers::ClockStats,
    };

    use super::*;

    #[test]
    fn test_init() {
        let world = init();
        println!("Rank: {}", world.world().rank);
    }

    #[test]
    fn test_custom_type() {
        let world = init();
        let rank = world.world().rank;
        let comm = world.world();

        if rank == 0 {
        } else {
            let res = comm.recv::<i32>(MPI_INT32_T, 0, 1);
            assert_eq!(res, 10);
        }
    }

    #[test]
    fn test_send() {
        let world = init();
        let rank = world.world().rank;
        let comm = world.world();
        if rank == 0 {
            let res = comm.recv::<i32>(MPI_INT32_T.into(), 1, 1);
            assert_eq!(res, 10);
        } else {
            let mut s = 10;
            comm.send(&s, MPI_INT32_T, 0, 1);
        }
    }

    struct CustomObject {
        name: [u8; 13],
        id: i32,
        value: f64,
    }

    #[test]
    fn test_send_object() {
        let world = init();
        let rank = world.world().rank;
        let comm = world.world();
        if rank == 0 {
            let res = comm.recv_object::<CustomObject>(1, 1);
            assert_eq!(res.id, 10);
            assert_eq!(res.value, 3.14);
            assert_eq!(res.name, *b"Hello, world!");
        } else {
            let mut s = CustomObject {
                name: *b"Hello, world!",
                id: 10,
                value: 3.14,
            };
            comm.send_object(&s, 0, 1);
        }
    }

    #[test]
    fn test_send_object_vector() {
        let world = init();
        let rank = world.world().rank;
        let comm = world.world();
        if rank == 0 {
            let res = comm.recv_object_vector::<CustomObject>(1, 1);
            assert_eq!(res[0].id, 10);
            assert_eq!(res[0].value, 3.14);
            assert_eq!(res[0].name, *b"Hello, world!");
            assert_eq!(res[1].id, 20);
            assert_eq!(res[1].value, 2.71);
            assert_eq!(res[1].name, *b"Hello, world!");
        } else {
            let mut s = vec![
                CustomObject {
                    name: *b"Hello, world!",
                    id: 10,
                    value: 3.14,
                },
                CustomObject {
                    name: *b"Hello, world!",
                    id: 20,
                    value: 2.71,
                },
            ];
            comm.send_object_vector(&s, 0, 1);
        }
    }

    #[test]
    fn test_broadcast() {
        let world = init();
        let rank = world.world().rank;
        let comm = world.world();
        if rank == 0 {
            let mut s = [0u8; 1024];
            let data = b"Hello, world!";
            s[..data.len()].copy_from_slice(data);
            comm.broadcast(&mut s, 0, MPI_UINT8_T).unwrap();
        } else {
            let mut s = [0u8; 1024];
            comm.broadcast(&mut s, 0, MPI_UINT8_T).unwrap();
            println!("s.len(): {}", String::from_utf8_lossy(&s));
        }
    }
}
