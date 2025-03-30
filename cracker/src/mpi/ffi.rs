include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

pub const MPI_FLOAT_INT: u32 = 0x8c000000;
pub const MPI_DOUBLE_INT: u32 = 0x8c000001;
pub const MPI_LONG_INT: u32 = 0x8c000002;
pub const MPI_SHORT_INT: u32 = 0x8c000003;
pub const MPI_2INT: i32 = 0x4c000816;
pub const MPI_LONG_DOUBLE_INT: u32 = 0x8c000004;
pub const MPI_REAL4: i32 = 0x4c000427;
pub const MPI_REAL8: i32 = 0x4c000829;
pub const MPI_REAL16: i32 = 0x4c00102b;
pub const MPI_COMPLEX8: i32 = 0x4c000828;
pub const MPI_COMPLEX16: i32 = 0x4c00102a;
pub const MPI_COMPLEX32: i32 = 0x4c00202c;
pub const MPI_INTEGER1: i32 = 0x4c00012d;
pub const MPI_INTEGER2: i32 = 0x4c00022f;
pub const MPI_INTEGER4: i32 = 0x4c000430;
pub const MPI_INTEGER8: i32 = 0x4c000831;
pub const MPI_INT8_T: i32 = 0x4c000137;
pub const MPI_INT16_T: i32 = 0x4c000238;
pub const MPI_INT32_T: i32 = 0x4c000439;
pub const MPI_INT64_T: i32 = 0x4c00083a;
pub const MPI_UINT8_T: i32 = 0x4c00013b;
pub const MPI_UINT16_T: i32 = 0x4c00023c;
pub const MPI_UINT32_T: i32 = 0x4c00043d;
pub const MPI_UINT64_T: i32 = 0x4c00083e;
pub const MPI_COMM_WORLD: MPI_Comm = 0x44000000 as MPI_Comm;
pub const MPI_COMM_SELF: MPI_Comm = 0x44000001 as MPI_Comm;

pub const MPI_BOTTOM: i32 = 0;
pub const MPI_IN_PLACE: i32 = -1;
pub const MPI_BUFFER_AUTOMATIC: i32 = -2;
pub const MPI_STATUS_IGNORE: *mut MPI_Status = 1 as *mut MPI_Status;
pub const MPI_STATUSES_IGNORE: i32 = 1;
pub const MPI_ERRCODES_IGNORE: i32 = 0;
pub const MPI_ARGV_NULL: i32 = 0;
pub const MPI_ARGVS_NULL: i32 = 0;

#[repr(u32)]
pub enum MpiType {
    MPI_FLOAT_INT = 0x8c000000,
    MPI_DOUBLE_INT = 0x8c000001,
    MPI_LONG_INT = 0x8c000002,
    MPI_SHORT_INT = 0x8c000003,
    MPI_2INT = 0x4c000816,
    MPI_LONG_DOUBLE_INT = 0x8c000004,
    MPI_REAL4 = 0x4c000427,
    MPI_REAL8 = 0x4c000829,
    MPI_REAL16 = 0x4c00102b,
    MPI_COMPLEX8 = 0x4c000828,
    MPI_COMPLEX16 = 0x4c00102a,
    MPI_COMPLEX32 = 0x4c00202c,
    MPI_INTEGER1 = 0x4c00012d,
    MPI_INTEGER2 = 0x4c00022f,
    MPI_INTEGER4 = 0x4c000430,
    MPI_INTEGER8 = 0x4c000831,
    MPI_INT8_T = 0x4c000137,
    MPI_INT16_T = 0x4c000238,
    MPI_INT32_T = 0x4c000439,
    MPI_INT64_T = 0x4c00083a,
    MPI_UINT8_T = 0x4c00013b,
    MPI_UINT16_T = 0x4c00023c,
    MPI_UINT32_T = 0x4c00043d,
    MPI_UINT64_T = 0x4c00083e,
}

impl Into<i32> for MpiType {
    fn into(self) -> i32 {
        self as i32
    }
}
impl Into<u32> for MpiType {
    fn into(self) -> u32 {
        self as u32
    }
}
