include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

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
