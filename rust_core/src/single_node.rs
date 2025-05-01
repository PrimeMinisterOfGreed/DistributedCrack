use std::ffi::CString;

use rayon::result;

use crate::options::ARGS;
use crate::{
    compute_context::{ComputeContext, compute},
    sequence_generator::{ChunkGenerator, SequenceGenerator},
};

pub fn single_node_routine() -> String {
    let dict = { ARGS.lock().unwrap().dictionary != "NONE" };
    if dict {
        // Dictionary routine
        chunked_routine()
    } else {
        // Brute force routine
        brute_routine()
    }
}

pub fn chunked_routine() -> String {
    todo!()
}

pub fn brute_routine() -> String {
    let mut result = String::new();
    let chunksize = ARGS.lock().unwrap().chunk_size;
    let mut addresses: [usize; 2] = [0, chunksize as usize];
    let target = CString::new(ARGS.lock().unwrap().target_md5.clone()).unwrap();
    while result.is_empty() {
        let context = ComputeContext::Brute(addresses[0], addresses[1], &target);
        let res = compute(context);
        if let Some(res) = res {
            result = res;
        }
    }
    result
}
