use std::ffi::{CStr, CString};

use rayon::{
    iter::{IntoParallelIterator, ParallelBridge, ParallelIterator},
    string,
};

use crate::{
    ARGS,
    gpu::{md5_brute, md5_cpu, md5_transform},
};

pub enum ComputeContext<'a> {
    Brute(usize, usize, &'a CString),
    Chunked(Vec<u8>, Vec<u8>),
}

pub enum ComputeResult {
    BruteResult(Option<String>),
    ChunkedResult(Vec<String>),
}

pub fn compute(context: ComputeContext) -> ComputeResult {
    match context {
        ComputeContext::Brute(start, end, target) => brute_mode(start, end, target),
        ComputeContext::Chunked(strings, sizes) => chunked_mode(strings, sizes),
    }
}

struct SplittedIterator<'a> {
    data: &'a Vec<u8>,
    sizes: &'a Vec<u8>,
    idx: usize,
    dataptr: usize,
}

impl<'a> Iterator for SplittedIterator<'a> {
    type Item = CString;

    fn next(&mut self) -> Option<Self::Item> {
        if self.idx >= self.sizes.len() {
            None
        } else {
            let result = Some(
                CString::new(
                    &self.data[self.dataptr..(self.dataptr + self.sizes[self.idx] as usize)],
                )
                .unwrap(),
            );
            self.dataptr += self.sizes[self.idx] as usize;
            self.idx += 1;
            result
        }
    }
}

impl<'a> SplittedIterator<'a> {
    pub fn new(data: &'a Vec<u8>, sizes: &'a Vec<u8>) -> Self {
        let mut offsets = Vec::<u32>::with_capacity(sizes.len());
        offsets.push(0);
        for i in 1..sizes.len() {
            offsets.push(offsets[i - 1] as u32 + sizes[i] as u32);
        }
        Self {
            data: data,
            sizes: sizes,
            dataptr: 0,
            idx: 0,
        }
    }
}

pub fn chunked_mode(string: Vec<u8>, sizes: Vec<u8>) -> ComputeResult {
    let threads = { ARGS.lock().unwrap().num_threads };
    let gpuon = { ARGS.lock().unwrap().use_gpu };
    if gpuon {
        ComputeResult::ChunkedResult(md5_transform(&string, &sizes, threads as u32))
    } else {
        let mut itr = SplittedIterator::new(&string, &sizes);
        let result = itr.par_bridge().map(|x| md5_cpu(&x)).collect();
        ComputeResult::ChunkedResult(result)
    }
}

pub fn brute_mode(start: usize, end: usize, target: &CString) -> ComputeResult {
    let threads = { ARGS.lock().unwrap().num_threads };
    let brutestart = { ARGS.lock().unwrap().brutestart };
    ComputeResult::BruteResult(md5_brute(
        start,
        end,
        &target,
        threads as u32,
        brutestart as u32,
    ))
}

impl ComputeResult {
    pub fn unwrap_brute(self) -> Option<String> {
        match self {
            ComputeResult::BruteResult(result) => result,
            _ => panic!("Expected BruteResult"),
        }
    }

    pub fn unwrap_chunked(self) -> Vec<String> {
        match self {
            ComputeResult::ChunkedResult(result) => result,
            _ => panic!("Expected ChunkedResult"),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::sequence_generator::{ChunkGenerator, GeneratorResult, SequenceGenerator};

    use super::*;

    #[test]
    fn test_splitted_iterator() {
        let mut generator = SequenceGenerator::new(4);
        let result = generator.generate_flatten_chunk(100);
        let mut itr = SplittedIterator::new(&result.strings, &result.sizes);
        let itrresult: Vec<CString> = itr.collect();
        assert_eq!(itrresult.len(), 100);
        assert_eq!(itrresult[0].to_str().unwrap(), "!!!!");
        for i in 0..10 {
            println!("{}: {}", i, itrresult[i].to_str().unwrap());
        }
    }

    #[test]
    fn test_splitted_iterator_parallel() {
        let mut generator = SequenceGenerator::new(4);
        let result = generator.generate_flatten_chunk(100);
        let mut itr = SplittedIterator::new(&result.strings, &result.sizes);
        let itrresult: Vec<CString> = itr.par_bridge().collect();
        assert_eq!(itrresult.len(), 100);
        for i in 0..10 {
            println!("{}: {}", i, itrresult[i].to_str().unwrap());
        }
    }
}
