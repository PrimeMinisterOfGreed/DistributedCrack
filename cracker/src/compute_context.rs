use std::ffi::{CStr, CString};

use rayon::{
    ThreadPool, ThreadPoolBuilder,
    iter::{IntoParallelIterator, ParallelBridge, ParallelIterator},
    string,
};

use crate::{
    ARGS,
    gpu::{md5_brute, md5_cpu, md5_transform},
    sequence_generator::{ChunkGenerator, SequenceGenerator},
};

pub enum ComputeContext<'a> {
    Brute(usize, usize, &'a CString),
    Chunked(Vec<u8>, Vec<u8>, &'a CString),
}

pub fn compute(context: ComputeContext) -> Option<String> {
    match context {
        ComputeContext::Brute(start, end, target) => brute_mode(start, end, target),
        ComputeContext::Chunked(strings, sizes, target) => chunked_mode(strings, sizes, target),
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
                    &self.data[self.dataptr..(self.dataptr + (self.sizes[self.idx]) as usize)],
                )
                .unwrap(),
            );
            self.dataptr += (self.sizes[self.idx]) as usize;
            self.idx += 1;
            result
        }
    }
}

impl<'a> SplittedIterator<'a> {
    pub fn new(data: &'a Vec<u8>, sizes: &'a Vec<u8>) -> Self {
        Self {
            data: data,
            sizes: sizes,
            dataptr: 0,
            idx: 0,
        }
    }
}

pub fn chunked_mode(string: Vec<u8>, sizes: Vec<u8>, target: &CString) -> Option<String> {
    let threads = { ARGS.lock().unwrap().num_threads };
    let gpuon = { ARGS.lock().unwrap().use_gpu };
    if gpuon {
        md5_transform(&string, &sizes, threads as u32)
            .iter()
            .par_bridge()
            .find_any(|x| **x == target.to_string_lossy().to_string())
            .cloned()
    } else {
        let mut itr = SplittedIterator::new(&string, &sizes);
        itr.par_bridge()
            .map(|x| {
                if md5_cpu(&x) == target.to_string_lossy().to_string() {
                    return Some(x.to_string_lossy().to_string());
                }
                return None;
            })
            .find_any(|x| x.is_some())?
    }
}

pub fn brute_mode(start: usize, end: usize, target: &CString) -> Option<String> {
    let threads = { ARGS.lock().unwrap().num_threads };
    let brutestart = { ARGS.lock().unwrap().brutestart };
    let gpuon = { ARGS.lock().unwrap().use_gpu };
    if gpuon {
        return md5_brute(start, end, target, threads as u32, brutestart as u32);
    } else {
        let mut generator = SequenceGenerator::new(brutestart as u8);
        generator.skip_to(start);
        let data = generator.generate_flatten_chunk(end - start);
        let itr = SplittedIterator::new(&data.strings, &data.sizes);
        itr.par_bridge()
            .map(|x| {
                if md5_cpu(&x) == target.to_string_lossy().to_string() {
                    return Some(x.to_string_lossy().to_string());
                }
                return None;
            })
            .find_first(|x| x.is_some())?
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        dictionary_reader::DictionaryReader,
        sequence_generator::{ChunkGenerator, GeneratorResult, SequenceGenerator},
    };

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
    fn test_splitted_iterator_limits() {
        let mut generator = SequenceGenerator::new(4);
        let result = generator.generate_flatten_chunk(1000);
        let mut itr = SplittedIterator::new(&result.strings, &result.sizes).par_bridge();
        let target = md5_cpu(&CString::new("!!!!").unwrap());
        let res = itr.map(|x| md5_cpu(&x)).find_any(|x| *x == target);
        assert!(res.is_some());
    }

    #[test]
    fn test_iterator_on_dictionary() {
        let mut reader = DictionaryReader::new(
            "/home/drfaust/Scrivania/uni/Magistrale/SCPD/Project/DistributedCrack/dictionary.txt",
        )
        .expect("Failed to read file");
        let result = reader.generate_flatten_chunk(1000);
        let mut itr = SplittedIterator::new(&result.strings, &result.sizes);
        let target = md5_cpu(&CString::new("#name?").unwrap());
        let res = itr
            .par_bridge()
            .map(|x| md5_cpu(&x))
            .find_any(|x| *x == target);
        assert!(res.is_some());
    }
}
