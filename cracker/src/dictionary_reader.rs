use std::fs::File;
use std::io::{self, BufRead, BufReader, Read};

use crate::sequence_generator::{ChunkGenerator, GeneratorResult};

pub struct DictionaryReader {
    buffer: BufReader<File>,
}

impl DictionaryReader {
    pub fn new(file_path: &str) -> io::Result<Self> {
        let file = File::open(file_path)?;
        let reader = BufReader::new(file);
        Ok(Self { buffer: reader })
    }
}

impl ChunkGenerator for DictionaryReader {
    fn generate_flatten_chunk(&mut self, chunksize: usize) -> GeneratorResult {
        let mut strings: Vec<u8> = Vec::new();
        let mut sizes: Vec<u8> = Vec::new();
        let mut lines_used = 0;
        loop {
            if lines_used >= chunksize {
                break;
            }

            let mut line = String::new();
            if self.buffer.read_line(&mut line).is_ok() {
                if line.is_empty() {
                    break;
                }
                strings.extend(line.trim().as_bytes());
                sizes.push(line.len() as u8);
            } else {
                break;
            }
            lines_used += 1;
        }
        GeneratorResult { strings, sizes }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dictionary_reader() {
        let mut reader = DictionaryReader::new(
            "/home/drfaust/Scrivania/uni/Magistrale/SCPD/Project/DistributedCrack/dictionary.txt",
        )
        .expect("Failed to read file");
        let result = reader.generate_flatten_chunk(1000);
        assert_eq!(result.sizes.len(), 1000);
    }
}
