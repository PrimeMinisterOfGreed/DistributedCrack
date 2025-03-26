use std::fs::File;
use std::io::{self, BufRead, BufReader};

pub struct DictionaryReader {
    lines: Vec<String>,
}

pub struct ReaderResult {
    pub strings: Vec<u8>,
    pub sizes: Vec<u8>,
}

impl DictionaryReader {
    pub fn new(file_path: &str) -> io::Result<Self> {
        let file = File::open(file_path)?;
        let reader = BufReader::new(file);
        let lines = reader
            .lines()
            .filter_map(|line| line.ok())
            .collect::<Vec<String>>();
        Ok(Self { lines })
    }

    pub fn generate_flatten_chunk(&self) -> ReaderResult {
        let mut strings: Vec<u8> = Vec::new();
        let mut sizes: Vec<u8> = Vec::new();

        for line in &self.lines {
            sizes.push(line.len() as u8);
            strings.extend_from_slice(line.as_bytes());
        }

        ReaderResult { strings, sizes }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dictionary_reader() {
        let reader = DictionaryReader::new(
            "/home/drfaust/Scrivania/uni/Magistrale/SCPD/Project/DistributedCrack/dictionary.txt",
        )
        .expect("Failed to read file");
        let result = reader.generate_flatten_chunk();

        assert_eq!(result.sizes.len(), reader.lines.len());
        assert_eq!(
            result.strings.len(),
            reader.lines.iter().map(|line| line.len()).sum::<usize>()
        );
    }
}
