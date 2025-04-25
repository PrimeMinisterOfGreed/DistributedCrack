use std::{
    io::{Read, Write},
    mem::transmute,
};

use serde::{Deserialize, Serialize};

use crate::options::{ARGS, ProgramOptions};

#[derive(Debug, Serialize, Deserialize, Default, Clone)]
pub struct State {
    pub last_address: usize,
    pub last_dictionary: [u8; 32],
}

#[derive(Debug, Serialize, Deserialize, Default, Clone)]
pub struct StateFile {
    last_state: State,
}

impl StateFile {
    pub fn new() -> Self {
        Self {
            last_state: State::default(),
        }
    }

    pub fn update_address(&mut self, address: usize) {
        self.last_state.last_address = address;
    }
    pub fn update_dictionary(&mut self, dictionary: String) {
        self.last_state.last_dictionary[..{
            if dictionary.len() > 32 {
                32
            } else {
                dictionary.len()
            }
        }]
            .copy_from_slice(dictionary.as_bytes());
    }

    pub fn load_from_file(filepath: &str) -> Option<Self> {
        let file = std::fs::File::open(filepath).ok()?;
        let mut reader = std::io::BufReader::new(file);
        let mut buf = Vec::new();
        let bytes = reader.read_to_end(&mut buf).ok()?;
        if bytes < size_of::<StateFile>() {
            return None;
        }
        let res = unsafe { transmute(*(buf.as_mut_ptr().cast::<[u8; size_of::<StateFile>()]>())) };
        Some(res)
    }

    pub fn save_to_file(&self, filepath: &str) -> std::io::Result<()> {
        let file = std::fs::File::create(filepath)?;
        let mut writer = std::io::BufWriter::new(file);
        let bytes = unsafe { transmute::<&StateFile, &[u8; size_of::<StateFile>()]>(self) };
        writer.write_all(bytes)?;
        writer.flush()?;
        Ok(())
    }

    pub fn save_default(&self) -> std::io::Result<()> {
        let file = ARGS.lock().unwrap().savefile.clone();
        self.save_to_file(&file)
    }

    pub fn last_state(&self) -> &State {
        &self.last_state
    }
}

#[cfg(test)]
mod tests {
    use clap::Parser;

    use super::*;

    #[test]
    fn test_storage_save() {
        let options = ProgramOptions::parse_from(vec!["--savefile storage.dat"]);
        let mut state = StateFile::new();
        let filepath = "storage.dat";
        state.update_address(1000);
        state.save_to_file(filepath).unwrap();
        let loaded_state = StateFile::load_from_file(filepath).unwrap();
        assert_eq!(state.last_state.last_address, 1000);
        //std::fs::remove_file(filepath).unwrap();
    }
}
