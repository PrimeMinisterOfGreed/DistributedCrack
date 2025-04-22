use std::io::{Read, Write};

use serde::{Deserialize, Serialize};

use crate::options::{ARGS, ProgramOptions};

#[derive(Debug, Serialize, Deserialize, Default, Clone)]
pub struct State {
    pub last_address: usize,
    pub last_dictionary: String,
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
        self.last_state.last_dictionary = dictionary;
    }

    pub fn load_from_file(filepath: &str) -> Option<Self> {
        let file = std::fs::File::open(filepath).ok()?;
        let mut reader = std::io::BufReader::new(file);
        let mut buffer = String::new();
        let _ = reader.read_to_string(&mut buffer).ok()?;
        toml::de::from_str(&buffer).ok()
    }

    pub fn save_to_file(&self, filepath: &str) -> std::io::Result<()> {
        let file = std::fs::File::create(filepath)?;
        let mut writer = std::io::BufWriter::new(file);
        let toml_string = toml::ser::to_string(self).unwrap();
        writer.write_all(toml_string.as_bytes())?;
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
        let options = ProgramOptions::parse_from(vec!["--savefile storage.toml"]);
        let mut state = StateFile::new();
        let filepath = "storage.toml";
        state.update_address(1000);
        state.save_to_file(filepath).unwrap();
        let loaded_state = StateFile::load_from_file(filepath).unwrap();
        assert_eq!(state.last_state.last_address, 1000);
        //std::fs::remove_file(filepath).unwrap();
    }
}
