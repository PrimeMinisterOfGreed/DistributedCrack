#include "statefile.hpp"
#include "log.hpp"
#include "options.hpp"
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
std::optional<std::unique_ptr<StateFile>> StateFile::load(const char *filename) {

  auto file = std::ifstream{filename,  std::ios::in};
  if (!file) {
    debug("Failed to open file: %s", filename);
    return {};
  }
  char buffer[sizeof(StateFile)]{};
  file.read(buffer, sizeof(StateFile));
  if (!file) {
    debug("Failed to read file: %s", filename);
    return {};
  }
    auto state = std::make_unique<StateFile>();
    std::memcpy(state.get(), buffer, sizeof(StateFile));
    return state;
}

std::unique_ptr<StateFile> StateFile::create(const char* filename) {
    if(std::filesystem::exists(filename)) {
        debug("State file already exists: %s", filename);
        auto state = load(filename);
    }
    debug("Creating new state file: %s", filename);
    StateFile state{};
    state.current_address = 0;
    state.current_dictionary[0] = '\0'; // Initialize with empty string
    memcpy(state.filename, filename, std::min(sizeof(state.filename), strlen(filename)));
    return std::make_unique<StateFile>(state);
}


void StateFile::save() const {
  auto file = std::ofstream{filename, std::ios::out};
  if (!file) {
    exception("Failed to open file: %s", filename);
    abort();
  }
  file.write(reinterpret_cast<const char *>(this), sizeof(StateFile));
  if (!file) {
    exception("Failed to write file: %s", filename);
    abort();
  }
}