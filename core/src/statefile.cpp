#include "statefile.hpp"
#include "log.hpp"
#include "options.hpp"
#include <cstdlib>
#include <fstream>
#include <memory>
std::unique_ptr<StateFile> StateFile::load(const char *filename) {
  auto file = std::ifstream{filename,  std::ios::in};
  if (!file) {
    exception("Failed to open file: %s", filename);
    abort();
  }
  char buffer[sizeof(StateFile)]{};
  file.read(buffer, sizeof(StateFile));
  if (!file) {
    exception("Failed to read file: %s", filename);
    abort();
  }
    auto state = std::make_unique<StateFile>();
    std::memcpy(state.get(), buffer, sizeof(StateFile));
    return state;
}


void StateFile::save(const char *filename) const {
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