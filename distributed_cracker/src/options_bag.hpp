#pragma once
#include <string>

struct ProgramOptions {
  std::string config_file;
  bool use_gpu;
  std::string target_md5;
  int num_threads;
  int chunk_size;
  int verbosity;
  std::string savefile;
  bool ismpi;
  bool restore_from_file;
  bool use_mpi;
  std::string dictionary;
  constexpr bool use_dictionary() { return dictionary != "NONE"; }
  static ProgramOptions* instance(){
    static ProgramOptions _instance{};
    return &_instance;
  }
};

