#pragma once
#include "result.hpp"
#include <variant>
#include <stdbool.h>
#include <stdint.h>
struct options{
    char* config_file;
    bool use_gpu;
    bool use_mpi;
    char* target_md5;
    uint32_t num_threads;
    uint32_t chunk_size;
    int verbosity;
    char* save_file;
    char* dictionary_file;
    int brute_start;
    int cluster_degree;
    bool enable_watcher;
    static Result<options, const char*> load_from_file(const char* filename);
    bool brute_mode() const;
};

extern options ARGS;