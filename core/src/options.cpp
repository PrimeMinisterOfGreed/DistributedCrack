#include "options.hpp"
#include "toml.h"
#include <cstring>
#include <unistd.h>
#include <variant>
#include <filesystem>

Result<options, const char*> options::load_from_file(const char* filename) {    
    options opt;
    if (!std::filesystem::exists(filename)) {
        return "File does not exist";
    }
    if (!std::filesystem::is_regular_file(filename)) {
        return "Not a regular file";
    }

    FILE* file = fopen(filename, "r");
    toml_table_t* table = toml_parse_file(file, nullptr, 0);
    if (table == nullptr) {
        return "Failed to parse TOML file";
    }
    opt.config_file = strdup(filename);
    opt.use_gpu = toml_bool_in(table, "use_gpu").u.b;
    opt.use_mpi = toml_bool_in(table, "use_mpi").u.b;
    opt.target_md5 = strdup(toml_raw_in(table, "target_md5"));
    opt.num_threads = toml_int_in(table, "num_threads").u.i;
    opt.chunk_size = toml_int_in(table, "chunk_size").u.i;
    opt.verbosity = toml_int_in(table, "verbosity").u.i;
    opt.save_file = strdup(toml_raw_in(table, "savefile"));
    opt.dictionary_file = strdup(toml_raw_in(table, "dictionary"));
    opt.brute_start = toml_int_in(table, "brutestart").u.i;

    toml_free(table);
    fclose(file);
    return opt;
}

options ARGS;

