#pragma once
#include "string_generator.hpp"
#include <string>
#include <fstream>
struct LoaderState{
    size_t actualseq = 0;
    char filename[128] = {};
};

struct ChunkLoader{
    private:
    LoaderState _state{};
    std::fstream _state_store{};
    std::ifstream _dictionary{};
    AssignedSequenceGenerator _generator{1};
    
    public:
    ChunkLoader();
    std::vector<std::string> get_chunk(int dim);
};