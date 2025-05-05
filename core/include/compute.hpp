#pragma once 
#include <cstdint>
#include <optional>
#include <string>

union ComputeContext{

    struct BruteContext{
        uint64_t start;
        uint64_t end;
        const char * target;
    }brute_ctx;

    struct ChunkContext{
        uint8_t* data;
        uint8_t* sizes;
        uint64_t chunk_size;
        const char * target;
    }chunk_ctx;

    ComputeContext(ChunkContext ctx){
        chunk_ctx = ctx;
        type = 1;
    }
    ComputeContext(BruteContext ctx){
        brute_ctx = ctx;
        type = 0;
    }
    uint8_t type;

};

std::optional<std::string> compute(ComputeContext ctx);