#pragma once 
#include <cstdint>
#include <optional>
#include <string>

union ComputeContext{
    uint8_t type;

    struct BruteContext{
        uint64_t start;
        uint64_t end;
        const char * target;
    }brute_ctx;

    struct ChunkContext{
        uint8_t* data;
        uint8_t* sizes;
        const char * target;
        uint64_t chunk_size;
    }chunk_ctx;

    ComputeContext(ChunkContext ctx){
        chunk_ctx = ctx;
        type = 1;
    }
    ComputeContext(BruteContext ctx){
        brute_ctx = ctx;
        type = 2;
    }
};

std::optional<std::string> compute(ComputeContext ctx);