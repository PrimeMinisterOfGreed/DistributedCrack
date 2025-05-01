#include "compute.hpp"
#include "options.hpp"
#include <md5.h>
#include <md5_gpu.h>


std::optional<std::string> compute_brute(ComputeContext::BruteContext ctx) {
    auto res = md5_bruter(ctx.start, ctx.end, ctx.target,ARGS.num_threads, ARGS.brute_start);
    if (res.found) {
        return {res.data};
    }
    return {};
}

std::optional<std::string> compute_chunk(ComputeContext::ChunkContext ctx) {
    auto res = md5_gpu(reinterpret_cast<char*>(ctx.data), ctx.sizes, ctx.chunk_size, ARGS.num_threads);
    uint64_t offsets[ctx.chunk_size];
    for(int i = 1; i < ctx.chunk_size; i++) {
        offsets[i] = ctx.sizes[i-1] + offsets[i-1];
    }
    std::optional<std::string> result = std::nullopt;
    #pragma omp parallel for
    for(int i = 0; i < ctx.chunk_size; i++) {

        if(strncmp(&res.data[offsets[i]],ctx.target, 32) == 0){
            #pragma omp critical
            {
                result = std::string(&res.data[offsets[i]], 32);
            }
        }
    }
    return result;
}


std::optional<std::string> compute(ComputeContext ctx) {
       switch (ctx.type) {
        case 1: // ChunkContext
            return compute_chunk(ctx.chunk_ctx);
        case 2: // BruteContext
            return compute_brute(ctx.brute_ctx);
        default:
            errno = EINVAL; // Invalid argument
            perror("Invalid context type");
            return std::nullopt; // Invalid context type
       
       }
}


