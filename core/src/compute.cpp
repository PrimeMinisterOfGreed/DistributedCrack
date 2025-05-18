#include "compute.hpp"
#include "log.hpp"
#include "options.hpp"
#include "utils.hpp"
#include <cstdint>
#include <md5.h>
#include <optional>
#include <sched.h>
#include <thread>
#ifdef CUDA_GPU
#include <md5_gpu.h>
constexpr bool gpu_available = true;
#else 
constexpr bool gpu_available = false;
#endif

#ifdef CUDA_GPU
#include <chrono>
#include <nvml.h>

struct GpuDescriptor {
    int device_id;
    nvmlDevice_t device;
    nvmlPciInfo_t pci_info;
    nvmlUtilization_t utilization;
    nvmlMemory_t memory;
};
struct GpuManager{
    private:
        std::vector<GpuDescriptor> gpu_descriptors;
        int num_gpus;
    public:
    
};

std::optional<std::string> compute_brute_gpu(ComputeContext::BruteContext ctx) {
    trace("compute_brute start %lu end %lu, threads : %d", ctx.start, ctx.end, ARGS.num_threads);
    auto res = md5_bruter(ctx.start, ctx.end, ctx.target,ARGS.num_threads, ARGS.brute_start);
    if (res.found) {
        return {res.data};
    }
    return {};
}

std::optional<std::string> compute_chunk_gpu(ComputeContext::ChunkContext ctx) {
    auto res = md5_gpu(reinterpret_cast<char*>(ctx.data), ctx.sizes, ctx.chunk_size, ARGS.num_threads);
    uint64_t offsets[ctx.chunk_size];
    for(int i = 1; i < ctx.chunk_size; i++) {
        offsets[i] = ctx.sizes[i-1] + offsets[i-1];
    }
    std::optional<std::string> result = std::nullopt;
    #pragma omp parallel for
    for(int i = 0; i < ctx.chunk_size; i++) {

        if(strncmp(&res.data[i*32],ctx.target, 32) == 0){
            #pragma omp critical
            {
                result = std::string(reinterpret_cast<const char*>(&ctx.data[offsets[i]]), ctx.sizes[i]);
            }
        }
    }
    return result;
}
#else 
std::optional<std::string> compute_chunk_gpu(ComputeContext::ChunkContext ctx) {
    errno = ENOTSUP; // Function not supported
    perror("GPU computation not supported");
    return std::nullopt;
}
std::optional<std::string> compute_brute_gpu(ComputeContext::BruteContext ctx) {
    errno = ENOTSUP; // Function not supported
    perror("GPU computation not supported");
    return std::nullopt;
}
#endif  


std::optional<std::string> compute_chunk(ComputeContext::ChunkContext ctx) {
    if constexpr (gpu_available)  {
        if(ARGS.use_gpu) {
            return compute_chunk_gpu(ctx);
        }
    }
    else{
        //TODO switch to pure pthread implementation
    }
    return {};
}

std::optional<std::string> compute_brute_cpu(ComputeContext::BruteContext ctx){
        std::optional<std::string> result = std::nullopt;
        trace("compute_brute start %lu end %lu, threads : %d on cpu", ctx.start, ctx.end, ARGS.num_threads);
        #pragma omp parallel
        for(int i = ctx.start; i < ctx.end; i++) {
            auto generator = SequenceGenerator{(uint8_t)ARGS.brute_start};
            generator.skip_to(i);
            auto seq = generator.current();
            uint8_t digest[16]{};
            md5String( const_cast<char*>(seq.c_str()), digest);
            char hex[33]{};
            md5HexDigest(digest, hex);
            if (strncmp(hex, ctx.target, 32) == 0) {
                #pragma omp critical
                {
                    result = seq;
                }   
            }
        }
        return result;
}


std::optional<std::string> compute_brute(ComputeContext::BruteContext ctx) {
    if constexpr (gpu_available) {
        if(ARGS.use_gpu) {
            return compute_brute_gpu(ctx);
        }
        else{
            return compute_brute_cpu(ctx);
        }
    }
    return compute_brute_cpu(ctx);
}

std::optional<std::string> compute(ComputeContext ctx) {
       switch (ctx.type) {
        case 1: // ChunkContext
            return compute_chunk(ctx.chunk_ctx);
        case 0: // BruteContext
            return compute_brute(ctx.brute_ctx);
        default:
            errno = EINVAL; // Invalid argument
            perror("Invalid context type");
            return std::nullopt; // Invalid context type
       
       }
}


