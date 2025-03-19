#include "md5gpu.cuh"
#include "gpu_string_generator.cuh"
#include "../cuda_manager.hpp"
#include "md5bruter.cuh"

inline __device__ void hexdigest(const uint8_t digest[16], char hex_output[33]) {
    static const char hex_chars[] = "0123456789abcdef";

    for (int i = 0; i < 16; i++) {
        hex_output[i * 2] = hex_chars[(digest[i] >> 4) & 0xF];
        hex_output[i * 2 + 1] = hex_chars[digest[i] & 0xF];
    }
    hex_output[32] = '\0'; // Null-terminate la stringa
}

#define dbgline() printf("process %d line reached %d\n",i,__LINE__);
__device__ bool cmpstr(const char* a, const char* b, size_t size){
    for(size_t i = 0 ; i < size/8; i++){
        if(((uint64_t*)a)[i] != ((uint64_t*)b)[i]){
            return false;
        }
    }
    return true;
}

#define print_request(request)  printf("request: %s %s %d %ld %ld\n",request->target_md5,request->target_found,request->base_str_len,request->address_start,request->address_end);


__global__ void md5_brute_apply(struct md5_bruter_request * request){
    int i = threadIdx.x+blockDim.x*blockIdx.x;
    size_t span = request->address_end - request->address_start;
    if(i < span){
        char sequence[24];
        memset(sequence,0,24);
        GpuStringGenerator gen = new_generator(request->base_str_len);
        assign_address(&gen,request->address_start + i);
        next_sequence(&gen,sequence);
        MD5Gpu algo{sequence,(size_t)gen.currentSequenceLength};
        const uint8_t* digest = algo.getdigest();
        char result[33]{};
        hexdigest(digest,result);
        if(cmpstr(result, request->target_md5, 32)){
            memcpy(request->target_found,sequence,gen.currentSequenceLength);
            request->target_found[gen.currentSequenceLength] = 0;
        }
        free(sequence);
    }
}


__host__ void CheckGpuCondition();

struct md5_bruter_request * dev_request = nullptr;

__host__ cudaError_t alloc_request(){
    return cudaMalloc(&dev_request, sizeof(md5_bruter_request));
}

__host__ cudaError_t free_request(){
    return cudaFree(dev_request);
}

__host__ cudaError_t copy_request_to_device(struct md5_bruter_request* request){
    return cudaMemcpy(dev_request, request, sizeof(md5_bruter_request), cudaMemcpyHostToDevice);
}

__host__ cudaError_t copy_request_to_host(struct md5_bruter_request * request){
    return cudaMemcpy(request, dev_request, sizeof(md5_bruter_request), cudaMemcpyDeviceToHost);
}

#define handle(op) error = op; if(error) goto ERROR;


void md5_gpu_brute(struct md5_bruter_request* request, int threads){
    cudaError_t error = cudaSuccess;
    CudaManager::instance()->select_gpu();
    int span = request->address_end - request->address_start;
    int blocks = ceil(static_cast<double>(span) / threads);
    handle(alloc_request());
    handle(copy_request_to_device(request));
    cudaDeviceSynchronize();

    md5_brute_apply<<<threads, blocks>>>(dev_request);
    cudaDeviceSynchronize();
   // handle(cudaGetLastError());
    handle(copy_request_to_host(request));
    handle(free_request());
    return;
    ERROR:
        printf("error on computing %s \n",cudaGetErrorString(error));
        CudaManager::instance()->disable_current_gpu();
        md5_gpu_brute(request, threads);
        return;
}


