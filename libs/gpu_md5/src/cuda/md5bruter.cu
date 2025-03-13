#include "md5bruter.cuh"
#include "md5gpu.cuh"
#include "gpu_string_generator.cuh"
#include "cuda_manager.hpp"

inline __device__ void hexdigest(const uint8_t digest[16], char hex_output[33]) {
    static const char hex_chars[] = "0123456789abcdef";

    for (int i = 0; i < 16; i++) {
        hex_output[i * 2] = hex_chars[(digest[i] >> 4) & 0xF];
        hex_output[i * 2 + 1] = hex_chars[digest[i] & 0xF];
    }
    hex_output[32] = '\0'; // Null-terminate la stringa
}

__device__ bool cmpstr(const char* a, const char* b, size_t size){
    for(size_t i = 0 ; i < size; i++){
        if(a[i]!=b[i]) return false;
    }
    return true;
}


__global__ void md5_brute_apply(const char target_md5[33],size_t address_start,size_t address_end, int base_str_len, char * target_found){
    int i = threadIdx.x+blockDim.x*blockIdx.x;
    size_t span = address_end - address_start;
    if(i < span){
        GpuStringGenerator gen{base_str_len};
        gen.assign_address(address_start + i);
        char * sequence = new char[gen._currentSequenceLength]{};
        gen.next_sequence(sequence);
        MD5Gpu algo{sequence,gen._currentSequenceLength};
        const uint8_t* digest = algo.getdigest();
        char result[33]{};
        hexdigest(digest,result);
        if(cmpstr(result, target_md5, 32)){
            memcpy(target_found,sequence,gen._currentSequenceLength);
            target_found[gen._currentSequenceLength] = 0;
        }
        free(sequence);
    }
}


__host__ void CheckGpuCondition();

char *_dev_res = nullptr;
char * _dev_target = nullptr;
bool _inited = false;

void bruter_initialize(){
    if(cudaError_t error = cudaMalloc(&_dev_res,64);error){
        printf("error %s\n",cudaGetErrorString(error));
        //return;
    }
    cudaMalloc(&_dev_target,33);
}

void bruter_destroy(){
    cudaFree(_dev_res);
    cudaFree(_dev_target);
}

void md5_gpu_brute(const char target_md5[33], size_t address_start,size_t address_end, int base_str_len, char target_found[64], int threads){

    CudaManager::instance()->select_gpu();
    bruter_initialize();
    if (cudaError_t error = cudaMemcpy(_dev_target, target_md5, 32, cudaMemcpyHostToDevice);error) {
        printf("error %s\n", cudaGetErrorString(error));
        exit(1);
        // return;
      }

    int span = address_end - address_start;
    int blocks = ceil(static_cast<double>(span) / threads);
    cudaDeviceSynchronize();

    md5_brute_apply<<<threads, blocks>>>(_dev_target, address_start,
                                         address_end, base_str_len, _dev_res);
    cudaDeviceSynchronize();
    if(auto error = cudaGetLastError();error){
        printf("error on computing %s \n",cudaGetErrorString(error));
        CudaManager::instance()->disable_current_gpu();
        md5_gpu_brute(target_md5, address_start, address_end, base_str_len, target_found, threads);
        return;
    }
    memset(target_found, 0, 64);
    cudaMemcpy(target_found, _dev_res, 64, cudaMemcpyDeviceToHost);
    bruter_destroy();
}
