#include "md5bruter.cuh"
#include "md5gpu.cuh"
#include "gpu_string_generator.cuh"


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
    target_found[i] = 0xa;
    if(i < span){
        GpuStringGenerator gen{base_str_len};
        gen.assign_address(address_start);
        char * sequence = new char[gen._currentSequenceLength]{};
        gen.next_sequence(sequence);
        MD5Gpu algo{sequence,gen._currentSequenceLength};
        const uint8_t* digest = algo.getdigest();
        char result[33]{};
        hexdigest(digest,result);
        printf("result %s",result);
        if(cmpstr(result, target_md5, 33)){
            memcpy(target_found,sequence,gen._currentSequenceLength);
        }
        free(sequence);
    }
}

__host__ void CheckGpuCondition();

void md5_gpu_brute(const char target_md5[33], size_t address_start,size_t address_end, int base_str_len, char target_found[64], int threads){
    char *_devResultPtr = nullptr;
    CheckGpuCondition();
    if(cudaError_t error = cudaMalloc(&_devResultPtr,64);error){
        printf("error %s\n",cudaGetErrorString(error));
        return;
    }
    int span = address_end - address_start;
    int blocks = ceil(static_cast<double>(span)/threads); 
    md5_brute_apply<<<threads,blocks>>>(target_md5, address_start, address_end, base_str_len, _devResultPtr);
    cudaDeviceSynchronize();
   auto err = cudaGetLastError();
     if(err != cudaSuccess){
    //throw CudaMemoryError(err);
  }
    cudaMemcpy(target_found, _devResultPtr, 64, cudaMemcpyDeviceToHost);
    
}


