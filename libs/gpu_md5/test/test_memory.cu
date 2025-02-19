#include "../src/cuda_memory_support.cuh"
#include "gtest/gtest.h"


__global__ void addone(int* data, size_t size){
    auto i = blockDim.x*blockIdx.x+threadIdx.x;
    if(i<size){
        data[i] += 1;
    }
}


TEST(TestMemorySupport, test_object_alloc){
    int* ptr;
    int hostptr[30]{};
    cuInit(0);
    cudaMallocManaged(&ptr,30);
    addone<<<1,1>>>(ptr,30);
    cudaMemcpy(hostptr, ptr, 30, cudaMemcpyDeviceToHost);
    printf("Data is %d",ptr[0]);
}


__global__ void reduce(uint8_t ** a, uint8_t *b  ,size_t size){
    auto i = blockDim.x*blockIdx.x+ threadIdx.x;
    if(i < size){
        auto vec = a[i];
        auto num = 0;
        for(int k = 0 ; k < size; k++){
            num += vec[k];
        }
        b[i] = num;
    }
}

__global__ void gen_array(uint8_t ** a, size_t size){
    auto i = blockDim.x* blockIdx.x + threadIdx.x;
    if (i < size){
        auto array = reinterpret_cast<uint8_t*>(malloc(size));
        memset(array,1,size);
        a[i] = array;
    }
}


TEST(TestMemorySupport, test_malloc){
    uint8_t** dev_array = nullptr;
    uint8_t * dev_result = nullptr;
    cudaMallocManaged(&dev_array,32);
    cudaMallocManaged(&dev_result,32);
    gen_array<<<32,1>>>(dev_array, 32);
    reduce<<<32,1>>>(dev_array, dev_result, 32);
    uint8_t result[32]{};
    cudaMemcpy(&result, dev_result, 32, cudaMemcpyDeviceToHost);
    printf("element at 1 %d",result[0]);
}

