#include "../src/cuda_memory_support.cuh"
#include "gtest/gtest.h"


__global__ void increase(gpumemblock& blk){
    auto i = blockDim.x*blockIdx.x + threadIdx.x;
   if(i == 0)
        printf("Size of block %d\n",blk.get_size());
    if(i < blk.get_size()){
        blk.getblock()[i] += 1;
    }
}


TEST(TestMemorySupport, test_gpu_functions){
    uint32_t *data = nullptr;
    uint32_t transf[32]{};
    for(int i = 0 ; i < sizeof(transf); i++){
        transf[i] = i ;
    }
    GpuMalloc<uint32_t>(&data,32);
    GpuCopy(data,transf,sizeof(transf),cudaMemcpyHostToDevice);
    GpuFree(data);
}

TEST(TestMemorySupport, test_block_instance){
    auto block = gpumemblock{1024};
    char ch[1024]{};
    char tgt[1024]{};
    for(int i = 0 ; i < 1024; i++){
        ch[i]=i;
    }
    block.copyfrom(ch);
    increase<<<1,1>>>(block);
    cudaDeviceSynchronize();
    block.copyto(tgt);
    cudaDeviceSynchronize();
    ASSERT_EQ(1, ch[0]);
}