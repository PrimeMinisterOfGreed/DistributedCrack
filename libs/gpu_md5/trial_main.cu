#include "cstdio"
#include "src/cuda_memory_support.cuh"


__global__ void trial(gpumemblock& blk){
    printf("SIze %ld",blk.get_size());
}

int main(){
    printf("print");
    auto block = gpumemblock{1024};
    trial<<<2,1>>>(block);
    cudaDeviceSynchronize();
}