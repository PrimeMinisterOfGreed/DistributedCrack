#include "stdio.h"



__global__ void threadcall(){
    printf("BD:%d  BI:%d  TI:%d\n",blockDim.x, blockIdx.x , threadIdx.x);
}


int main(){
    threadcall<<<100,10>>>();
    cudaDeviceSynchronize();
}