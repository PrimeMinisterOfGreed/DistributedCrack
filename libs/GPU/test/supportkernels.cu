#include <cuda.h>

template <typename T>
__global__ void diff(T *a, T *b, T *c, size_t size)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i < size){
        c[i] = a[i] - b[i];
    }
}


template __global__ void diff<uint8_t>(uint8_t *a, uint8_t *b, uint8_t *c, size_t size);

