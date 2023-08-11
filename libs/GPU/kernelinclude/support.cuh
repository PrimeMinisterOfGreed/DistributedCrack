#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

template <typename T>
__global__ void diff(T *a, T *b, T *c, size_t size)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < size)
    {
        c[i] = a[i] - b[i];
    }
}
