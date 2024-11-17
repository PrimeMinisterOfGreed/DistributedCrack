#pragma once
#include <cstddef>
#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>

__host__ void md5_gpu(const uint8_t *data, const uint32_t *sizes, uint8_t *result, uint32_t size, int threads);
__host__ int md5_gpu(const uint8_t *data, const uint32_t *sizes, uint32_t size, int threads, uint8_t * targetDigest);
__host__ void CheckGpuCondition();




template <typename T> cudaError_t GpuMalloc(T **pointer, size_t size)
{
    return cudaMalloc(pointer, sizeof(T) * size);
}

template <typename T> cudaError_t GpuCopy(T *dst, const T *src, size_t size, cudaMemcpyKind kind)
{
    return cudaMemcpy(dst, src, sizeof(T) * size, kind);
}

template <typename T> cudaError_t GpuManagedMalloc(T **pointer, size_t size)
{
    return cudaMallocManaged(pointer, sizeof(T) * size);
}