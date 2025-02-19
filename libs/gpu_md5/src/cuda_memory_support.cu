#include "cuda_memory_support.cuh"


template<> void GpuMalloc<void>(void**devPtr, size_t size){
    cudaError_t error;
    if(( error = cudaMalloc(devPtr,  size)) != cudaSuccess){
        throw CudaMemoryError(error);
    }
}

template<> void GpuCopy<void>(void*dst, const void* src, size_t size, cudaMemcpyKind kind){
     cudaError_t error;
    if((error = cudaMemcpy(dst, src, size, kind)) != cudaSuccess){
        throw  CudaMemoryError(error);
    }
}
