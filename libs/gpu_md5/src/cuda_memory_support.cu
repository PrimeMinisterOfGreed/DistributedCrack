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

gpumemblock::gpumemblock(size_t size) : _size(size) {
  GpuMalloc(&_data, size);
  GpuMalloc(&_gpusize, 1);
  GpuCopy(_gpusize, &_size, 1, cudaMemcpyHostToDevice);
}

__device__ uint8_t *gpumemblock::getblock() { return static_cast<uint8_t *>(_data); }

__device__ size_t gpumemblock::get_size() { return *_gpusize; }

__host__ void gpumemblock::copyfrom(void *data) {
  GpuCopy(_data, data, _size, cudaMemcpyHostToDevice);
}
__host__ void gpumemblock::copyto(void *data) {
  GpuCopy(data, _data, _size, cudaMemcpyDeviceToHost);
}

__device__ void gpumemblock::copyto(gpumemblock &blk) {
  GpuCopy(blk._data, _data, _size, cudaMemcpyDeviceToDevice);
}
