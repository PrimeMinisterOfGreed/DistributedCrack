#pragma once
#include <exception>
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_types.h>


struct CudaMemoryError : std::exception{
    cudaError_t _error;
    CudaMemoryError(cudaError_t error): _error(error){}
    const char * what() const noexcept override{
        return cudaGetErrorString(_error);
    }
};





template <typename T> void GpuMalloc(T **pointer, size_t size)
{
    cudaError_t error;
    if(( error = cudaMalloc(pointer, sizeof(T) * size)) != cudaSuccess){
        throw CudaMemoryError(error);
    }
}

template <typename T>  void GpuCopy(T *dst, const T *src, size_t size, cudaMemcpyKind kind)
{
    cudaError_t error;
    if((error = cudaMemcpy(dst, src, sizeof(T) * size, kind)) != cudaSuccess){
        throw  CudaMemoryError(error);
    }
}

template <typename T> void GpuManagedMalloc(T **pointer, size_t size)
{
    cudaError_t error;
    if((error = cudaMallocManaged(pointer, sizeof(T) * size)) != cudaSuccess)
        throw CudaMemoryError(error);
}

template<typename T> void GpuFree(T* pointer){
    cudaError_t error;
    if((error = cudaFree(pointer)) != cudaSuccess)
        throw CudaMemoryError(error);
}

template<typename... Args> void GpuFree(Args...ptrs){
    (GpuFree(ptrs),...);
}

struct gpumemblock{
    private:
    uint8_t * _data = nullptr;
    size_t * _gpusize = nullptr;
    size_t _size = 0;
    public:
    gpumemblock(size_t size): _size(size){
        GpuMalloc(&_data, size);
        GpuMalloc(&_gpusize, sizeof(size_t));
        GpuCopy(_gpusize, &_size, sizeof(size_t), cudaMemcpyHostToDevice);
    }

    gpumemblock(gpumemblock&) = delete;
    __device__ uint8_t* getblock(){
        return _data;
    }

    __device__ size_t get_size(){
        return *_gpusize;
    }

    __host__ void copyfrom(void* data){
        GpuCopy((uint8_t*)_data, (uint8_t*)data, _size, cudaMemcpyHostToDevice);
    }

    __host__ void copyto(void*data){
        GpuCopy((uint8_t*)data, (uint8_t*)_data, _size, cudaMemcpyDeviceToHost);
    }

    __device__ void copyto(gpumemblock& blk){
        GpuCopy(blk._data, _data, _size, cudaMemcpyDeviceToDevice);
    }

    ~gpumemblock(){
        //GpuFree(_gpusize,_data);
    }
};