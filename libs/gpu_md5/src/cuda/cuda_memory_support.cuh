#pragma once
#include <cstdio>
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

template <typename T> void GpuMallocManaged(T *pointer, size_t size)
{
    cudaError_t error;
    if((error = cudaMallocManaged<T>(&pointer, size)) != cudaSuccess)
        throw CudaMemoryError(error);
}

static void GpuFree(void* pointer){
    cudaError_t error;
    if((error = cudaFree(pointer)) != cudaSuccess)
        throw CudaMemoryError(error);
}



template<typename T>
struct gpu_object{
    private:
    T* _ptr;
    size_t size;
    public:
    gpu_object(size_t elems) : size(size){
        GpuMalloc(&_ptr, elems);
    }

    gpu_object(): gpu_object(1){

    }

    ~gpu_object(){
        GpuFree(_ptr);
    }

    __host__ void copytogpu(const T*data){
        GpuCopy(_ptr, data, size, cudaMemcpyHostToDevice);
    }

    __host__ void copyback(T*data){
        GpuCopy(data, _ptr, size, cudaMemcpyDeviceToHost);
    }

    __host__ T* ptr(){return _ptr;}
};


