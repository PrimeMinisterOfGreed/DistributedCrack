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

static void GpuFree(void* pointer){
    cudaError_t error;
    if((error = cudaFree(pointer)) != cudaSuccess)
        throw CudaMemoryError(error);
}

template<typename... Args> void GpuFree(Args...ptrs){
    (GpuFree(ptrs),...);
}

struct gpumemblock{
    private:
    void * _data = nullptr;
    size_t * _gpusize = nullptr;
    size_t _size = 0;
    public:
    gpumemblock(size_t size);

    gpumemblock(gpumemblock&) = delete;
    __device__ uint8_t* getblock();

    __device__ size_t get_size();

    __host__ void copyfrom(void* data);

    __host__ void copyto(void*data);

    __device__ void copyto(gpumemblock &blk);

    ~gpumemblock(){
        //GpuFree(_gpusize,_data);
    }
};

