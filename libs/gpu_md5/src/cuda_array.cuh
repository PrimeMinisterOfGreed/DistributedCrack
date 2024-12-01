#pragma once
#include "cuda_memory_support.cuh"

template<typename T>

struct gpuarray{
    private:
    T* _localdata;
    T* _gpudata;
    bool _changed;
    size_t _size;
    public:
    gpuarray(T* data, size_t size): _size(size), _localdata(data){
                
    }
    
    
};