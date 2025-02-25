#include "cudagpu.hpp"
#ifdef USE_CUDA
#include "cuda.h"
#include <cuda_runtime_api.h>

struct CudaDevice{
    bool busy = false;
    int index = -1;
    cudaDeviceProp props;
    CudaDevice(int index):index(index){
        cudaGetDeviceProperties(&props,index);
    }

    void select(){
        int dev = -1;
        cudaGetDevice(&dev);
        if(dev != index){
            cudaSetDevice(index);
        }
    }
};

std::vector<CudaDevice> devices;


CudaResource::CudaResource()
{
    int count = 0;
    cudaGetDeviceCount(&count);
    for(int i = 0 ; i < count; i++){
        devices.push_back(CudaDevice(i));
    }    
}

size_t CudaResource::compute(ComputeContext &context) {
    
}


CudaResource cuda_resource;

#endif
