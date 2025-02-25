#pragma once
#include "resource_scheduler.hpp"
#ifdef USE_CUDA


struct CudaResource : Resource{
    
    CudaResource();
    size_t compute(ComputeContext &context) override;
};


#endif