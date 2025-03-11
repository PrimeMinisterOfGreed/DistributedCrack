#pragma once
#include "resource_scheduler.hpp"
#ifdef USE_CUDA


struct CudaResource : Resource{
    
    CudaResource();
    void compute(ComputeContext &context) override;
    bool available() override;
    std::vector<ResourceStats> get_stats() override;
    ResourceType type() override{return ResourceType::GPU;}
};


#endif