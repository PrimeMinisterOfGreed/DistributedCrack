#pragma once
#include "resource_scheduler.hpp"

struct CpuResource: Resource{
    ResourceStats stats{};
    CpuResource();
    bool available() override;
    std::vector<ResourceStats> get_stats() override;
    void compute(ComputeContext &context) override;
    ResourceType type() override{return ResourceType::CPU;}
};