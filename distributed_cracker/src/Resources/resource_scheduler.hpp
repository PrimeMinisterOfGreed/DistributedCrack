#pragma once
#include "MultiThread/autoresetevent.hpp"
#include "clock_register.hpp"
#include <string>
#include <vector>

enum ContextTypes{
    StringVector = 0,
    PairSize = 1
};

struct ComputeContext{
    void* data;
    int type = 0;
    std::optional<std::string>result;
};

struct ResourceStats{
    std::string dev_name;
    size_t task_completed = 0;
    size_t observation = 0;
    size_t busy_time = 0; 

    double throughput(){return (double)task_completed/ClockRegister::clock_since_start();}
    double utilization(){return (double)busy_time/ClockRegister::clock_since_start();}
    double mean_service_time(){return (double)busy_time/task_completed;}
    
};

enum ResourceType{
    CPU,
    GPU,
    COPROCESSOR,
    SIZE
};

struct Resource{
    void operator()(ComputeContext ctx, void(*on_completion)(void* env) = nullptr, void*env = nullptr);
    virtual ResourceType type() = 0;
    virtual void compute(ComputeContext& context) = 0;
    virtual bool available() = 0;
    virtual std::vector<ResourceStats> get_stats() = 0;
    Resource();
};

struct SchedulerStats{
    int num_of_task_scheduled[ResourceType::SIZE]{};
    int num_of_cpu_in_use = 0;
    int num_of_gpu_in_use = 0;
};

struct ResourceManager{
    SchedulerStats stats{};
    void(*on_context_result)(ComputeContext& ctx) = nullptr;
    static ResourceManager& instance();
    void add_resource(Resource* resource);
    void schedule_task(ComputeContext ctx);
     
    private:
    ResourceManager();
};


