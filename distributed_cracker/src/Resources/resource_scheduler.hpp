#pragma once
#include <vector>

struct ComputeContext{
    void* data;
    int type = 0;
};


struct Resource{
    bool busy = false;
    int task_assigned = 0;
    int tasks_pending = 0;
    size_t busy_time = 0;
    size_t observation = 0;
    size_t task_completed = 0;
    void operator()(ComputeContext ctx);

    virtual size_t compute(ComputeContext& context) = 0;
    Resource();
};

struct ResourceManager{
    std::vector<Resource*> resources;
    static ResourceManager& instance();
    void add_resource(Resource* resource);
};

