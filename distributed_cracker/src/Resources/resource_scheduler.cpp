#include "resource_scheduler.hpp"
#include "options_bag.hpp"
#include <future>


struct ResourceDescriptor{
    Resource* res = nullptr;
    bool busy = false;
    ManualResetEvent waitHandle{false};
    ResourceDescriptor(Resource* res):res(res){}
};

std::vector<std::shared_ptr<ResourceDescriptor>> resources[ResourceType::SIZE]{};
    

ResourceManager& ResourceManager::instance() {
    static ResourceManager manager{};
    return manager;
}

inline void ResourceManager::add_resource(Resource* resource) {
    resources[resource->type()].push_back(std::shared_ptr<ResourceDescriptor>(new ResourceDescriptor{resource}));
}

std::optional<std::shared_ptr<ResourceDescriptor>> get_resource(){
    if(options.use_gpu){
        for(auto&gpu: resources[GPU]){
            if(!gpu->busy)return gpu;
        }
    }   
    else if (!options.gpu_only){
        for(auto& cpu: resources[CPU]){
            if(!cpu->busy) return cpu;
        }
    }
    return {};
}

std::vector<WaitHandle*> get_all_handles(){
    std::vector<WaitHandle*> handles{};
    for(auto&res: resources){
        for(auto &desc: res){
            handles.push_back(&desc->waitHandle);
        }
    }
    return handles;
}

void ResourceManager::schedule_task(ComputeContext ctx) {
    //selection stage
    auto res = get_resource();
    while(!res.has_value()){
        int index = WaitAny(get_all_handles());   
        res= get_resource();
    }
    auto resource = res.value().get();
    resource->waitHandle.Set();
    auto t = std::async([resource,ctx]{
        resource->busy = true;
        (*resource->res)(ctx,[](void*env){
            auto self = static_cast<ResourceDescriptor*>(env);
            self->busy = false;
            self->waitHandle.Reset();
        },resource);
    });

}

ResourceManager::ResourceManager()
{
}


void Resource::operator()(ComputeContext ctx, void(*on_completion)(void*env), void* env) {
  compute(ctx);
  if(on_completion != nullptr) on_completion(env);
}

Resource::Resource() { ResourceManager::instance().add_resource(this); }
