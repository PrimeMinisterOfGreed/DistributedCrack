#include "resource_scheduler.hpp"
ResourceManager& ResourceManager::instance() {
    static ResourceManager manager{};
    return manager;
}

inline void ResourceManager::add_resource(Resource* resource) {
    this->resources.push_back(resource);
}


void Resource::operator()(ComputeContext ctx) {
        busy = true;
        task_completed += compute(ctx);
        busy = false;
}

Resource::Resource()
{
    ResourceManager::instance().add_resource(this);
}
