#include "cuda_manager.hpp"
#include <algorithm>
#include <cstdio>
#include <vector>
#include "cuda.h"
#include "cuda_runtime.h"
struct GpuDescriptor{
    size_t assigned_tasks = 0;
    int device_num = 0;
    bool disabled = false;
    cudaDeviceProp props;
    void select(){
        cudaSetDevice(device_num);
        cudaGetDeviceProperties_v2(&props, device_num);
        assigned_tasks++;
    }
    GpuDescriptor(int dev_num) : device_num(dev_num){
        cudaInitDevice(dev_num, 0, 0);
    }

    void disable() {disabled = true;}
};

std::vector<GpuDescriptor> descriptors;

void CudaManager::init() {
    cuInit(0);
    int numgpu = 0;
    cudaGetDeviceCount(&numgpu);
    printf("there are %d gpus in the system \n",numgpu);
    for(int i = 0; i < numgpu; i++){
        descriptors.push_back(GpuDescriptor{i});
    }    
}
void CudaManager::select_gpu() {
    if(descriptors.size() == 0){
        perror("All GPUs are unusable quitting \n");
        exit(1);
    }
    GpuDescriptor&desc = descriptors[0];
    for(auto&d : descriptors){
        if(d.assigned_tasks < desc.assigned_tasks) desc = d;
    }
    printf("Select device %d\n",desc.device_num);
    desc.select();
}

void CudaManager::disable_current_gpu() {
    int dev= 0;
    cudaGetDevice(&dev);
    printf("Disabled gpu %d due to error\n",dev);
    auto itr = descriptors.begin() + dev;
    printf("Disabling %d \n",itr->device_num);
    printf("Size before %d\n",descriptors.size());
    descriptors.erase(itr);
    printf("Size after %ld",descriptors.size());
}

CudaManager::CudaManager()
{
    
}


CudaManager* CudaManager::instance(){
    static CudaManager ins{};
    static bool init = false;
    if(!init){
        ins.init();
        init = true;
    }
    return &ins;
}