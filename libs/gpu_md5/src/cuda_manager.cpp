#include "cuda_manager.hpp"
#include <algorithm>
#include <cstdio>
#include <vector>
#include "cuda.h"
#include "cuda_runtime.h"
#include "nvml.h"

constexpr float cross_edge = 0;

struct GpuDescriptor{
    size_t assigned_tasks = 0;
    unsigned long long stress_factor = 0;
    int device_num = 0;
    bool disabled = false;
    cudaDeviceProp props;
    void select(){
        cudaSetDevice(device_num);
        assigned_tasks++;
    }
    GpuDescriptor(int dev_num) : device_num(dev_num){
        cudaInitDevice(dev_num, 0, 0);
        cudaGetDeviceProperties_v2(&props, device_num);
        
    }

    void disable() {disabled = true;}

    void reset(){
        disabled = false;
        select();
        cudaDeviceReset();
    }

    double stress_index(){

        nvmlUtilization_st util{};
        nvmlDevice_t device{};
        nvmlDeviceGetHandleByIndex_v2(device_num, &device);
        auto res = nvmlDeviceGetUtilizationRates(device, &util);
        stress_factor += util.gpu;
        return (double)stress_factor/assigned_tasks+1;
    }
};

const int maxrebirths = 3;

std::vector<GpuDescriptor> descriptors;

void CudaManager::init() {
    cuInit(0);
    nvmlInit();
    int numgpu = 0;
    cudaGetDeviceCount(&numgpu);
    for(int i = 0; i < numgpu; i++){
        descriptors.push_back(GpuDescriptor{i});
    }    

}



void CudaManager::select_gpu() {
    if(descriptors.size() == 0){
        perror("All GPUs are unusable quitting \n");
        if(rebirths >= maxrebirths)
            exit(1);
        else{
            rebirths++;
            for(auto&d : descriptors) d.reset();
        }
    }

    GpuDescriptor* desc = &descriptors[0];
    for(auto&d : descriptors){
        if((d.stress_index() - desc->stress_index()) > cross_edge|| (desc->disabled && !d.disabled)) desc = &d;
    }
    if(desc->disabled){
        perror("Computing on disabled GPU\n");
        exit(1);
    }
    desc->select();
}

void CudaManager::disable_current_gpu() {
    int dev= 0;
    cudaGetDevice(&dev);
    for(auto&d :descriptors){
        if(d.device_num == dev) {
            d.disabled = true;
            break;
        }
    } 
}

void CudaManager::force_gpu(int gpu) {
    descriptors[gpu].select();
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