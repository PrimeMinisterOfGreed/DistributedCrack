#include "cudagpu.hpp"
#include "md5_gpu.hpp"
#include "options_bag.hpp"
#include <chrono>
#ifdef USE_CUDA
#include "cuda.h"
#include <cuda_runtime_api.h>

struct CudaDevice{
    bool busy = false;
    int index = -1;
    int paralleltasks = 0;
    cudaDeviceProp props;
    ResourceStats stats{};
    CudaDevice(int index):index(index){
        cudaInitDevice(index, 0, 0);
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
    cuInit(0);
    cudaGetDeviceCount(&count);
    for(int i = 0 ; i < count; i++){
        devices.push_back(CudaDevice(i));
    }    
}

void CudaResource::compute(ComputeContext &context) {

  CudaDevice &device = devices[0];
  for (auto &d : devices) {
    if (device.stats.mean_service_time() > d.stats.mean_service_time()) {
      device = d;
    }
  }
  device.select();
  size_t completed = 0;
  ClockRegister::tick(this);
  switch (context.type) {
  case StringVector: {
    auto res = md5_gpu(*static_cast<std::vector<std::string> *>(context.data),
                       options.num_threads);
    for (auto &string : res) {
      if (string == options.target_md5) {
        context.result = string;
      }
    }
    completed = res.size();
  } break;

  case PairSize: {
    auto ctx = *static_cast<std::pair<size_t, size_t> *>(context.data);
    auto res = md5_bruter(ctx.first, ctx.second, options.target_md5,
                          options.num_threads);
    if (res.has_value()) {
      context.result = res.value();
    }
    completed = ctx.second - ctx.first;
  } break;
  }
  auto time = ClockRegister::tock(this);
  device.stats.busy_time += time;
  device.stats.task_completed += 1;
  device.stats.observation = ClockRegister::clock_since_start();
}

bool CudaResource::available() { return devices.size() > 0; }

std::vector<ResourceStats> CudaResource::get_stats() {
  auto result = std::vector<ResourceStats>{};
  for (auto device : devices) {
    result.push_back(device.stats);
  }
  return result;
}

CudaResource cuda_resource;

#endif
