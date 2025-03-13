#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <stddef.h>
#include <vector>

struct CudaManager{
    int rebirths = 0;
    static CudaManager* instance();
    void init();
    void select_gpu();
    void disable_current_gpu();
    void force_gpu(int gpu);
    private:
    CudaManager();
};