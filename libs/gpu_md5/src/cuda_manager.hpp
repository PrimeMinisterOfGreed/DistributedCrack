#pragma once

struct CudaManager{
    static CudaManager* instance();
    void init();
    void select_gpu();
    void disable_current_gpu();
    private:
    CudaManager();
};