#pragma  once
#include <cuda.h>
#include <cuda_runtime.h>

struct md5_bruter_request{
    char target_md5[33];
    size_t address_start;
    size_t address_end;
    int base_str_len;
    char target_found[64];
};


void md5_gpu_brute(struct md5_bruter_request* request, int threads);
