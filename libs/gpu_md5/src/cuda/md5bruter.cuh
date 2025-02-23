#pragma  once
#include <cuda.h>
#include <cuda_runtime.h>



void md5_gpu_brute(const char target_md5[33], size_t address_start,size_t address_end, int base_str_len, char target_found[64], int threads);