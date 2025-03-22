#pragma once
#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
struct Md5TransformResult{
    char* data;
    size_t size;
}; 

struct Md5BruterResult{
    char data[33];
    bool found;
};


struct Md5TransformResult md5_gpu(char* data, uint32_t * sizes, size_t array_size, int maxthreads);
struct Md5BruterResult md5_bruter(size_t start_address, size_t end_address, const char* target_md5, int maxthreads,int base_str_len);
#ifdef __cplusplus
}
#endif



