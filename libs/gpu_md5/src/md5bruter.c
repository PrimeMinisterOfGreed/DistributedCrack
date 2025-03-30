#include "md5_gpu.h"
#include "cuda/md5bruter.cuh"
#include <memory.h>
#include <stdbool.h>


struct Md5BruterResult md5_bruter(size_t start_address, size_t end_address, const char* target_md5, int maxthreads,int base_str_len){
    struct Md5BruterResult result;
    memset(&result, 0, sizeof(result));
    struct md5_bruter_request request = new_bruter_request((char*)target_md5, start_address, end_address, base_str_len);
    md5_gpu_brute(&request, maxthreads);
    if(request.target_found[0] != 0){
        result.found = true;
        memcpy(result.data, request.target_found, 33);
    }
    return result;
}

struct md5_bruter_request new_bruter_request(char *target_md5, size_t address_start, size_t address_end, int base_str_len){
  struct md5_bruter_request request;
  memset(&request, 0, sizeof(request));
  request.address_start = address_start;
  request.address_end = address_end;
  request.base_str_len = base_str_len;
  memcpy(request.target_md5, target_md5, 33);
  return request;
}