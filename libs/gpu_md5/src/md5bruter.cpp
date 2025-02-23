#include "md5_gpu.hpp"
#include "cuda/md5bruter.cuh"
#include <memory.h>

std::optional<std::string> md5_bruter(size_t start_address, size_t end_address, std::string target_md5, int maxthreads, int base_str_len){
  char target_found[64]{};
  char target_md5_buffer[33]{};
  memcpy(target_md5_buffer, target_md5.c_str(), 33);
  md5_gpu_brute(target_md5_buffer, start_address, end_address, base_str_len,
                target_found, maxthreads);
  for (int i = 0; i < 64; i++)
    if (target_found[i] != 0)
      return {target_found};
  return {};
}