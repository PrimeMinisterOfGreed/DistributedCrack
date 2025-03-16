#include "md5_gpu.hpp"
#include "cuda/md5bruter.cuh"
#include <memory.h>

std::optional<std::string> md5_bruter(size_t start_address, size_t end_address, std::string target_md5, int maxthreads, int base_str_len){
  struct md5_bruter_request req;
  memset(&req, 0, sizeof(req));
  req.address_start = start_address;
  req.address_end = end_address;
  req.base_str_len = base_str_len;
  memcpy(req.target_md5, target_md5.c_str(), 33);
  md5_gpu_brute(&req, maxthreads);
  for (int i = 0; i < 64; i++)
    if (req.target_found[i] != 0)
      return {req.target_found};
  return {};
}