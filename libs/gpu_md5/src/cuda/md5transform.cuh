#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include "_cdecl"
CDECL
struct md5_transform_request {
  uint8_t *data;
  uint8_t *sizes;
  char *result;
  uint32_t *offsets;
  size_t num_of_strings;

};


struct md5_transform_request new_request(uint8_t *data, uint8_t *sizes, size_t num_of_strings);
void free_request(struct md5_transform_request *req);
void md5_gpu_transform(struct md5_transform_request request, int maxthreads);
END