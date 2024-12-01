#pragma once
#include <cstddef>
#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>

__host__ void md5_gpu_transform(const uint8_t *data, const uint32_t *sizes, uint8_t *result, uint32_t size);
__host__ int md5_gpu_finder(const uint8_t *data, const uint32_t *sizes, uint32_t size, uint8_t * targetDigest);
__host__ void CheckGpuCondition();

