#pragma once
#include <cstddef>
#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>

void md5_gpu_transform(uint8_t *data, uint32_t *sizes, uint32_t *result, size_t size, int maxthreads);

void CheckGpuCondition();

