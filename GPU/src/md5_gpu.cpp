#include "md5_gpu.hpp"
#include "md5Cuda.cuh"
#include <cstdint>
#include <cstdlib>
#include <stdlib.h>
#include <string>
#include <sys/types.h>
#include <vector>

const uint8_t *stringToArray(const std::string string)
{
    const uint8_t *converted = (const uint8_t *)string.data();
    return converted;
}

std::vector<std::string> md5_gpu(std::vector<std::string> &chunk)
{
}

std::string md5_gpu(std::string string)
{
    const uint8_t *converted = stringToArray(string);
    uint32_t *result = (uint32_t *)std::malloc(sizeof(uint32_t) * 4);
    md5_gpu(converted, string.size(), result);
    char buf[33];
    for (int i = 0; i < 16; i++)
        sprintf(buf + i * 2, "%02x", result[i]);
    buf[32] = 0;
    return std::string(buf);
}
