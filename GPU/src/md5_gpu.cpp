#include "md5_gpu.hpp"
#include "md5Cuda.cuh"
#include <cstdint>
#include <string>
#include <sys/types.h>
#include <vector>

uint8_t *stringToArray(const std::string string)
{
    uint8_t *converted = new uint8_t[string.size()];
    for (int i = 0; i < string.size(); i++)
        converted[i] = string.at(i);
    return converted;
}

std::vector<std::string> md5_gpu(std::vector<std::string> &chunk)
{
}

std::string md5_gpu(std::string string)
{
    uint8_t *converted = stringToArray(string);
    uint32_t *result = new uint32_t[4];
    md5(converted, string.size(), result);
}