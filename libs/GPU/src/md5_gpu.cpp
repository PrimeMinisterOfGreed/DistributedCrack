#include "md5_gpu.hpp"
#include "DataTypes.cuh"
#include "md5Cuda.cuh"
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string.h>
#include <string>
#include <sys/types.h>
#include <vector>

const char *hexdigest(const uint8_t *digest)
{

    char *buf = new char[33];
    for (int i = 0; i < 16; i++)
        sprintf(buf + i * 2, "%02x", digest[i]);
    buf[32] = '\000';

    return buf;
}

uint8_t *digesthex(std::string md5)
{
    std::vector<unsigned char> digest;

    for (size_t i = 0; i < md5.size(); i += 2)
    {
        std::string byte_string = md5.substr(i, 2);
        unsigned char byte = (unsigned char)strtol(byte_string.c_str(), NULL, 16);
        digest.push_back(byte);
    }
    uint8_t *result = new uint8_t[4 * sizeof(uint32_t)];
    for (int i = 0; i < 16; i++)
    {
        result[i] = digest[i];
    }
    return result;
}

std::vector<std::string> &hexdigest(const std::vector<std::string> &results)
{
    std::vector<std::string> &result = *new std::vector<std::string>();
    for (size_t i = 0; i < results.size(); i++)
    {
        result.push_back(hexdigest((uint8_t *)results[i].c_str()));
    }
    return result;
}


int md5_gpu(const char ** chunk, int threads, const char* targetMd5, size_t chunkSize)
{
    CheckGpuCondition();
    return md5_gpu(chunk, chunkSize, threads, targetMd5);
}

