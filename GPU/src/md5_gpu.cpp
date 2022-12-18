#include "md5_gpu.hpp"
#include "md5Cuda.cuh"
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <cstdlib>
#include <sys/types.h>
#include <vector>
#include <string.h>

template <typename T> T *malloc(size_t size)
{
    auto mem = (T *)std::malloc(sizeof(T) * size);
    return mem;
}

__host__ const char *hexdigest(const uint8_t *digest)
{

    char *buf = (char *)std::malloc(33);
    for (int i = 0; i < 16; i++)
        sprintf(buf + i * 2, "%02x", digest[i]);
    buf[32] = 0;

    return buf;
}

std::vector<std::string> md5_gpu(std::vector<std::string> &chunk, int threads)
{
    std::vector<std::string> resultsVector = std::vector<std::string>();
    uint32_t *sizes = malloc<uint32_t>(chunk.size());
    uint8_t **data = malloc<uint8_t *>(chunk.size());
    uint8_t **results = malloc<uint8_t *>(chunk.size());
    for (int i = 0; i < chunk.size(); i++)
    {
        int size = chunk.at(i).size();
        data[i] = malloc<uint8_t>(size);
        sizes[i] = size;
        results[i] = malloc<uint8_t>(size);
        memcpy(data[i], chunk.at(i).c_str(), sizeof(uint8_t) * size);
    }
    md5_gpu((const uint8_t **)data, sizes, results, chunk.size(), threads);
    for (int i = 0; i < chunk.size(); i++)
    {
        resultsVector.push_back(std::string(hexdigest(results[i])));
    }
    std::free(data);
    std::free(results);
    std::free(sizes);
    return resultsVector;
}
