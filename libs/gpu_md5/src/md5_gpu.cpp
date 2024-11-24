#include "md5_gpu.hpp"
#include "DataTypes.cuh"
#include "md5Cuda.cuh"
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string.h>
#include <string>
#include <sys/types.h>
#include <vector>

const char *hexdigest(const uint8_t *digest) {

  char *buf = (char *)std::malloc(33);
  for (int i = 0; i < 16; i++)
    sprintf(buf + i * 2, "%02x", digest[i]);
  buf[32] = '\000';

  return buf;
}

std::unique_ptr<uint8_t[]> digesthex(std::string md5) {
  std::vector<unsigned char> digest;

  for (size_t i = 0; i < md5.size(); i += 2) {
    std::string byte_string = md5.substr(i, 2);
    unsigned char byte = (unsigned char)strtol(byte_string.c_str(), NULL, 16);
    digest.push_back(byte);
  }
  auto result = std::make_unique<uint8_t[]>(4 * sizeof(uint32_t));
  for (int i = 0; i < 16; i++) {
    result[i] = digest[i];
  }
  return result;
}

std::vector<std::string> hexdigest(const std::vector<std::string> &results) {
  std::vector<std::string> result{};
  for (size_t i = 0; i < results.size(); i++) {
    result.push_back(hexdigest((uint8_t *)results[i].c_str()));
  }
  return result;
}

int md5_gpu(const std::vector<std::string> &chunk,
            std::string targetMd5) {
  CheckGpuCondition();
  size_t sum = 0;

  auto sizes = new uint32_t[chunk.size()]{};
  for (size_t i = 0; i < chunk.size(); i++) {
    sizes[i] = chunk[i].size();
    sum += chunk[i].size();
  }
  auto data = new uint8_t[sum]{};
  auto results = new uint8_t[chunk.size() * sizeof(uint32_t) *4]{}; // every state vector is 4 elements composed of 4 bytes
  size_t offset = 0;
  for (int i = 0; i < chunk.size(); i++) {
    auto str = chunk.at(i).c_str();
    auto size = sizes[i];
    memcpy(data + offset, str, sizeof(uint8_t) * size);
    offset += size;
  }
  
  int result = md5_gpu_finder(data, sizes, chunk.size() , digesthex(targetMd5).get());
  delete[] sizes;
  delete[] data;
  delete[] results;
  return result;
}

std::vector<std::string> md5_gpu(const std::vector<std::string> &chunk) {
  CheckGpuCondition();
  std::vector<std::string> resultsVector{};
  size_t sum = 0;
  auto sizes = std::make_unique<uint32_t[]>(chunk.size());
  for (size_t i = 0; i < chunk.size(); i++) {
    sizes[i] = chunk[i].size();
    sum += chunk[i].size();
  }
  auto data = std::make_unique<uint8_t[]>(chunk.size() * sum);
  auto results = std::make_unique<uint8_t[]>(
      chunk.size() * sizeof(uint32_t) *
      4); // every state vector is 4 elements composed of 4 bytes
  size_t offset = 0;
  for (int i = 0; i < chunk.size(); i++) {
    auto str = chunk.at(i).c_str();
    auto size = sizes[i];
    memcpy(data.get() + offset, str, sizeof(uint8_t) * size);
    offset += size;
  }
  data[sum] = '\0';
  md5_gpu_transform(data.get(), sizes.get(), results.get(), chunk.size());
  for (int i = 0; i < chunk.size(); i++) {
    resultsVector.push_back(
        std::string((char *)results.get() + (i * sizeof(uint32_t) * 4),
                    sizeof(uint32_t) * 4));
  }
  return resultsVector;
}

std::string md5_gpu(size_t start, size_t end) { return std::string(); }