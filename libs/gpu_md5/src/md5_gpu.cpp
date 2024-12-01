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

void hexdigest(const uint8_t *digest, char* resultstring) {

  for (int i = 0; i < 16; i++)
    sprintf(resultstring + i * 2, "%02x", digest[i]);
  resultstring[32] = '\000';
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
    char resultstr[33]{};
    hexdigest((uint8_t *)results[i].c_str(),resultstr);
    result.push_back(resultstr);
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

  std::vector<uint32_t> offsets(chunk.size());
  std::vector<uint8_t> results(chunk.size()*4*sizeof(uint32_t));
  std::vector<uint32_t> sizes(chunk.size());

  size_t data_size = chunk[0].size();
  sizes[0] = chunk[0].size();
  auto _threads = std::min(16,(int) chunk.size()/1000);
  //calculate data size and store each size in sizes vector
  #pragma omp parallel threads(_threads) if(_threads > 1)
  for(int i = 1 ; i < chunk.size(); i++){
    data_size += chunk[i].size();
    sizes[i] = chunk[i].size();
    offsets[i] = chunk[i].size() + offsets[i-1];
  }
  std::vector<uint8_t> data(data_size);  

  for(int i = 0 ; i < chunk.size(); i++){
    memcpy(data.data() + offsets[i],chunk[i].data(),sizes[i]);
  }
  md5_gpu_transform(data.data(), sizes.data(), results.data(), chunk.size());
  std::vector<std::string> md5s;
  for(int i = 0 ; i < results.size(); i+=16){
    char resultstr[33]{};
    hexdigest(results.data() + i, resultstr);
    md5s.push_back(std::string{resultstr});
  }
  return md5s;

}

std::string md5_gpu(size_t start, size_t end) { return std::string(); }