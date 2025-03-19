#include "md5_gpu.hpp"
#include "cuda/md5transform.cuh"
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


std::vector<std::string> md5_gpu(const std::vector<std::string> &chunk, int maxthreads) {

  uint32_t * results = new uint32_t[chunk.size()*4]{},
  *sizes = new uint32_t[chunk.size()]{};

  size_t data_size = chunk[0].size();
  sizes[0] = chunk[0].size();
  //calculate data size and store each size in sizes vector
  for(int i = 1 ; i < chunk.size(); i++){
    data_size += chunk[i].size();
    sizes[i] = chunk[i].size();
  }
  uint8_t *data = new uint8_t[data_size]{};
  size_t offset = 0;
  for(int i = 0 ; i < chunk.size(); i++){
    memcpy(&data[offset],chunk[i].c_str(),sizes[i]);
    offset+=sizes[i];
  }

  md5_gpu_transform(data, sizes, results, chunk.size(),maxthreads);
  std::vector<std::string> md5s;
  for(int i = 0 ; i < chunk.size()*4; i+=4){
    char resultstr[33]{};
    hexdigest(reinterpret_cast<uint8_t*>(&results[i]), resultstr);
    md5s.push_back(std::string(resultstr));
  }
  free(results);
  free(sizes);
  free(data);
  return md5s;

}


extern "C" void pong(){
  
}