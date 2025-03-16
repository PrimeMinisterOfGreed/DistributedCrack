#include <future>
#include <string>
#include "../include/md5_gpu.hpp"
#include <optional>
#include "../src/cuda_manager.hpp"


int main(){
  CudaManager::instance()->init();
  std::string target = "5d41402abc4b2a76b9719d911017c592";
  std::optional<std::string> res{};
  res.reset();
  size_t currentaddress=0, chunksize =100000,threads=100000;
  while(!res.has_value()){
    res=md5_bruter(currentaddress, currentaddress+chunksize, target,chunksize);
    currentaddress+=chunksize;
  }
}