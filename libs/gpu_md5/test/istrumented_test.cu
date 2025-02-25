#include <string>
#include "md5_gpu.hpp"

int main(){

  std::string target = "5d41402abc4b2a76b9719d911017c592";
  std::optional<std::string> res{};
  res.reset();
  size_t currentaddress=0, chunksize =20000;
  while(!res.has_value()){
    res=md5_bruter(currentaddress, currentaddress+chunksize, target,300);
  }
}