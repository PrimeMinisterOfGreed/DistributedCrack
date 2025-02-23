#include <string>
#include "md5_gpu.hpp"

int main(){

  std::string target = "5d41402abc4b2a76b9719d911017c592";
  std::optional<std::string> res{};
  res.reset();
  size_t currentaddress=0, chunksize =10000;
  for(int i = 0 ; i < 10; i++){
    printf("computing with %ld threads \n",chunksize);
    res=md5_bruter(currentaddress, currentaddress+chunksize, target,chunksize);
  }
}