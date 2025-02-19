#include <string>
#include "md5_gpu.hpp"

int main(){
    std::vector<std::string> chunk{"foo","bar","hello world"};
    std::string tgt = "5eb63bbbe01eeed093cb22bb8f5acdc3";
    auto index = md5_gpu(chunk);
    
}