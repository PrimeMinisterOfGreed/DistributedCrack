#include "functions.hpp"
#include "log_engine.hpp"
#include "md5.hpp"

std::optional<std::string> compute_chunk(std::vector<std::string>& chunk,std::string target ,int threads) {
    size_t position = 0;
    bool found = false;
    #pragma omp parallel num_threads(threads) if(threads>1) 
    for(size_t i = 0; i < chunk.size(); i++){
        auto md5 = MD5(chunk[i]);
        if(md5.hexdigest() == target){
            position = i;
            found = true;
        }
    };
    if(found) return chunk[position];
    return {};
}

flatten_result flatten_chunk(std::vector<std::string>& chunk, char* buffer) {
    std::vector<int> sizes{};
    std::vector<int> disp{};
    sizes.resize(chunk.size());
    disp.resize(chunk.size());
    for (auto i = 0; i < chunk.size(); i++) {
      sizes[i] = chunk[i].size();
      if (i > 0){
        disp[i] = chunk[i - 1].size() + disp[i - 1];
      }
    }
    for (auto i = 1; i < chunk.size(); i++) {
    }
#pragma omp parallel
    for (auto i = 0; i < chunk.size(); i++) {
      memcpy(&buffer[disp[i]], chunk[i].c_str(), sizes[i]);
    }
    return {sizes, disp};
}
