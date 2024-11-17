#include "functions.hpp"
#include "md5.hpp"

std::optional<size_t> compute_chunk(std::vector<std::string>&chunk,std::string target ,int threads) {
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
    if(found) return position;
    return {};
}
