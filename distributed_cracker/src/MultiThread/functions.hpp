#pragma once 
#include<cstddef>
#include<string>
#include<optional>
#include<vector>

std::optional<std::string> compute_chunk(std::vector<std::string>&chunk,std::string target ,int threads);

struct flatten_result{
    std::vector<int> sizes{};
    std::vector<int> disp{};
};

flatten_result flatten_chunk(std::vector<std::string>& chunk, char* buffer);
