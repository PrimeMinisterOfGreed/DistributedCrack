#pragma once 
#include<cstddef>
#include<string>
#include<optional>
#include<vector>

std::optional<size_t> compute_chunk(std::vector<std::string>&chunk,std::string target ,int threads);