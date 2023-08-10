#pragma once
#include <string>
#include <vector>
#include "macro.hpp"
constexpr int maxStringSize = 16;


DLL std::vector<std::string> &hexdigest(const std::vector<std::string> &results);

DLL int md5_gpu(const char**chunk,int threads, std::string targetMd5, size_t chunkSize);