#pragma once
#include "macro.hpp"
#include <string>
#include <vector>

DLL std::vector<std::string> md5_gpu(const std::vector<std::string> &chunk,
                                     int threads);
DLL std::vector<std::string> hexdigest(const std::vector<std::string> &results);

DLL int md5_gpu(const std::vector<std::string> &chunk, int threads,
                std::string targetMd5);