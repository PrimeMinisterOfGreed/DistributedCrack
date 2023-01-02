#pragma once
#include <string>
#include <vector>
#include "macro.hpp"


DLL std::vector<std::string>& md5_gpu(const std::vector<std::string> &chunk, int threads);
DLL std::vector<std::string> &hexdigest(const std::vector<std::string> &results);
