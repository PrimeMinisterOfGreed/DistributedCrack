#pragma once
#include <string>
#include <vector>
#include "macro.hpp"
DLL std::vector<std::string> md5_gpu(std::vector<std::string> &chunk, int threads);