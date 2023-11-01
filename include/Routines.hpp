#pragma once
#include "Async/Async.hpp"
#include <string>
#include <vector>

Future<std::string, std::vector<std::string>>
compute_gpu(std::vector<std::string> chunk);

Future<std::string, std::vector<std::string>>
compute(std::vector<std::string> chunk, int maxThreads = 16);