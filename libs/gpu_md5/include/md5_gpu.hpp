#pragma once
#include <string>
#include <vector>

 std::vector<std::string> md5_gpu(const std::vector<std::string> &chunk, int maxthreads = 10000);
 std::vector<std::string> hexdigest(const std::vector<std::string> &results);

bool md5_bruter(size_t start_address, size_t end_address, std::string target_md5);