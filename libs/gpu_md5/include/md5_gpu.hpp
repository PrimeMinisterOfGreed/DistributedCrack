#pragma once
#include <string>
#include <vector>

 std::vector<std::string> md5_gpu(const std::vector<std::string> &chunk,
                                     int threads);
 std::vector<std::string> hexdigest(const std::vector<std::string> &results);

 int md5_gpu(const std::vector<std::string> &chunk, int threads,
                std::string targetMd5);
