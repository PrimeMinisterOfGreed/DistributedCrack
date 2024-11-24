#pragma once
#include <string>
#include <vector>

 std::vector<std::string> md5_gpu(const std::vector<std::string> &chunk);
 std::vector<std::string> hexdigest(const std::vector<std::string> &results);

 int md5_gpu(const std::vector<std::string> &chunk,
                std::string targetMd5);
