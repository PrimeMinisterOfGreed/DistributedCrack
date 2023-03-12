#pragma once
#include "TaskProvider/Tasks.hpp"
#include <boost/serialization/access.hpp>
#include <boost/serialization/vector.hpp>
#include <cstddef>
#include <string>
#include <vector>

struct HashTask : public ITask
{
    std::vector<std::string> &chunk;
    std::string target;
    std::string * result = nullptr;
    HashTask(std::vector<std::string> &chunk, std::string target) : chunk(chunk), target(target)
    {
    }
};
