#pragma once
#include "TaskProvider/Tasks.hpp"
#include <boost/serialization/access.hpp>
#include <boost/serialization/vector.hpp>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

struct HashTask : public ITask
{
    int64_t _boundaries[2];
    int _startSequence;
    std::string target;
    std::string * result = nullptr;
};
