#pragma once
#include "Concepts.hpp"
#include "TaskProvider/HashTask.hpp"
#include <cstdint>


class HashTaskGenerator
{
  private:
    int64_t _currentAddress;
    int32_t _chunkSize;
    std::string _targetHash;
  public:
    HashTaskGenerator(int32_t chunkSize, std::string targetHash): _chunkSize(chunkSize){}
    HashTask &operator()();
};

