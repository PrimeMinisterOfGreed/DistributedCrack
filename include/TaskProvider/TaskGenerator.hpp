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
    int _initialSequenceLength = 0;
  public:
    HashTaskGenerator(int32_t chunkSize, std::string targetHash, int32_t initialSequennceLength = 1): _chunkSize(chunkSize), _targetHash(targetHash), _initialSequenceLength(initialSequennceLength){}
    HashTask &operator()();
};

