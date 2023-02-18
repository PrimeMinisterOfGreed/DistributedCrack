#pragma once
#include "StringGenerator.hpp"

class MultiThreadStringGenerator : public AssignedSequenceGenerator
{
  private:
    std::mutex& _guard = *new std::mutex();
  public:
    MultiThreadStringGenerator(int initialSequenceLength);
    std::vector<std::string> & SafeGenerateChunk(int num);
};

