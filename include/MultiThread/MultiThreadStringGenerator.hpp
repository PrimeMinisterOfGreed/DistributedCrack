#pragma once
#include "StringGenerator.hpp"

class MultiThreadStringGenerator : public AssignedSequenceGenerator
{
  private:
    std::mutex _guard;
  public:
    MultiThreadStringGenerator(int initialSequenceLength);
    std::vector<std::string> & SafeGenerateChunk(int num);
};

