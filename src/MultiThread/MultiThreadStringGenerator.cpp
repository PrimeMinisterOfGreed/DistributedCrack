#include "MultiThread/MultiThreadStringGenerator.hpp"
#include "StringGenerator.hpp"
#include <mutex>

MultiThreadStringGenerator::MultiThreadStringGenerator(int initialSequenceLength)
    : AssignedSequenceGenerator(initialSequenceLength)
{
}

std::vector<std::string> &MultiThreadStringGenerator::SafeGenerateChunk(int num)
{
    std::lock_guard<std::mutex> l(_guard);
    auto &result = ISequenceGenerator::generateChunk(num);
    return result;
}
