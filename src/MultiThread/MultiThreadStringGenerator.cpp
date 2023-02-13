#include "MultiThread/MultiThreadStringGenerator.hpp"
#include "StringGenerator.hpp"

MultiThreadStringGenerator::MultiThreadStringGenerator(int initialSequenceLength)
    : AssignedSequenceGenerator(initialSequenceLength)
{
}

std::vector<std::string> &MultiThreadStringGenerator::SafeGenerateChunk(int num)
{
    _guard.lock();
    auto &result = ISequenceGenerator::generateChunk(num);
    _guard.unlock();
    return result;
}
