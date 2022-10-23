#include "StringGenerator.hpp"
#include <boost/exception/exception.hpp>
#include <cstdint>
#include <exception>
#include <string>
#include <vector>

SequentialGenerator::SequentialGenerator(int initialSequenceLength) : _current(*new std::string())
{
    for (int i = 0; i < initialSequenceLength; i++)
    {
        _current += minCharInt;
    }
}

std::string SequentialGenerator::nextSequence()
{
    auto result = std::string(_current);
    for (int i = _current.size() - 1; i >= 0; i--)
    {
        _current[i]++;
        if (_current[i] > maxCharint)
        {
            _current[i] = minCharInt;
            if (i == 0)
            {
                _current += minCharInt;
                _currentSequenceLength = _current.length();
            }
        }
        else
        {
            break;
        }
    }
    return result;
};

std::vector<std::string> &SequentialGenerator::generateChunk(int num)
{
    std::vector<std::string> &result = *new std::vector<std::string>{};
    for (int i = 0; i < num; i++)
        result.push_back(nextSequence());
    return result;
};

AssignedSequenceGenerator::AssignedSequenceGenerator(int initlength)
    : SequentialGenerator(initlength), _currentSequenceIndex(0)
{
}

void AssignedSequenceGenerator::AssignAddress(uint64_t address)
{
    if (address < _currentSequenceIndex)
        throw GeneratorException((char *)"Address is beyond bounds");
    for (int i = 0; i < address - _currentSequenceIndex; i++)
    {
        nextSequence().clear();
    }
}

std::string AssignedSequenceGenerator::nextSequence()
{
    _currentSequenceIndex++;
    return SequentialGenerator::nextSequence();
}