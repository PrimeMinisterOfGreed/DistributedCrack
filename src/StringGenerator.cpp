#include "StringGenerator.hpp"
#include <boost/exception/exception.hpp>
#include <cstdint>
#include <exception>
#include <mutex>
#include <string>
#include <vector>

SequentialGenerator::SequentialGenerator(int initialSequenceLength) : _current(*new std::string())
{
    for (int i = 0; i < (initialSequenceLength>0?initialSequenceLength:1); i++)
    {
        _current += minCharInt;
    }
}

std::string SequentialGenerator::nextSequence()
{
    auto &result = *new std::string(_current);
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

std::vector<std::string> &ISequenceGenerator::generateChunk(int num)
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
    int div = maxCharint - minCharInt;
    int q = address;
    int r = 0;
    int it = 0;
    while (q > 0)
    {
        r = q % div;
        q /= div;
        if (it == _current.size())
        {
            _current.insert(_current.begin(), (char)minCharInt);
        }
        _current.at(_current.size() - it - 1) = (char)(r + minCharInt);
        it++;
    }
}

std::string AssignedSequenceGenerator::nextSequence()
{
    _currentSequenceIndex++;
    return SequentialGenerator::nextSequence();
}

