#include "string_generator.hpp"
#include <boost/exception/exception.hpp>
#include <cstdint>
#include <exception>
#include <mutex>
#include <string>
#include <vector>

SequentialGenerator::SequentialGenerator(int initialSequenceLength) : _current()
{
    for (int i = 0; i < (initialSequenceLength>0?initialSequenceLength:1); i++)
    {
        _current += minCharInt;
    }
}

std::string SequentialGenerator::next_sequence()
{
    auto result = _current;
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

std::vector<std::string> ISequenceGenerator::generate_chunk(int num)
{
    std::vector<std::string> result{};
    for (int i = 0; i < num; i++)
        result.push_back(next_sequence());
    return result;
};

void ISequenceGenerator::generate_chunk(char*buffer, size_t * sizes, int num) {
  std::string current{};
  size_t displ = 0;  
  for(int i = 0 ; i < num; i++){
    current = next_sequence();
    memcpy(&buffer[displ], current.c_str(), current.size());
    displ+= current.size();
    sizes[i] = current.size();
  }
}

AssignedSequenceGenerator::AssignedSequenceGenerator(int initlength)
    : SequentialGenerator(initlength), _currentSequenceIndex(0)
{
}

void AssignedSequenceGenerator::assign_address(uint64_t address)
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
    _currentUsed = false;
}

std::string AssignedSequenceGenerator::next_sequence()
{
    if(!_currentUsed){
        _currentUsed = true;
        return _current;
    }
    _currentSequenceIndex++;
    return SequentialGenerator::next_sequence();
}

