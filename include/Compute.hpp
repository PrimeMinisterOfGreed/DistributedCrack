#pragma once
#include "Concepts.hpp"
#include <cmath>
#include <cstddef>
#include <functional>
#include <future>
#include <regex.h>
#include <string>
#include <thread>
#include <vector>

class IHashComparer
{
  public:
    virtual bool Compute(const std::vector<std::string> &chunk, std::string target, std::string *result, std::function<std::string(std::string)> hashFnc) = 0;
};

class IAsyncHashComparer: public IHashComparer
{
  public:
   virtual std::future<bool> ComputeAsync(const std::vector<std::string> &chunk, std::string target, std::string *result,
                               std::function<std::string(std::string)> hashFnc) = 0;
};





template <HashFunction Fnc>
bool Compute(const std::vector<std::string> &chunk, std::string target, std::string *result, Fnc hashFnc)
{
    for (auto &val : chunk)
    {
        if (hashFnc(val) == target)
        {
            *result = val;
            return true;
        }
    }
    return false;
}

template <HashFunction Fnc>
std::future<bool> ComputeAsync(const std::vector<std::string> &chunk, std::string target, std::string *result,
                               Fnc hashFnc)
{
    return std::async([chunk, target, result, hashFnc]() -> bool { Compute(chunk, result, hashFnc); });
}

template <HashFunction Fnc>
bool MTCompute(const std::vector<std::string> &chunk, std::string target, std::string *result, Fnc hashFnc,
               int threads = 16)
{
    size_t perSizeDivision = std::ceil(chunk.size() / threads);
    std::thread threadArray[threads];
    bool forceEnd = false;
    size_t currentPtr = 0;
    for (int i = 0; i < threads; i++)
    {
        threadArray[i] = std::thread([&forceEnd, hashFnc, currentPtr, perSizeDivision, chunk, target,result]() {
            for (size_t i = currentPtr; i < currentPtr + perSizeDivision && i < chunk.size() && !forceEnd; i++)
            {
                if (hashFnc(chunk.at(i)) == target)
                {
                    *result = chunk.at(i);
                    forceEnd = true;
                }
            }
        });
    }
    for (int i = 0; i < threads; i++)
        threadArray[i].join();
}


