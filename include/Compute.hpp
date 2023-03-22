#pragma once
#include "Concepts.hpp"
#include <cmath>
#include <cstddef>
#include <fmt/core.h>
#include <functional>
#include <future>
#include <regex.h>
#include <string>
#include <thread>
#include <vector>

template <HashFunction Fnc = std::function<std::string(std::string)>> struct Compute
{
  protected:
    Fnc _fnc;

  public:
    Compute(Fnc fnc) : _fnc(fnc)
    {
    }
    virtual bool operator()(const std::vector<std::string> &chunk, std::string target, std::string *result)
    {
        for (auto &val : chunk)
        {
            auto hash = _fnc(val);
            if ( hash == target)
            {
                
                *result = val;
                return true;
            }
        }
        return false;
    }
};

template <HashFunction Fnc = std::function<std::string(std::string)>> struct AsyncCompute
{
  protected:
    Compute<Fnc> _compute;

  public:
    AsyncCompute(Fnc fnc) : _compute(Compute<Fnc>(fnc))
    {
    }
    virtual std::future<bool> operator()(const std::vector<std::string> &chunk, std::string target, std::string *result)
    {
        return std::async(_compute);
    };
};

template <HashFunction Fnc = std::function<std::string(std::string)>> struct MTCompute : public Compute<Fnc>
{
  protected:
    int _threads;

  public:
    MTCompute(Fnc fnc, int threads = 16) : _threads(threads), Compute<Fnc>(fnc)
    {
    }
    virtual bool operator()(const std::vector<std::string> &chunk, std::string target, std::string *result) override
    {
        size_t perSizeDivision = std::ceil(chunk.size() / _threads);
        std::thread threadArray[_threads];
        bool forceEnd = false;
        size_t currentPtr = 0;
        auto fncPtr = Compute<Fnc>::_fnc;

        for (int i = 0; i < _threads; i++)
        {
            threadArray[i] = std::thread([&forceEnd, currentPtr, perSizeDivision, chunk, target, result,fncPtr]() {
                for (size_t i = currentPtr; i < currentPtr + perSizeDivision && i < chunk.size() && !forceEnd; i++)
                {
                    if (fncPtr(chunk.at(i)) == target)
                    {
                        *result = chunk.at(i);
                        forceEnd = true;
                    }
                }
            });
        }
        for (int i = 0; i < _threads; i++)
            threadArray[i].join();
        return forceEnd;
    }
};

