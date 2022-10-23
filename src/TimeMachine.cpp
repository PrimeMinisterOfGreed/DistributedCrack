#include "TimeMachine.hpp"
#include <chrono>
#include <ctime>
#include <functional>
#include <ratio>

auto executeTime(std::function<void()> lambda)
{
    auto startTime = std::chrono::system_clock::now();
    lambda();
    auto endTime = std::chrono::system_clock::now();
    return endTime - startTime;
}

std::chrono::milliseconds executeTimeComparison(std::function<void()> lambda)
{
    return std::chrono::duration_cast<std::chrono::milliseconds>(executeTime(lambda));
}

std::chrono::microseconds executeMicroTimeComparison(std::function<void()> lambda)
{
    return std::chrono::duration_cast<std::chrono::microseconds>(executeTime(lambda));
}