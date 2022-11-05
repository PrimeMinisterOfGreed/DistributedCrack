#pragma once
#include <chrono>
#include <functional>

std::chrono::milliseconds executeTimeComparison(std::function<void()> lambda);
std::chrono::microseconds executeMicroTimeComparison(std::function<void()> lambda);

