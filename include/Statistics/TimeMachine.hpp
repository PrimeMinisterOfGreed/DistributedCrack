#pragma once
#include <chrono>
#include <functional>
#include "Statistics/Event.hpp"

std::chrono::milliseconds executeTimeComparison(std::function<void()> lambda);
std::chrono::microseconds executeMicroTimeComparison(std::function<void()> lambda);

using namespace std::chrono;

class IStopWatch
{
private:
	time_point<steady_clock> _lastTime = steady_clock::now();
public:
	Event& RecordEvent(std::function<void(Event& e)> function);

};