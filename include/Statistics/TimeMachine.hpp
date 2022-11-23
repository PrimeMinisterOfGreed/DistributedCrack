#pragma once
#include <chrono>
#include <functional>
#include "Statistics/Event.hpp"

struct Event;
std::chrono::milliseconds executeTimeComparison(std::function<void()> lambda);
std::chrono::microseconds executeMicroTimeComparison(std::function<void()> lambda);

using namespace std::chrono;

class StopWatch
{
private:
	time_point<system_clock> _instantiationTime = system_clock::now();
	nanoseconds _lastTime;
public:
	Event& RecordEvent(std::function<void(Event& e)> function);
	void Start();
	nanoseconds Now();
};