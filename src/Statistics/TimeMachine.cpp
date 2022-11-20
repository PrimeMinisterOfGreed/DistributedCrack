#include "Statistics/TimeMachine.hpp"
#include <chrono>
#include <ctime>
#include <functional>
#include <ratio>

using namespace std::chrono;
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

Event& StopWatch::RecordEvent(std::function<void(Event& e)> function)
{
	auto startTime = Now();
	double interarrival = (startTime - _lastTime).count();
		Event & newEvent = *new Event();
	//TODO assegna start time, computa la funzione e aggiungi serviceTime
	newEvent.arrivalTime = startTime.count();
	function(newEvent);
	newEvent.serviceTime = (Now() - startTime).count();
	return newEvent;
}

void StopWatch::Start()
{
	_instantiationTime = system_clock::now();
	_lastTime = Now();
}

nanoseconds StopWatch::Now()
{
	return system_clock::now() - _instantiationTime;
}
