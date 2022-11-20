#pragma once
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics.hpp>
#include "Statistics/TimeMachine.hpp"
#include <chrono>

struct Event
{
	friend class IStopWatch;
	double arrivalTime;
	double serviceTime;
	double completitions = 1.0;
	Event(double arrivalTime, double serviceTime, double completitions) : arrivalTime{ arrivalTime }, serviceTime{ serviceTime }, completitions{ completitions } {}
	Event(){}
};



