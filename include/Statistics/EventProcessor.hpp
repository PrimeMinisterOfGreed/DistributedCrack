#pragma once
#include "Statistics/Event.hpp"
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics.hpp>

using namespace boost::accumulators;

using Accumulator = accumulator_set<double, features<tag::mean, tag::max, tag::min, tag::variance>>;

class IEventProcessor
{
private:
	Accumulator _arrivalTimes;
	Accumulator _interArrivals;
	Accumulator _serviceTimes;
	std::vector<Event>& _events = *new std::vector<Event>();
public:
	virtual void ComputeStatistics();
	virtual void AddEvent(Event& e) const { _events.push_back(e); }
	IEventProcessor(std::vector<Event>& events) : _events{ events } {}
	IEventProcessor() {}
};
