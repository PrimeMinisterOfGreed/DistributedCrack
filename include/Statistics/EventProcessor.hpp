#pragma once
#include "Statistics/Event.hpp"
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics.hpp>
#include <boost/serialization/serialization.hpp>
using namespace boost::accumulators;

using Accumulator = accumulator_set<double, features<tag::mean, tag::max, tag::min, tag::variance>>;

struct Statistics
{
	double meanInterarrival;
	double meanServiceTime;
	double maxInterArrival;
	double maxServiceTime;
	double arrivalRate;
	double throughput;
	double serviceRate;
	double utilization;
	double completitions;
	double busyTime;
	double observationPeriod;
};



class EventProcessor
{
private:
	Accumulator _arrivalTimes;
	Accumulator _interArrivals;
	Accumulator _serviceTimes;
	Statistics& _actual = *new Statistics;
	std::vector<Event>& _events = *new std::vector<Event>();
public:
	virtual Statistics& ComputeStatistics();
	virtual void AddEvent(Event& e) const { _events.push_back(e); }
	EventProcessor(std::vector<Event>& events) : _events{ events } {}
	EventProcessor() {}
};

namespace boost::serialization
{
	template<class Archive>
	void serialize(Archive& ar, Statistics& stat, const unsigned int version)
	{
		ar& meanInterarrival;
		ar& meanServiceTime;
		ar& maxInterArrival;
		ar& maxServiceTime;
		ar& arrivalRate;
		ar& throughput;
		ar& serviceRate;
		ar& utilization;
		ar& completitions;
		ar& busyTime;
		ar& observationPeriod;
	}

}