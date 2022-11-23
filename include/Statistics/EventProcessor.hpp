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
	std::string& ToString()
	{
		std::stringstream stream{};
		stream << "Mean interval: " << meanInterarrival << std::endl
			<< "Mean service time: " << meanServiceTime << std::endl
			<< "Max interval: " << maxInterArrival << std::endl
			<< "Max service time: " << maxServiceTime << std::endl
			<< "Arrival rate: " << arrivalRate << std::endl
			<< "Throughput: " << throughput << std::endl
			<< "Service rate: " << serviceRate << std::endl
			<< "Utilization: " << utilization << std::endl
			<< "Completitions: " << completitions << std::endl
			<< "Busy Time: " << busyTime << "nanoseconds" << std::endl
			<< "Observation time: " << observationPeriod << "nanoseconds" << std::endl;
		return *new std::string(stream.str());
	}
};



class EventProcessor
{
private:
	Accumulator _arrivalTimes;
	Accumulator _interArrivals;
	Accumulator _serviceTimes;
	Statistics& _actual = *new Statistics();
	std::vector<Event>& _events = *new std::vector<Event>();
public:
	virtual Statistics& ComputeStatistics();
	virtual void AddEvent(Event& e) const { _events.push_back(e); }
	int ToCompute() const { return _events.size(); }
	EventProcessor(std::vector<Event>& events) : _events{ events } {}
	EventProcessor() {}
};

namespace boost::serialization
{
	template<class Archive>
	void serialize(Archive& ar, Statistics& stat, const unsigned int version)
	{
		ar& stat.meanInterarrival;
		ar& stat.meanServiceTime;
		ar& stat.maxInterArrival;
		ar& stat.maxServiceTime;
		ar& stat.arrivalRate;
		ar& stat.throughput;
		ar& stat.serviceRate;
		ar& stat.utilization;
		ar& stat.completitions;
		ar& stat.busyTime;
		ar& stat.observationPeriod;
	}

}