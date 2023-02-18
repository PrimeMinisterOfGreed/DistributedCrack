#pragma once
#include "Statistics/Event.hpp"
#include "Statistics/Measure.hpp"
#include <cstddef>
#include <vector>
using namespace boost::accumulators;

struct Statistics
{
    double meanInterarrival;
    double meanServiceTime;
    double maxInterArrival;
    double maxServiceTime;
    double varianceInterArrival;
    double varianceServiceTime;
    double arrivalRate;
    double throughput;
    double serviceRate;
    double utilization;
    double completitions;
    double busyTime;
    double observationPeriod;
    std::string &ToString()
    {
        std::stringstream stream{};
        stream << "Mean interval: " << meanInterarrival << "ms" << std::endl
               << "Mean service time: " << meanServiceTime << "ms" << std::endl
               << "Max interval: " << maxInterArrival << "ms" << std::endl
               << "Max service time: " << maxServiceTime << "ms" << std::endl
               << "variace service time: " << varianceServiceTime << "ms" << std::endl
               << "variance inter arrival time: " << varianceInterArrival << "ms" << std::endl
               << "Arrival rate: " << arrivalRate << "job/ms" << std::endl
               << "Throughput: " << throughput << "job/ms" << std::endl
               << "Service rate: " << serviceRate << "job/ms" << std::endl
               << "Utilization: " << utilization << std::endl
               << "Completitions: " << (size_t)completitions << "jobs" << std::endl
               << "Busy Time: " << busyTime << "ms" << std::endl
               << "Observation time: " << observationPeriod << "ms" << std::endl;
        return *new std::string(stream.str());
    }
};

class EventProcessor
{
  private:
    double _clock = 0.0;
    double _busyTime = 0.0;
    size_t _completions = 0;
    Measure<> _interArrivals;
    Measure<> _serviceTimes;

  public:
    virtual Statistics &ComputeStatistics();
    virtual void AddEvent(Event &e);
    EventProcessor(std::vector<Event> &events);
    EventProcessor()
    {
    }
};

namespace boost::serialization
{
template <class Archive> void serialize(Archive &ar, Statistics &stat, const unsigned int version)
{
    ar &stat.meanInterarrival;
    ar &stat.meanServiceTime;
    ar &stat.maxInterArrival;
    ar &stat.maxServiceTime;
    ar &stat.varianceInterArrival;
    ar &stat.varianceServiceTime;
    ar &stat.arrivalRate;
    ar &stat.throughput;
    ar &stat.serviceRate;
    ar &stat.utilization;
    ar &stat.completitions;
    ar &stat.busyTime;
    ar &stat.observationPeriod;
}

} // namespace boost::serialization
