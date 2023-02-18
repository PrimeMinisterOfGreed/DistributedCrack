#include "Statistics/EventProcessor.hpp"

Statistics &EventProcessor::ComputeStatistics()
{
    Statistics &result = *new Statistics();
    result.busyTime = _busyTime;
    result.completitions = _completions;
    result.meanInterarrival = _interArrivals.mean();
    result.meanServiceTime = _serviceTimes.mean();
    result.maxInterArrival = _interArrivals.max();
    result.maxServiceTime = _serviceTimes.max();
    result.varianceInterArrival = _interArrivals.variance();
    result.varianceServiceTime = _serviceTimes.variance();
    result.observationPeriod = _clock;
    result.arrivalRate = result.completitions / result.observationPeriod;
    result.serviceRate = result.completitions / result.busyTime;
    result.utilization = result.busyTime / result.observationPeriod;
    result.throughput = result.completitions / result.observationPeriod;
    return result;
}

void EventProcessor::AddEvent(Event &e)
{
    _serviceTimes(e.serviceTime);
    _interArrivals(e.arrivalTime - _clock);
    _clock = e.arrivalTime;
    _busyTime += e.serviceTime;
    _completions += e.completitions;
}

EventProcessor::EventProcessor(std::vector<Event> &events): EventProcessor()
{
    for (auto &evt : events)
    {
        AddEvent(evt);
    }
}
