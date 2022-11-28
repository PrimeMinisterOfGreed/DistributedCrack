#include "Statistics/EventProcessor.hpp"

Statistics& EventProcessor::ComputeStatistics()
{
	double clock = max(_arrivalTimes);
	double lastService = 0;
	Statistics& result = _actual;
	for (auto& ev : _events)
	{
		_arrivalTimes(ev.arrivalTime);
		_serviceTimes(ev.serviceTime);
		_interArrivals(ev.arrivalTime - clock);
		result.busyTime += ev.serviceTime;
		clock = ev.arrivalTime;
		result.completitions += ev.completitions;
		lastService = ev.serviceTime;
	}
	result.meanInterarrival = mean(_interArrivals);
	result.meanServiceTime = mean(_serviceTimes);
	result.maxInterArrival = max(_interArrivals);
	result.maxServiceTime = max(_serviceTimes);
	result.observationPeriod = clock + lastService;
	result.arrivalRate = result.completitions / result.observationPeriod;
	result.serviceRate = result.completitions / result.busyTime;
	result.utilization = result.busyTime / result.observationPeriod;
	result.throughput = result.completitions / result.observationPeriod;
	_events.clear();
	return result;
}