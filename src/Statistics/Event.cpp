#include "Statistics/Event.hpp"

void IEventProcessor::ComputeStatistics()
{
	double clock = max(_arrivalTimes);
	for (auto ev : _events)
	{
		_arrivalTimes(ev.arrivalTime);
		_serviceTimes(ev.serviceTime);
		_interArrivals(ev.arrivalTime - clock);
		clock = ev.arrivalTime;
	}
}
