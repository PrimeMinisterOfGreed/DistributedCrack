#include "FCFSStation.hpp"
#include "Event.hpp"
#include "ISimulator.hpp"
#include "LogEngine.hpp"
#include "Station.hpp"
#include "ToString.hpp"

void FCFSStation::ProcessArrival(Event *evt)
{
    Station::ProcessArrival(evt);
    if (_eventUnderProcess == nullptr)
    {
        evt->CreateTime = _clock;
        evt->ArrivalTime = _clock;
        evt->OccurTime = _clock + evt->ServiceTime;
        evt->Type = EventType::DEPARTURE;
        _eventUnderProcess = evt;
        _scheduler->Schedule(evt);
    }
    else
    {
        _eventQueue.Enqueue(evt);
    }
}

void FCFSStation::ProcessDeparture(Event *evt)
{
    Station::ProcessDeparture(evt);
    if (evt != _eventUnderProcess)
        throw std::runtime_error("The departure requested is not equal as the event under process");
    if (_sysClients > 0)
    {
        _eventUnderProcess = &_eventQueue.Dequeue();
        _eventUnderProcess->ArrivalTime = _clock;
        _eventUnderProcess->CreateTime = _clock;
        _eventUnderProcess->OccurTime = _clock + _eventUnderProcess->ServiceTime;
        _eventUnderProcess->Type = EventType::DEPARTURE;
        _scheduler->Schedule(_eventUnderProcess);
    }
    else
        _eventUnderProcess = nullptr;
    _completions++;
}

void FCFSStation::ProcessEnd(Event *evt)
{
    Station::ProcessEnd(evt);
}

void FCFSStation::ProcessProbe(Event *evt)
{
    Station::ProcessProbe(evt);
}


void FCFSStation::Reset()
{
    Station::Reset();
    _eventQueue.Clear();
    _eventUnderProcess = nullptr;
}

FCFSStation::FCFSStation(ILogEngine *logger, IScheduler *scheduler, int stationIndex)
    : Station(logger, stationIndex), _scheduler(scheduler)
{
}
