#pragma once
#include "DataProvider.hpp"
#include "Event.hpp"
#include "LinkedList.hpp"
#include "LogEngine.hpp"
#include "Station.hpp"
#include "ISimulator.hpp"
#include "ToString.hpp"
#include <queue>
#include <vector>




class FCFSStation : public Station
{
  protected:
    EventList& _eventQueue= *new EventList();
    Event* _eventUnderProcess;
    IScheduler *_scheduler;
  public:
    void ProcessArrival(Event *evt) override;
    void ProcessDeparture(Event *evt) override;
    void ProcessEnd(Event *evt) override;
    void ProcessProbe(Event *evt) override;
    void Reset() override;
    FCFSStation(ILogEngine * logger, IScheduler * scheduler, int stationIndex);
};


