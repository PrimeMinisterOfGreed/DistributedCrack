#pragma once
#include "Event.hpp"
#include "EventHandler.hpp"
class ISimulator
{
  protected:
    int _clientsInSystem = 0;
    double _clock = 0.0;
public:

public:
    virtual void Execute() = 0;
    virtual void Report() = 0;
    int clientsInSystem() const { return _clientsInSystem; }
    void setClientsInSystem(int clientsInSystem) { _clientsInSystem = clientsInSystem; }
};

class ISimulatorEsembler : public ISimulator
{
  public:
    virtual void Reset() = 0;
};

class IScheduler
{
  public:
    virtual void Schedule(Event *event) = 0;
};