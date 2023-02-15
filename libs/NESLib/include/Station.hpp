#pragma once
#include "Event.hpp"
#include "LogEngine.hpp"
#include "DataProvider.hpp"



struct StationStatistic
{
    double avgInterArrival;
    double avgServiceTime;
    double avgDelay;
    double avgWaiting;
    double utilization;
    double throughput;
    double inputRate;
    double arrivalRate;
    double serviceRate;
    double traffic;
    double meanCustomInQueue;
    double meanCustomerInService;
    double meanCustomerInSystem;
    std::string &ToString() const;
};



class Station
{
  protected:
    std::string _name;
    int _stationIndex = 0;
    int _arrivals;
    int _completions;
    int _sysClients;
    int _maxClients;
    double _busyTime;
    double _observationPeriod;
    double _lastArrival;
    double _areaN;
    double _areaS;
    double _oldclock = 0.0;
    double _clock = 0.0;
    virtual void ProcessArrival(Event *evt);
    virtual void ProcessDeparture(Event *evt);
    virtual void ProcessEnd(Event *evt);
    virtual void ProcessProbe(Event *evt);
    virtual void ProcessMaintenance(Event *evt);
    ILogEngine *_logger;
  public:
    void Process(Event *event);
    virtual void Initialize();
    virtual std::string &ToString();
    StationStatistic GetStatistics();
    virtual void Reset();
    Station(ILogEngine *logger, int station);

    int stationIndex() const { return _stationIndex; }

    int arrivals() const { return _arrivals; }

    int completions() const { return _completions; }

    int sysClients() const { return _sysClients; }
    

};


class DelayStation
{
  protected:
    std::string _name;
    double _thinkTime;
    int _numClients;
    
  public:
    Event& Process();
    DelayStation(double thinkTime, int numClients);
};