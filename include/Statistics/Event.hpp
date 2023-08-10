#pragma once
#include "Statistics/TimeMachine.hpp"
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics.hpp>
#include <chrono>

struct Event
{
    double arrivalTime;
    double serviceTime;
    double completitions = 1.0;
    Event(double arrivalTime, double serviceTime, double completitions)
        : arrivalTime{arrivalTime}, serviceTime{serviceTime}, completitions{completitions}
    {
    }
    Event()
    {
    }

    Event(Event &) = delete;
    
};
