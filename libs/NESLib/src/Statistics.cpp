#include "Statistics.hpp"
#include <boost/math/distributions.hpp>

void StatisticCollector::Accumulate(const StationStatistic &stat)
{
    _samples++;
    _avgInterArrival(stat.avgInterArrival);
    _avgServiceTime(stat.avgServiceTime);
    _avgDelay(stat.avgDelay);
    _avgWaiting(stat.avgWaiting);
    _utilization(stat.utilization);
    _throughput(stat.throughput);
    _inputRate(stat.inputRate);
    _arrivalRate(stat.arrivalRate);
    _serviceRate(stat.serviceRate);
    _traffic(stat.traffic);
    _meanCustomInQueue(stat.meanCustomInQueue);
    _meanCustomerInService(stat.meanCustomerInService);
    _meanCustomerInSystem(stat.meanCustomerInSystem);
}

int StatisticCollector::getSamples() const
{
    return _samples;
}

std::map<std::string, Measure<double>> &StatisticCollector::GetAccumulators()
{
    auto &map = *new std::map<std::string, Measure<double>>();
    map["avgInterArrival"] = _avgInterArrival;
    map["avgServiceTime"] = _avgServiceTime;
    map["avgDelay"] = _avgDelay;
    map["avgWaiting"] = _avgWaiting;
    map["utilization"] = _utilization;
    map["throughput"] = _throughput;
    map["inputRate"] = _inputRate;
    map["arrivalRate"] = _arrivalRate;
    map["serviceRate"] = _serviceRate;
    map["traffic"] = _traffic;
    map["meanCustomInQueue"] = _meanCustomInQueue;
    map["meanCustomerInService"] = _meanCustomerInService;
    map["meanCustomerInSystem"] = _meanCustomerInSystem;
    return map;
}

Interval GetValueConfidence(int samples, double confidence, Measure<double> acc)
{

    double u = acc.mean(0);
    double sigma = acc.variance();
    double alpha = 1 - 0.95;
    double delta =
            idfStudent(samples - 1, 1 - (alpha / 2)) * (sigma / sqrt(samples - 1));
    return {u - delta, u + delta};
}

std::map<std::string, Interval> &StatisticCollector::GetConfidenceIntervals(double confidence)
{
    auto &map = *new std::map<std::string, Interval>();

    map["avgInterArrival"] = GetValueConfidence(_samples, confidence, _avgServiceTime);
    map["avgServiceTime"] = GetValueConfidence(_samples, confidence, _avgServiceTime);
    map["avgDelay"] = GetValueConfidence(_samples, confidence, _avgDelay);
    map["avgWaiting"] = GetValueConfidence(_samples, confidence, _avgWaiting);
    map["utilization"] = GetValueConfidence(_samples, confidence, _utilization);
    map["throughput"] = GetValueConfidence(_samples, confidence, _throughput);
    map["inputRate"] = GetValueConfidence(_samples, confidence, _inputRate);
    map["arrivalRate"] = GetValueConfidence(_samples, confidence, _arrivalRate);
    map["serviceRate"] = GetValueConfidence(_samples, confidence, _serviceRate);
    map["traffic"] = GetValueConfidence(_samples, confidence, _traffic);
    map["meanCustomInQueue"] = GetValueConfidence(_samples, confidence, _meanCustomInQueue);
    map["meanCustomerInService"] = GetValueConfidence(_samples, confidence, _meanCustomerInService);
    map["meanCustomerInSystem"] = GetValueConfidence(_samples, confidence, _meanCustomerInSystem);

    return map;
}

std::string StatisticCollector::ToString()
{

    auto &valueMap = GetAccumulators();
    auto &intervalMap = GetConfidenceIntervals(0.95);
    std::string result = "";
    result += "|Value|Mean|Variance|IntervalLowerBound|IntervalMaxBound|\n";
    result += "|---|---|---|---|---|\n";
    for (auto &value: valueMap)
    {
        auto interval = intervalMap[value.first];
        result += makeformat("|{}|{}|{}|{}|{}|\n", value.first, value.second.mean(), value.second.variance(),
                             interval.first, interval.second);
    }
    return result;
}

double idfStudent(double df, double quantile)
{
    boost::math::students_t stud{df};
    return
            boost::math::quantile(stud, quantile
            );
}


void BaseVariableMonitor::AddRandomVar(std::string name, BaseRandomVariable *variable)
{
    _register[name] = variable;
    _handlers[name] = new FunctionHandler<double>{[this,name](double value){
        this->Collect(name,value);
    }};
    _register[name]->OnVariableGenerated += _handlers[name];
}

void BaseVariableMonitor::RemoveRandomVar(std::string name)
{
    _register[name]->OnVariableGenerated -= _handlers[name];
    _handlers.erase(name);
    _register.erase(name);
}


