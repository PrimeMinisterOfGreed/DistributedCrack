#include "DataProvider.hpp"
#include "Generators.hpp"
#include "rngs.hpp"
#include <stdexcept>

enum
{
    ARRIVAL,
    INTERARRIVAL,
    SERVICE
};

double GetValue(std::ifstream &stream)
{
    float result = -1;
    stream >> result;
    return result;
}

double BaseDataProvider::GetArrival()
{
    if (_dataUsed[ARRIVAL])
        Next();
    _dataUsed[ARRIVAL] = true;
    return _data[ARRIVAL];
}

double BaseDataProvider::GetService()
{
    if (_dataUsed[SERVICE])
        Next();
    _dataUsed[SERVICE] = true;
    return _data[SERVICE];
}

double BaseDataProvider::GetInterArrival()
{
    if (_dataUsed[INTERARRIVAL])
        Next();
    _dataUsed[INTERARRIVAL] = true;
    return _data[INTERARRIVAL];
}

void BaseDataProvider::Reset()
{
    _data[0] = 0.0;
    _data[1] = 0.0;
    _data[2] = 0.0;
    _dataUsed[0] = false;
    _dataUsed[1] = false;
    _dataUsed[2] = false;
}

BaseDataProvider::BaseDataProvider()
{
}

void TraceDrivenDataProvider::Next()
{
    if ((!_dataUsed[ARRIVAL] || !_dataUsed[SERVICE]) && _init)
        throw std::invalid_argument("Trying to acquire new data when the old one is not used");
    _dataUsed[ARRIVAL] = false;
    _dataUsed[SERVICE] = false;
    _dataUsed[INTERARRIVAL] = false;
    if (_interArrivalMode)
    {

        auto interarrival = GetValue(_filestream);
        _data[ARRIVAL] += interarrival;
        _data[INTERARRIVAL] = interarrival;
    }
    else
    {
        auto arrival = GetValue(_filestream);
        _data[INTERARRIVAL] = arrival - _data[ARRIVAL];
        _data[ARRIVAL] = arrival;
    }
    _data[SERVICE] = GetValue(_filestream);
    if (_data[SERVICE] < 0)
        _end = true;
}

TraceDrivenDataProvider::TraceDrivenDataProvider(std::string &filePath, bool interArrivalMode)
    : _filestream{*new std::ifstream{}}, _interArrivalMode{interArrivalMode}
{
    _filestream.open(filePath);
    Next();
    _init = true;
}

inline bool DoubleStreamNegExpRandomDataProvider::end()
{
    return _data[ARRIVAL] >= _endTime;
}

void DoubleStreamNegExpRandomDataProvider::Next()
{
    if ((!_dataUsed[ARRIVAL] || !_dataUsed[SERVICE]) && _inited)
        throw std::invalid_argument("Trying to acquire new data when the old one is not used");
    _dataUsed[ARRIVAL] = false;
    _dataUsed[SERVICE] = false;
    _dataUsed[INTERARRIVAL] = false;
    SelectStream(0);
    double interarrival = NegExp(_interArrivalLambda);
    SelectStream(1);
    double serviceTime = NegExp(_serviceLambda);

    if (_data[INTERARRIVAL] != 0)
    {
        _data[ARRIVAL] += interarrival;
    }
    _data[INTERARRIVAL] = interarrival;
    _data[SERVICE] = serviceTime;
}

void DoubleStreamNegExpRandomDataProvider::Reset()
{
    _inited = false;
    BaseDataProvider::Reset();
    Next();
    _inited = true;
}

DoubleStreamNegExpRandomDataProvider::DoubleStreamNegExpRandomDataProvider(double endTime, double interArrivalLambda,
                                                                           double serviceLambda)
    : _endTime(endTime), _interArrivalLambda(interArrivalLambda), _serviceLambda(serviceLambda)
{
    Next();
    _inited = true;
}

