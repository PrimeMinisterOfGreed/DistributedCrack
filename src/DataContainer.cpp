#include "DataContainer.hpp"
#include "LogEngine.hpp"
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <ios>
#include <sstream>
#include <string>
#include <vector>

#define _PARAM(param) param << ";"

std::vector<std::string> split(std::string str, char delim)
{
    using namespace std;
    std::string token;
    stringstream stream(str);
    vector<string> &result = *new vector<string>();
    while (getline(stream, token, delim))
    {
        result.push_back(token);
    }
    return result;
}

size_t getLastExecution(const char *filePath)
{
    using namespace std;
    fstream file = fstream{filePath, ios_base::in};
    char *buffer = new char[1024];
    std::string toSplit;
    while (!file.eof())
    {
        file.getline(buffer, 1024);
        if (!file.eof())
            toSplit = string(buffer);
    }

    auto splitted = split(toSplit, ';');
    return atol(splitted.at(0).c_str());
}

std::string getCsvLine(int execNum, ExecutionResult &executionResult)
{
    std::stringstream str{};
    auto &e = executionResult;
    auto &stat = e.statistics;
    str << _PARAM(execNum) << _PARAM(e.method) << _PARAM(e.process) << _PARAM(stat.meanInterarrival)
        << _PARAM(stat.meanServiceTime) << _PARAM(stat.maxInterArrival) << _PARAM(stat.maxServiceTime)
        << _PARAM(stat.arrivalRate) << _PARAM(stat.throughput) << _PARAM(stat.serviceRate) << _PARAM(stat.utilization)
        << _PARAM(stat.completitions) << _PARAM(stat.busyTime) << stat.observationPeriod << std::endl;
    return str.str();
}

void DataContainer::AddResult(ExecutionResult &executionResult)
{
    _result.push_back(executionResult);
}

void DataContainer::SaveToFile(const char *filePath)
{
    using namespace std;
    fstream file;
    if (!filesystem::exists(filePath))
    {
        file = fstream(filePath, ios_base::out);
        std::string header = "execution(#);process(#);method;mean_interArrival(ms);mean_serviceTime(ms);max_"
                             "interArrival(ms);max_serviceTime(ms);arrivalRate(ms);throughput;serviceRate;utilization;"
                             "completions;busyTime(ms);observationPeriod(ms)\n";
        file.write(header.c_str(), header.size());
        _lastExecution = 0;
    }
    else
    {
        _lastExecution = getLastExecution(filePath);
        file = fstream(filePath, ios_base::app);
    }
    for (auto &exec : _result)
    {
        _lastExecution++;
        auto csvLine = getCsvLine(_lastExecution, exec);
        file.write(csvLine.c_str(), csvLine.size());
    }
}
