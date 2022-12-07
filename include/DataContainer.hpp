#pragma once
#include "Statistics/EventProcessor.hpp"
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

struct ExecutionResult
{
    int process;
    std::string method;
    Statistics statistics;
};

class DataContainer
{
  private:
    std::vector<ExecutionResult> &_result = *new std::vector<ExecutionResult>();
    size_t _lastExecution = 0;

  public:
    void SaveToFile(const char *filePath);
    void AddResult(ExecutionResult &executionResult);
    const std::vector<ExecutionResult> &Results()
    {
        return _result;
    }
    inline bool HasDataToSave()
    {
        return _result.size() > 0;
    }
};