#include "TaskProvider/TaskGenerator.hpp"
#include "TaskProvider/HashTask.hpp"

HashTask &HashTaskGenerator::operator()()
{
    auto &task = *new HashTask{};
    task._boundaries[0] = _currentAddress;
    task._boundaries[1] = _currentAddress + _chunkSize;
    task.target = std::string(_targetHash);
    _currentAddress = task._boundaries[1];
    task._startSequence = _initialSequenceLength;
    return task;
}
