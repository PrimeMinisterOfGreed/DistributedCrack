#pragma once
#include "Compute.hpp"
#include "Concepts.hpp"
#include "DataContainer.hpp"
#include "EventHandler.hpp"
#include "LogEngine.hpp"
#include "MultiThread/AutoResetEvent.hpp"
#include "Statistics/EventProcessor.hpp"
#include "TaskProvider/Tasks.hpp"
#include "md5.hpp"
#include <cstddef>
#include <functional>
#include <future>
#include <queue>
#include <string>

class INode
{
  public:
    virtual void Execute() = 0;
};

class Node : public INode
{
  private:
    void BeginRoutine();
    void EndRoutine();
    void ExecuteRoutine();
    bool _end = false;

  protected:
    ILogEngine *_logger;
    virtual void Routine() = 0;
    virtual void Initialize() = 0;
    virtual void OnBeginRoutine() = 0;
    virtual void OnEndRoutine() = 0;

  public:
    Node(ILogEngine *logger) : _logger(logger)
    {
    }
    virtual void Execute() override;
    void Stop();
};

class BaseComputeNode : public Node
{
  protected:
    BaseComputeNode(ILogEngine *logEngine) : Node(logEngine)
    {
    }
    AutoResetEvent _taskReceived{false};
    AutoResetEvent _onAbortRequested{false};
    DataContainer *_container = new DataContainer();
    EventProcessor &_processor = *new EventProcessor();
    StopWatch &_stopWatch = *new StopWatch();
    void AddResult(Statistics &statistic, int process, std::string method);
    void OnEndRoutine() override;
    virtual void WaitTask();

  public:
    Statistics &GetNodeStats() const;
    virtual void Abort();
};

template <typename Task = ITask> class ComputeNode : public BaseComputeNode
{
  protected:
    std::queue<Task> _taskList{};
    Task *_taskUnderProcess;
    void Enqueue(Task &task);
    virtual void FireTaskReceived(Task &task);
    virtual void Routine() override;
    virtual void ProcessTask(Task &task);

  public:
    ComputeNode(ILogEngine *logEngine) : BaseComputeNode(logEngine)
    {
    }
};

template <typename Task> inline void ComputeNode<Task>::Enqueue(Task &task)
{
    _taskList.push(task);
    FireTaskReceived(task);
}

template <typename Task> inline void ComputeNode<Task>::FireTaskReceived(Task &task)
{
    _logger->TraceInformation("Arrived task");
    _taskReceived.Set();
}

template <typename Task> inline void ComputeNode<Task>::Routine()
{
    WaitTask();
    while (_taskList.size() > 0)
    {
        _taskUnderProcess = _taskList.front();
        _taskList.pop();
        ProcessTask(_taskUnderProcess);
        _taskUnderProcess = nullptr;
    }
}




