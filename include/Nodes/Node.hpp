#pragma once
#include "Compute.hpp"
#include "Concepts.hpp"
#include "DataContainer.hpp"
#include "EventHandler.hpp"
#include "LogEngine.hpp"
#include "MultiThread/AutoResetEvent.hpp"
#include "Statistics/EventProcessor.hpp"
#include "StringGenerator.hpp"
#include "TaskProvider/HashTask.hpp"
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
    bool ShouldEnd() const
    {
        return _end;
    }
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

template <typename Task, TaskProvider<Task> Provider, NodeSignaler Signaler> class ComputeNode : public BaseComputeNode
{
  protected:
    std::queue<Task> _taskList{};
    Task *_taskUnderProcess;
    Provider _provider;
    void Enqueue(Task &task);
    virtual void FireTaskReceived(Task &task);
    virtual void Routine() override;
    virtual void ProcessTask(Task &task);
  public:
    EventHandler<Task&> OnTaskCompleted;
    ComputeNode(ILogEngine *logEngine, Provider& provider, Signaler signaler) : BaseComputeNode(logEngine), _provider(provider)
    {
        signaler.RegisterNode(this);
    }
};

template <typename Task, TaskProvider<Task> Provider, NodeSignaler Signaler> inline void ComputeNode<Task,Provider,Signaler>::Enqueue(Task &task)
{
    _taskList.push(task);
    FireTaskReceived(task);
}

template <typename Task, TaskProvider<Task> Provider, NodeSignaler Signaler> inline void ComputeNode<Task,Provider,Signaler>::FireTaskReceived(Task &task)
{
    _logger->TraceInformation("Arrived task");
    _taskReceived.Set();
}

template <typename Task, TaskProvider<Task> Provider, NodeSignaler Signaler> inline void ComputeNode<Task,Provider,Signaler>::Routine()
{
    WaitTask();
    while (_taskList.size() > 0 && !ShouldEnd())
    {
        _taskUnderProcess = _taskList.front();
        _taskList.pop();
        ProcessTask(_taskUnderProcess);
        _taskUnderProcess = nullptr;
    }
    _provider.RequestTask(this);
}

template <typename Hash, ComputeFunction HashFunction, TaskProvider<HashTask> Provider, NodeSignaler Signaler> class HashNode : public ComputeNode<HashTask,Provider,Signaler>
{
  protected:
    HashFunction _functor;
  public:
    HashNode(HashFunction functor, Hash hash, Provider & provider ,ILogEngine* logger) : _functor(HashFunction(hash)), ComputeNode<HashTask, Provider,Signaler>(logger,provider)
    {
    }
    void ProcessTask(HashTask &task) override;
};

template <typename Hash, ComputeFunction HashFunction, TaskProvider<HashTask> Provider, NodeSignaler Signaler>
inline void HashNode<Hash, HashFunction,Provider,Signaler>::ProcessTask(HashTask &task)
{
    auto res = BaseComputeNode::_stopWatch.RecordEvent([this, task](Event &ev) {
        AssignedSequenceGenerator generator{task._startSequence};
        auto chunkSize = task._boundaries[1] - task._boundaries[0];
        generator.AssignAddress(task._boundaries[0]);
        _functor(generator.generateChunk(chunkSize), task.target, task.result);
        ev.completitions = chunkSize;
    });
    BaseComputeNode::_processor.AddEvent(res);
    ComputeNode<HashTask,Provider,Signaler>::OnTaskCompleted.Invoke(task);
}

