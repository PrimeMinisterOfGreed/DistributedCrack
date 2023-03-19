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
    virtual void AddResult(Statistics &statistic, int process, std::string method);
    virtual void OnEndRoutine() override;
    virtual void OnBeginRoutine() override;
    virtual void Initialize() override;
    virtual void WaitTask();

  public:
    Statistics &GetNodeStats() const;
    virtual void Abort();
};

template <typename Task, typename Provider = ITaskProvider<IComputeObject<Task>, Task>,
          typename Signaler = ISignalProvider<IComputeObject<Task>>>
    requires TaskProvider<IComputeObject<Task>, Task, Provider> && NodeSignaler<IComputeObject<Task>, Signaler>
    class ComputeNode : public BaseComputeNode, public IComputeObject<Task>
{
  protected:
    std::queue<Task> _taskList{};
    Task *_taskUnderProcess;
    Provider &_provider;

    void Abort() override
    {
        BaseComputeNode::Abort();
    }
    virtual void Enqueue(Task &task) override
    {
        _taskList.push(task);
        FireTaskReceived(task);
    }

    virtual void FireTaskReceived(Task &task)
    {
        _logger->TraceInformation("Arrived task");
        _taskReceived.Set();
    }

    virtual void Routine() override
    {
        _provider.RequestTask(*this);
        WaitTask();
        while (_taskList.size() > 0 && !ShouldEnd())
        {
            _taskUnderProcess = &_taskList.front();
            _taskList.pop();
            ProcessTask(*_taskUnderProcess);
            _taskUnderProcess = nullptr;
        }
    }

    virtual void ProcessTask(Task &task) = 0;

  public:
    EventHandler<Task &> OnTaskCompleted;
    ComputeNode(ILogEngine *logEngine, Provider &provider,
                Signaler& signaler)
        : BaseComputeNode(logEngine), _provider(provider)
    {
        signaler.RegisterNode(*this);
    }
};

template <ComputeFunction Hasher, typename Provider = ITaskProvider<IComputeObject<HashTask>, HashTask>,
          typename Signaler = ISignalProvider<IComputeObject<HashTask>>>

class HashNode : public ComputeNode<HashTask,Provider,Signaler>
{
  protected:
    Hasher _functor;
    virtual void ProcessTask(HashTask &task) override
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

  public:
    HashNode(Hasher functor, Provider &provider, Signaler& signaler ,ILogEngine *logger)
        : _functor(functor), ComputeNode<HashTask,Provider,Signaler>(logger, provider,signaler)
    {
    }
};
