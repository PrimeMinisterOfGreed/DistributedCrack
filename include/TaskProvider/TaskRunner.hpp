#pragma once
#include "Concepts.hpp"
#include "EventHandler.hpp"
#include "MultiThread/AutoResetEvent.hpp"
#include "Nodes/Node.hpp"
#include <list>
#include <mutex>
#include <thread>

template <typename Task>
class BaseTaskRunner : public ISignalProvider<IComputeObject<Task>>, public ITaskProvider<IComputeObject<Task>, Task>
{
  private:
    std::queue<IComputeObject<Task> *> _requests;
    AutoResetEvent _onTaskRequested{false};
    std::mutex _queueLock;
    std::vector<IComputeObject<Task> *> _dependentNodes;

  protected:
    auto &PullRequestFromQueue()
    {
        _queueLock.lock();
        if (_requests.size() == 0)
        {
            _onTaskRequested.Reset();
            _queueLock.unlock();
            _onTaskRequested.WaitOne();
            _queueLock.lock();
        }
        auto &request = _requests.front();
        _requests.pop();
        _queueLock.unlock();
        return request;
    }

    BaseTaskRunner()
    {
    }

  public:
    virtual void RequestTask(IComputeObject<Task> &node) override
    {
        _queueLock.lock();
        _requests.push(&node);
        _onTaskRequested.Set();
        _queueLock.unlock();
    }
    virtual void CheckResult(Task &task) = 0;

    virtual void RegisterNode(IComputeObject<Task> &node) override
    {
        _dependentNodes.push_back(&node);
    }

    virtual void StopNodes()
    {
        for (auto node : _dependentNodes)
            node->Abort();
    }

    ~BaseTaskRunner()
    {
        while (_requests.size() > 0)
            _requests.pop();
        _dependentNodes.clear();
    }
};

template <typename Task, typename Generator, typename Predicate = std::function<bool(Task &)>>
    requires Callable<Predicate, bool, Task &> && TaskGenerator<Task, Generator>
class TaskRunner : public BaseTaskRunner<Task>
{
  protected:
    Generator _generator;
    Predicate _stopCondition;
    AutoResetEvent _onAbortRequested{false};
    bool _end = false;

  public:
    EventHandler<Task &> OnResultAcquired;

    TaskRunner(Generator generator, Predicate stopCondition) : _generator(generator), _stopCondition(stopCondition)
    {
    }

    void Execute()
    {
        auto executeThread = std::thread{[this]() {
            while (!_end)
            {
                auto &request = BaseTaskRunner<Task>::PullRequestFromQueue();
                auto &task = _generator();
                request->Enqueue(task);
            }
        }};
        executeThread.detach();
    }

    inline void Abort()
    {
        _onAbortRequested.Set();
    }
    virtual void CheckResult(Task &task) override
    {
        if (_stopCondition(task))
        {
            _end = true;
            BaseTaskRunner<Task>::StopNodes();
            OnResultAcquired.Invoke(task);
            Abort();
        }
    }
};