#pragma once
#include "Concepts.hpp"
#include "EventHandler.hpp"
#include "MultiThread/AutoResetEvent.hpp"
#include "Nodes/Node.hpp"
#include <list>
#include <mutex>

template <typename Task> class BaseTaskRunner
{
  private:
    std::queue<ComputeNode<Task, BaseTaskRunner<Task>,BaseTaskRunner<Task>>&> _requests;
    AutoResetEvent _onTaskRequested;
    std::mutex _queueLock;
    std::vector<BaseComputeNode*> _dependentNodes;
  protected:
    virtual ComputeNode<Task, BaseTaskRunner<Task>, BaseTaskRunner<Task>> &PullRequestFromQueue()
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
        _queueLock.unlock();
        return request;
    }

  public:
    virtual void RequestTask(ComputeNode<Task, BaseTaskRunner<Task>,BaseTaskRunner<Task>>& node)
    {
        _queueLock.lock();
        _requests.push_back(node);
        _onTaskRequested.Set();
        _queueLock.unlock();
    }
    virtual void CheckResult(Task &task) = 0;

    void RegisterNode(BaseComputeNode &node)
    {
        _dependentNodes.push_back(&node);
    }

    virtual void StopNodes()
    {
        for (auto node : _dependentNodes)
            node->Abort();
    }

    ~BaseTaskRunner<Task>()
    {
        while (_requests.size() > 0)
            _requests.pop();
        _dependentNodes.clear();
    }
};

template <typename Task, TaskGenerator<Task> Generator, typename Predicate>
    requires Callable<Predicate, void, Task>
class TaskRunner : public BaseTaskRunner<Task>
{
  protected:
    Generator _generator;
    Predicate _stopCondition;
    AutoResetEvent _onAbortRequested;
    bool _end = false;
  public:
    EventHandler<Task &> OnResultAcquired;
    
    
    TaskRunner(Generator generator, Predicate stopCondition) : _generator(generator)
    {
    }

    void Execute();
    inline void Abort()
    {
        _onAbortRequested.Set();
    }
    virtual void CheckTask(Task& task) override;
};

template <typename Task, TaskGenerator<Task> Generator, typename Predicate>
    requires Callable<Predicate, void, Task>
void TaskRunner<Task, Generator, Predicate>::Execute()
{
    while (!_end)
    {
        ComputeNode<Task, BaseTaskRunner<Task>,BaseTaskRunner<Task>> &request = BaseTaskRunner<Task>::PullRequestFromQueue();
        auto& task = _generator();
        request.Enqueue(task);
    }
}

template <typename Task, TaskGenerator<Task> Generator, typename Predicate>
    requires Callable<Predicate, void, Task>
void TaskRunner<Task, Generator, Predicate>::CheckTask(Task &task)
{
    if (_stopCondition(task))
    {
        _end = true;
        BaseTaskRunner<Task>::StopNodes();
        OnResultAcquired.Invoke(task);
    }
}
