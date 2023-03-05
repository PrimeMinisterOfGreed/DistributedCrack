#pragma once
#include "Compute.hpp"
#include "Concepts.hpp"
#include "DataContainer.hpp"
#include "EventHandler.hpp"
#include "LogEngine.hpp"
#include "MultiThread/AutoResetEvent.hpp"
#include "Statistics/EventProcessor.hpp"
#include "TaskProvider/TaskProvider.hpp"
#include "md5.hpp"
#include <future>
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

class ComputeNode : public Node
{
  private:
    AutoResetEvent _taskReceived{false};
    AutoResetEvent _onAbortRequested{false};
    ITaskProvider &_taskProvider;
  protected:
    DataContainer *_container = new DataContainer();
    EventProcessor &_processor = *new EventProcessor();
    StopWatch &_stopWatch = *new StopWatch();
    void Routine() override;
    void AddResult(Statistics &statistic, int process, std::string method);
    void OnEndRoutine() override;
    virtual void WaitTask();
  public:
    Statistics &GetNodeStats() const;
    ComputeNode(ITaskProvider &provider, ILogEngine *logEngine) : Node(logEngine), _taskProvider(provider)
    {
        
    }
};

class NodeHasher : public Node
{
  protected:
    std::string _target;
    IHashComparer &_computeFnc;

  public:
    NodeHasher(std::string target, ILogEngine *logger, IHashComparer &computeFnc)
        : _target(target), Node(logger), _computeFnc(computeFnc)
    {
    }
};

class MPINode : public NodeHasher
{
  protected:
    virtual void DeleteRequest(boost::mpi::request &request);
    std::vector<boost::mpi::request> &_requests = *new std::vector<boost::mpi::request>{};
    boost::mpi::communicator _communicator;

  public:
    MPINode(boost::mpi::communicator comm, std::string target, IHashComparer &computeFnc)
        : _communicator{comm}, NodeHasher(target, MPILogEngine::Instance(), computeFnc){

                               };
};
