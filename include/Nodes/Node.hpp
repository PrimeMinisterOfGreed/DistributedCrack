#pragma once
#include "DataContainer.hpp"
#include "LogEngine.hpp"
#include "Statistics/EventProcessor.hpp"
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

  protected:
    ILogEngine *_logger;
    DataContainer *_container = new DataContainer();
    virtual void Routine() = 0;
    virtual void Initialize() = 0;
    virtual void OnBeginRoutine() = 0;
    virtual void OnEndRoutine() = 0;
    void AddResult(Statistics &statistic, int process, std::string method);

  public:
    virtual void Execute() override;
};

class NodeHasher : public Node
{
  protected:
    EventProcessor &_processor = *new EventProcessor();
    StopWatch &_stopWatch = *new StopWatch();
    std::string _target;
    virtual bool Compute(const std::vector<std::string> &chunk, std::string *result,
                         std::function<std::string(std::string)> hashFnc = md5);
    virtual std::future<bool> ComputeAsync(const std::vector<std::string> &chunk,
                                           std::function<void(std::string)> callback);
    public:
    NodeHasher(std::string target) : _target(target)
    {
    }
    Statistics& GetNodeStats() const;
};

class MPINode : public NodeHasher
{
  protected:
    virtual void DeleteRequest(boost::mpi::request &request);
    std::vector<boost::mpi::request> &_requests = *new std::vector<boost::mpi::request>{};
    boost::mpi::communicator _communicator;

  public:
    MPINode(boost::mpi::communicator comm, std::string target = "NOTARGET") : _communicator{comm}, NodeHasher(target)
    {
        _logger = MPILogEngine::Instance();
    };
};
