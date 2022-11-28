#pragma once
#include "LogEngine.hpp"
#include <future>
#include "md5.hpp"
#include "Statistics/EventProcessor.hpp"
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
	ILogEngine* _logger;
	virtual void Routine() = 0;
	virtual void Initialize() = 0;
	virtual void OnBeginRoutine() = 0;
	virtual void OnEndRoutine() = 0;
public:

	virtual void Execute() override;
};

class MPINode : public Node
{
protected:
	EventProcessor& _processor = *new EventProcessor();
	StopWatch& _stopWatch = *new StopWatch();
	virtual void DeleteRequest(boost::mpi::request& request);
	std::vector<boost::mpi::request>& _requests = *new std::vector<boost::mpi::request>{};
	boost::mpi::communicator _communicator;
	virtual bool Compute(const std::vector<std::string>& chunk, std::string* result, std::function<std::string(std::string)> hashFnc = md5);
	virtual std::future<bool> ComputeAsync(const std::vector<std::string>& chunk, std::function<void(std::string)> callback);
	std::string _target = *new std::string;
public:
	MPINode(boost::mpi::communicator comm, std::string target = "NOTARGET") :_communicator{ comm }, _target{ target } { _logger = MPILogEngine::Instance(); };
};
