#pragma once
#include "LogEngine.hpp"
#include <future>
class INode
{
public:
	virtual void Execute() = 0;
};


class Node: public INode
{
private:
	void BeginRoutine();
	void EndRoutine();
	void ExecuteRoutine();
protected:
	MPILogEngine* _logger = MPILogEngine::Instance();
public:
	virtual void Routine() = 0;
	virtual void Initialize() = 0;
	virtual void OnBeginRoutine() = 0;
	virtual void OnEndRoutine() = 0;
	virtual void Execute() override;
};

class MPINode : public Node
{
protected:
	virtual void DeleteRequest(boost::mpi::request * request);
	std::vector<boost::mpi::request>& _requests = *new std::vector<boost::mpi::request>{};
	boost::mpi::communicator _communicator;
	virtual bool Compute(const std::vector<std::string>& chunk, std::string* result);
	virtual std::future<bool> ComputeAsync(const std::vector<std::string>& chunk, std::function<void(std::string)> callback);
	std::string _target;
public:
	MPINode(boost::mpi::communicator comm, std::string target = "NOTARGET") :_communicator{comm}, _target{target} {};
};