#include "Nodes/Node.hpp"
#include "Functions.hpp"
#include "md5.hpp"
#include <thread>
#include <future>
void Node::Execute()
{
	try
	{
		Initialize();
		BeginRoutine();
		Routine();
		EndRoutine();
	}
	catch (std::exception ex)
	{
		_logger->TraceException() << ex.what() << std::endl;
		_logger->Finalize();
	}
}



void Node::BeginRoutine()
{
	_logger->TraceInformation() << "Routine Setup" << std::endl;
	try
	{
		OnBeginRoutine();
	}
	catch (std::exception e)
	{
		_logger->TraceException() << "Exception during routine setup:" << e.what() << std::endl;
		throw;
	}
	_logger->TraceInformation() << "Routine Setup completed" << std::endl;
}

void Node::EndRoutine()
{
	_logger->TraceInformation() << "Ending Routine" << std::endl;
	try
	{
		OnEndRoutine();
	}
	catch (std::exception e)
	{
		_logger->TraceException() << "Exception during routine ending: " << e.what() << std::endl;
		throw;
	}
	_logger->TraceInformation() << "Routine end done" << std::endl;
}

void Node::ExecuteRoutine()
{
	_logger->TraceInformation() << "Routine Execution" << std::endl;
	try
	{
		Routine();
	}
	catch (std::exception e)
	{
		_logger->TraceException() << "Exception during routine execution: " << e.what() << std::endl;
		throw;
	}
	_logger->TraceInformation() << "Routine execution done" << std::endl;
}

void MPINode::DeleteRequest(boost::mpi::request* request)
{
	_requests.erase(_requests.begin() +
		indexOf<boost::mpi::request>(_requests.begin(), _requests.end(),
			[&](boost::mpi::request val) -> bool { return &val == request; }));
}

bool MPINode::Compute(const std::vector<std::string>& chunk, std::string* result)
{
	for (auto string : chunk)
	{
		if (md5(string) == _target)
		{
			*result = string;
			return true;
		}
	}
	return false;
}

std::future<bool> MPINode::ComputeAsync(const std::vector<std::string>& chunk, std::function<void(std::string)> callback)
{
	return std::async([&]()->bool
		{
			std::string result = "";
			if (Compute(chunk, &result))
			{
				callback(result);
				return true;
			}
			else
			{
				callback("NULL");
				return false;
			}
		});
}
