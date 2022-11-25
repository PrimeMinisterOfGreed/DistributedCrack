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
	catch (const std::exception& ex)
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
	catch (const std::exception& e)
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
	catch (const std::exception& e)
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
	catch (const std::exception& e)
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
	bool comp = false;
	auto ev = _stopWatch.RecordEvent([&](Event& e)
		{
			size_t completions = 0;
	for (auto string : chunk)
	{
		completions++;
		if (md5(string) == _target)
		{
			_logger->TraceInformation() << "Founded password: " << string << std::endl;
			*result = string;
			comp = true;
			e.completitions = completions;
			break;
		}
	}
	e.completitions = completions;
		});
	_processor.AddEvent(ev);
	return comp;
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

