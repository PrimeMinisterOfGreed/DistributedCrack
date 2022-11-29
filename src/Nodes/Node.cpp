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
		_logger->TraceException(ex.what());
		_logger->Finalize();
	}
}



void Node::BeginRoutine()
{
	_logger->TraceInformation("Routine Setup");
	try
	{
		OnBeginRoutine();
	}
	catch (const std::exception& e)
	{
		_logger->TraceException("Exception during routine setup:{0}", e.what());
		throw;
	}
	_logger->TraceInformation("Routine Setup completed");
}

void Node::EndRoutine()
{
	_logger->TraceInformation("Ending Routine");
	try
	{
		OnEndRoutine();
	}
	catch (const std::exception& e)
	{
		_logger->TraceException("Exception during routine ending:{0}", e.what());
		throw;
	}
	_logger->TraceInformation("Routine end done");
}

void Node::ExecuteRoutine()
{
	_logger->TraceInformation("Routine Execution");
	try
	{
		Routine();
	}
	catch (const std::exception& e)
	{
		_logger->TraceException("Exception during routine execution:{0} ",e.what());
		throw;
	}
	_logger->TraceInformation("Routine execution done");
}


void MPINode::DeleteRequest(boost::mpi::request& request)
{
	int index = indexOf<boost::mpi::request>(_requests.begin(), _requests.end(),
		[&](boost::mpi::request val) -> bool { return &val == &request; });
	if (index == -1)
		throw std::invalid_argument("Index of request is not existent");
	_requests.erase(_requests.begin() +
		index);
}

bool MPINode::Compute(const std::vector<std::string>& chunk, std::string* result, std::function<std::string(std::string)> hashFnc)
{
	bool comp = false;
	auto ev = _stopWatch.RecordEvent([&](Event& e)
		{
			size_t completions = 0;
	for (auto& string : chunk)
	{
		completions++;
		if (hashFnc(string) == _target)
		{
			_logger->TraceInformation("Founded password: ",string);
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



