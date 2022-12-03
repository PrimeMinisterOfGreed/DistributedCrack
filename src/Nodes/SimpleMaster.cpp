#include "Nodes/SimpleMaster.hpp"
#include <boost/mpi/collectives.hpp>
#include <boost/serialization/vector.hpp>
#include "Schema.hpp"
#include "StringGenerator.hpp"
#include "md5.hpp"
#include "Functions.hpp"
#include "OptionsBag.hpp"
using namespace boost::mpi;

static bool operator==(request& r1, request& r2)
{
	return &r1 == &r2;
}


void SimpleMaster::OnEndRoutine()
{
	Statistics collected;

	for (int i = 1; i < _comm.size(); i++)
	{
		_logger->TraceTransfer("Sending terminate to process: {0}", i);
		_comm.send(i, TERMINATE);
		_comm.recv(i, MESSAGE, collected);
		this->_collectedStats.push_back(collected);
	}

}

void SimpleMaster::Initialize()
{
	_logger->TraceInformation("Broadcasting Target:{0}", _target);
	broadcast(_comm, _target, _comm.rank());
}

void SimpleMaster::Routine()
{
	using namespace boost::mpi;
	auto termReq = _comm.irecv(any_source, FOUND, _result);
	_requests.push_back(termReq);

	int chunkSize = 2000;
	int startLength = 4;
	int computed = 0;
	bool terminate = false;
	if (optionsMap.count("chunksize"))
	{
		chunkSize = optionsMap.at("chunksize").as<int>();
	}
	if (optionsMap.count("startlength"))
	{
		startLength = optionsMap.at("startlength").as<int>();
	}
	SequentialGenerator generator{ startLength };
	std::vector<std::string>& chunk = *new std::vector<std::string>();

	while (!terminate)
	{
		if (termReq.test())
			break;
		auto req = wait_any(_requests.begin(), _requests.end());
		switch (req.first.tag())
		{
		case WORK:
			chunk = generator.generateChunk(chunkSize);
			_comm.send(req.first.source(), MESSAGE, chunk);
			_requests.push_back(_comm.irecv(req.first.source(), WORK));
			computed += chunk.size();
			_logger->TraceTransfer("chunk sended:{0}", computed);
			break;

		case FOUND:
			terminate = true;
			break;

		default:
			break;
		}

		_requests.erase(req.second);
	}
	_logger->TraceInformation("Result Found:{0} ", _result);
}

void SimpleMaster::OnBeginRoutine()
{
	_stopWatch.Start();
	broadcast(_comm, _target, 0);
	for (int i = 1; i < _comm.size(); i++)
	{
		_requests.push_back(_comm.irecv(i, WORK));
	}
}

void SimpleMaster::Report()
{

	for (int i = 0; i < _collectedStats.size(); i++)
	{
		_logger->TraceInformation("Statistics of Process {0}\n{1}\n###########################", i, _collectedStats.at(i).ToString());
	}

}

