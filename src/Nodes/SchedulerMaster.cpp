#include "Nodes/SchedulerMaster.hpp"
#include <boost/mpi/collectives.hpp>
#include "Schema.hpp"
#include "OptionsBag.hpp"
using namespace boost::mpi;

void SchedulerMaster::Routine()
{
	bool terminate = false;
	size_t currentAddress = 0;
	int chunkSize = 2000;
	if (optionsMap.count("chunksize"))
		chunkSize = optionsMap.at("chunksize").as<int>();
	while (!terminate)
	{
		auto req = wait_any(_requests.begin(), _requests.end());
		switch (req.first.tag())
		{
			case WORK:
			_logger->TraceTransfer() << "Work request received from: " << req.first.source();
			_communicator.send(req.first.source(), MESSAGE, std::vector<size_t>({ currentAddress, (size_t)chunkSize }));
			_requests.push_back(_communicator.irecv(req.first.source(), WORK));
			currentAddress += chunkSize;
			_logger->TraceTransfer() << "Current Address: " << currentAddress << std::endl;
			break;

			case FOUND:
			terminate = true;
			break;


			default:
			break;
		}
		DeleteRequest(req.second.base());
	}
	_logger->TraceResult() << "Found password: " << _result << std::endl;
}

void SchedulerMaster::Initialize()
{
	_logger->TraceInformation() << "Broadcasting target" << std::endl;
	broadcast(_communicator, _target, 0);
}

void SchedulerMaster::OnBeginRoutine()
{
	_requests.push_back(_communicator.irecv(any_source, FOUND,_result));
	for (int i = 1; i < _communicator.size(); i++)
	{
		_requests.push_back(_communicator.irecv(i, WORK));
	}
	int startSequence = 4;
	if (optionsMap.count("startlength"))
	{
		startSequence = optionsMap.at("startlength").as<int>();
	}

	broadcast(_communicator, startSequence, 0);
}

void SchedulerMaster::OnEndRoutine()
{
	for (int i = 1; i < _communicator.size(); i++)
	{
		_communicator.send(i, TERMINATE);
	}
}
