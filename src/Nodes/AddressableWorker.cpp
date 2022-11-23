#include "Nodes/AddressableWorker.hpp"
#include <boost/mpi/collectives.hpp>
#include "OptionsBag.hpp"
#include "Schema.hpp"
using namespace boost::mpi;

void AddressableWorker::Routine()
{
	bool terminate = false;
	std::vector<size_t> address{};
	std::string* result = new std::string();
	std::vector<std::string> chunk{};
	_requests.push_back(_communicator.irecv(0, MESSAGE, address));
	char computations = 0;
	while (!terminate)
	{
		auto req = wait_any(_requests.begin(), _requests.end());
		switch (req.first.tag())
		{
		case MESSAGE:
			_logger->TraceTransfer() << "Received Address: " << address.at(0) << ",Received chunkSize: " << address.at(1) << std::endl;
			_generator->AssignAddress(address.at(0));
			chunk = _generator->generateChunk(address.at(1));
			if (Compute(chunk, result))
			{
				_communicator.send(0, FOUND, *result);
				terminate = true;
			}
			else
			{
				_requests.push_back(_communicator.irecv(0, MESSAGE, address));
				_communicator.send(0, WORK);
			}
			break;

		case TERMINATE:
			terminate = true;
			break;
		default:
			break;
		}
		DeleteRequest(req.second.base());
		computations++;
		if (computations > 100)
		{
			_logger->TraceTransfer() << "Executing statistics retrieval" << _processor.ToCompute() << std::endl;
			_processor.ComputeStatistics();
			computations = 0;
		}
	}
}

void AddressableWorker::Initialize()
{
	_logger->TraceTransfer() << "Broadcasting target" << std::endl;
	broadcast(_communicator, _target, 0);
	_logger->TraceTransfer() << "Target acquired: " << _target << std::endl;
}

void AddressableWorker::OnBeginRoutine()
{
	_logger->TraceInformation() << "Starting stopwatch" << std::endl;
	_stopWatch.Start();
	int startSequence = 4;
	broadcast(_communicator, startSequence, 0);
	_generator = new AssignedSequenceGenerator(startSequence);
	_communicator.send(0, WORK);
}

void AddressableWorker::OnEndRoutine()
{
	auto ev = _processor.ComputeStatistics();
	_communicator.send(0, MESSAGE, ev);
}
