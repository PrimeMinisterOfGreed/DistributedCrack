#include "Nodes/SimpleWorker.hpp"
#include <boost/mpi/collectives.hpp>
#include <boost/serialization/vector.hpp>
#include "Schema.hpp"
using namespace boost::mpi;


void SimpleWorker::Routine()
{
	bool terminate = false;
	std::vector<std::string> chunk{};
	std::string* result = new std::string();
	_requests.push_back(_communicator.irecv(0, MESSAGE, chunk));
	while (!terminate)
	{
		auto stat = wait_any(_requests.begin(), _requests.end());
		switch (stat.first.tag())
		{
			case MESSAGE:
			if (Compute(chunk, result))
			{
				_communicator.send(0, FOUND, *result);
				terminate = true;
			}
			else
			{
				_communicator.send(0, WORK);
				_requests.push_back(_communicator.irecv(0, MESSAGE, chunk));
			}
			break;

			case TERMINATE:
			terminate = true;
			break;
			default:
			break;
		}
		DeleteRequest(&*stat.second);
	}
}

void SimpleWorker::Initialize()
{
	_logger->TraceInformation() << "Broadcasting target" << std::endl;
	broadcast(_communicator, _target, 0);
}

void SimpleWorker::OnBeginRoutine()
{
	_stopWatch.Start();
	_communicator.send(0, WORK);
	_requests.push_back(_communicator.irecv(0, TERMINATE));
}

void SimpleWorker::OnEndRoutine()
{
	auto stat = _processor.ComputeStatistics();
	_communicator.send(0, MESSAGE, stat);
}

SimpleWorker::SimpleWorker(boost::mpi::communicator comm) : MPINode(comm,"NOTARGET")
{
}
