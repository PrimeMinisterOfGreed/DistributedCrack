#include "LogEngine.hpp"
#include <Schema.hpp>


MPILogEngine* MPILogEngine::_instance;

std::stringstream& _trashStream = *new std::stringstream();

void SendStream(boost::mpi::communicator& comm, std::istream& stream, int processTarget)
{
	char* buffer = new char[1024];
	while (stream.readsome(buffer,1024) > 0)
	{
		comm.send(0, MESSAGE, buffer, 1024);
	}
	comm.send(processTarget, TERMINATE);
	delete[] buffer;
}

std::string& ReceiveStream(boost::mpi::communicator& comm, int process)
{
	using namespace std;
	using namespace boost::mpi;
	char* buffer = new char[1024];
	string& result = *new std::string();
	vector<request> requests{};
	requests.push_back(comm.irecv(process, TERMINATE));
	requests.push_back(comm.irecv(process, MESSAGE,buffer,1024));
	bool terminate = false;
	while (!terminate)
	{
		auto req = wait_any(requests.begin(), requests.end());
		if (req.first.tag() == TERMINATE)
		{
			terminate = true;
			break;
		}
		else if (req.first.tag() == MESSAGE)
		{
			result += buffer;
		}
	}
	return result;

}


MPILogEngine::MPILogEngine(boost::mpi::communicator& comm, std::istream* loadStream, std::ostream* saveStream, int verbosity): _loadStream{loadStream}, _saveStream{saveStream}
,_communicator{comm }, _verbosity{verbosity}
{
	
}

std::ostream& MPILogEngine::log()
{
	*_saveStream << "(Process: " << _communicator.rank() << ")";
	return *_saveStream;
}

void MPILogEngine::Finalize()
{
	if (_communicator.rank() == 0)
	{
		for (int i = 1; i < _communicator.size(); i++)
		{
			this->log() << ReceiveStream(_communicator, i);
		}
	}
	else
	{
		SendStream(_communicator, *_loadStream, 0);
	}
}

std::ostream& MPILogEngine::TraceException()
{
	auto &stream = this->log() << "[Exception] ";
	return stream;
}

std::ostream& MPILogEngine::TraceInformation()
{
	if (_verbosity > 0)
	{
		auto & stream=  this->log() << "[Information] ";
		return stream;
	}
	_trashStream.clear();
	return _trashStream;
}

std::ostream& MPILogEngine::TraceTransfer()
{
	if (_verbosity > 1)
	{
		auto & stream = this->log() << "[Transfer]";
		return stream;
	}
	_trashStream.clear();
	return _trashStream;
}

std::ostream& MPILogEngine::TraceResult()
{
	auto& stream = this->log() << "[Result]";
	return stream;
}

void MPILogEngine::CreateInstance(boost::mpi::communicator& comm, std::istream* loadStream, std::ostream* saveStream, int verbosity)
{
	MPILogEngine::_instance = new MPILogEngine(comm, loadStream, saveStream, verbosity);
}

MPILogEngine* MPILogEngine::Instance()
{
	return MPILogEngine::_instance;
}

