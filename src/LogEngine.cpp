#include "LogEngine.hpp"
#include "MPIMessageType.hpp"


MPILogEngine* MPILogEngine::_instance;

std::stringstream& _trashStream = *new std::stringstream();

std::string& LogTypeToString(LogType logType)
{
	switch (logType)
	{
	case LogType::EXCEPTION:
		return *new std::string("[Exception]");
	case LogType::RESULT:
		return *new std::string("[Result]");
	case LogType::INFORMATION:
		return *new std::string("[Information]");
	case LogType::TRANSFER:
		return *new std::string("[Transfer]");
	case LogType::DEBUG:
		return *new std::string("[Debug]");
	}
}

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

void MPILogEngine::Trace(LogType type, std::string& message)
{
	if (_verbosity >= (int)type)
		this->log() << LogTypeToString(type) << message << std::endl;
}



void MPILogEngine::CreateInstance(boost::mpi::communicator& comm, std::istream* loadStream, std::ostream* saveStream, int verbosity)
{
	MPILogEngine::_instance = new MPILogEngine(comm, loadStream, saveStream, verbosity);
}

ILogEngine* MPILogEngine::Instance()
{
	return MPILogEngine::_instance;
}


void ConsoleLogEngine::Finalize()
{
	
}

void ConsoleLogEngine::Trace(LogType type, std::string& message)
{
	if(_verbosity >= (int)type)
		printf("%s%s\n", LogTypeToString(type).c_str(), message.c_str());
}
