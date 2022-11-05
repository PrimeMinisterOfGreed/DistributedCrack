#include "LogEngine.hpp"
#include <Schema.hpp>

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
	string result = "";
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


MPILogEngine::MPILogEngine(boost::mpi::communicator& comm, std::istream* loadStream, std::ostream* saveStream): _loadStream{loadStream}, _saveStream{saveStream}
,_communicator{comm }
{
	
}

void MPILogEngine::CreateInstance(boost::mpi::communicator& comm, std::istream* loadStream, std::ostream* saveStream)
{
	_instance = new MPILogEngine(comm, loadStream, saveStream);
}

MPILogEngine& MPILogEngine::Instance()
{
	return *_instance;
}

std::ostream& MPILogEngine::log()
{
	return *_saveStream;
}

void MPILogEngine::finalize()
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

std::ostream& logStream()
{
	return MPILogEngine::Instance().log();
}
