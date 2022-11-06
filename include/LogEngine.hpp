#pragma once 
#include <boost/mpi.hpp>
#include <iostream>


class MPILogEngine
{
private:
	std::istream* _loadStream;
	std::ostream* _saveStream;
	boost::mpi::communicator& _communicator;
	MPILogEngine(boost::mpi::communicator& comm, std::istream* loadStream, std::ostream* saveStream, int verbosity);
	int _verbosity;
	static MPILogEngine* _instance;
	std::ostream& log();
public:
	static void CreateInstance(boost::mpi::communicator& comm, std::istream* loadStream, std::ostream* saveStream, int verbosity = 0);
	static MPILogEngine* Instance();
	void Finalize();
	std::ostream& TraceException();
	std::ostream& TraceInformation();
	std::ostream& TraceTransfer();
	std::ostream& TraceResult();
};


