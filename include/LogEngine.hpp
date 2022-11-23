#pragma once 
#include <boost/mpi.hpp>
#include <iostream>


class ILogEngine
{
public:
	virtual void Finalize() = 0;
	virtual std::ostream& TraceException() = 0;
	virtual std::ostream& TraceInformation() = 0;
	virtual std::ostream& TraceTransfer() = 0;
	virtual std::ostream& TraceResult() = 0;
};

class ConsoleLogEngine : public ILogEngine
{

public:
	// Ereditato tramite ILogEngine
	virtual void Finalize() override;
	virtual std::ostream& TraceException() override;
	virtual std::ostream& TraceInformation() override;
	virtual std::ostream& TraceTransfer() override;
	virtual std::ostream& TraceResult() override;
};

class MPILogEngine: public ILogEngine
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
	virtual void Finalize() override;
	virtual std::ostream& TraceException() override;
	virtual std::ostream& TraceInformation() override;
	virtual std::ostream& TraceTransfer() override;
	virtual std::ostream& TraceResult() override;
};


