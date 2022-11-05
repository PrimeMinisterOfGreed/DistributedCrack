#include <boost/mpi.hpp>
#include <iostream>
class MPILogEngine
{
private:
	std::istream* _loadStream;
	std::ostream* _saveStream;
	boost::mpi::communicator& _communicator;
	MPILogEngine(boost::mpi::communicator& comm, std::istream* loadStream, std::ostream* saveStream);
	static MPILogEngine* _instance;
public:
	static void CreateInstance(boost::mpi::communicator& comm, std::istream* loadStream, std::ostream* saveStream);
	static MPILogEngine& Instance();
	std::ostream& log();
	void finalize();

};

std::ostream& logStream();