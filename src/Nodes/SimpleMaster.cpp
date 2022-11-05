#include "Nodes/SimpleMaster.hpp"
#include <boost/mpi/collectives.hpp>
using namespace boost::mpi;

SimpleMaster::SimpleMaster(boost::mpi::communicator comm, std::string target): _target{target},_comm{comm}
{

}

void SimpleMaster::OnEndRoutine()
{
}

void SimpleMaster::Initialize()
{
	_logger->TraceInformation() << "Broadcasting Target:" << _target << std::endl;
	broadcast(_comm, _target, _comm.rank());
}

void SimpleMaster::Routine()
{

}

void SimpleMaster::OnBeginRoutine()
{
}
