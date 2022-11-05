#include "Nodes/SimpleMaster.hpp"
#include <boost/mpi/collectives.hpp>
#include <boost/serialization/vector.hpp>
#include "Schema.hpp"
#include "StringGenerator.hpp"
#include "md5.hpp"
#include "Functions.hpp"
#include "OptionsBag.hpp"
using namespace boost::mpi;

void SimpleMaster::DeleteRequest(request* request)
{
        _requests.erase(_requests.begin() +
            indexOf<boost::mpi::request>(_requests.begin(), _requests.end(),
                [&](boost::mpi::request val) -> bool { return &val == request; }));   
}

SimpleMaster::SimpleMaster(boost::mpi::communicator comm, std::string target): _target{target},_comm{comm}
{

}

void SimpleMaster::OnEndRoutine()
{
    for (int i = 1; i < _comm.size(); i++)
        _comm.send(i, TERMINATE);
    _comm.barrier();
}

void SimpleMaster::Initialize()
{
	_logger->TraceInformation() << "Broadcasting Target:" << _target << std::endl;
	broadcast(_comm, _target, _comm.rank());
}

void SimpleMaster::Routine()
{
    using namespace boost::mpi;

    int chunkSize = 2000;
    int startLength = 4;
    if (optionsMap.count("chunksize"))
    {
        chunkSize = optionsMap.at("chunksize").as<int>();
    }
    if (optionsMap.count("startlength"))
    {
        startLength = optionsMap.at("startlength").as<int>();
    }
    SequentialGenerator generator{startLength};
    int computed = 0;

    while (_result == "")
    {
        auto worker = wait_any(_requests.begin(), _requests.end());
        auto chunk = generator.generateChunk(chunkSize);
        _comm.send(worker.first.source(), MESSAGE, chunk);
        _requests.push_back(_comm.irecv(worker.first.source(), WORK));
        computed += chunkSize;
        DeleteRequest(worker.second.base());
    }
}

void SimpleMaster::OnBeginRoutine()
{
   
    for (int i = 1; i < _comm.size(); i++)
    {
        _requests.push_back(_comm.irecv(i, WORK));
    }
    _requests.push_back(_comm.irecv(any_source, FOUND, _result));
}
