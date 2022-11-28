#include "Schema.hpp"
#include "StringGenerator.hpp"
#include "Statistics/TimeMachine.hpp"
#include "md5.hpp"
#include <boost/concept_check.hpp>
#include <boost/mpi/collectives.hpp>
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/nonblocking.hpp>
#include <boost/mpi/request.hpp>
#include <boost/mpi/status.hpp>
#include <boost/serialization/vector.hpp>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>
#include <Nodes/SimpleMaster.hpp>
#include <Nodes/SimpleWorker.hpp>
#include <Nodes/SchedulerMaster.hpp>
#include <Nodes/AddressableWorker.hpp>

using namespace boost::mpi;
using namespace boost::accumulators;

using Accumulator = accumulator_set<double, features<tag::mean, tag::max, tag::min, tag::variance>>;

void delreq(std::vector<request> &workRequests, request *worker)
{

    workRequests.erase(workRequests.begin() +
                       indexOf<boost::mpi::request>(workRequests.begin(), workRequests.end(),
                                                    [&](request val) -> bool { return &val == worker; }));
}


bool compute(std::vector<std::string> &chunk, std::string target, std::string *found, std::function<std::string(std::string)> hashfnc = md5)
{
    for (auto val : chunk)
    {
        auto res = hashfnc(val);

        if (res == target)
        {
            found = &res;
            return true;
        }
    }
    return false;
}



SimpleMasterWorker::SimpleMasterWorker(int chunkSize, std::string &target) : _chunkSize(chunkSize), _target(target)
{
}

void SimpleMasterWorker::ExecuteSchema(boost::mpi::communicator& comm)
{
    try
    {
        int rank = comm.rank();
        if (rank == 0)
        {
            SimpleMaster master{ comm,_target };
            master.Execute();
        }
        else
        {
            SimpleWorker worker{ comm };
            worker.Execute();
        }
    }
    catch (...)
    {
        MPILogEngine::Instance()->TraceException("Endend for exception");
    }
}


MasterWorkerDistributedGenerator::MasterWorkerDistributedGenerator(int chunkSize, std::string &target)
    : _chunkSize(chunkSize), _target(target)
{
}

void MasterWorkerDistributedGenerator::ExecuteSchema(boost::mpi::communicator &comm)
{
    try
    {
        if (comm.rank() == 0)
        {
            SchedulerMaster master(comm, _target);
            master.Execute();
        }
        else
        {
            AddressableWorker worker(comm);
            worker.Execute();
        }
    }
    catch (...)
    {
        std::cout << "Fatal Exception" << std::endl;
    }
}
