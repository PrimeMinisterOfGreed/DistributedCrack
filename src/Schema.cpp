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


bool compute(std::vector<std::string> &chunk, std::string target, std::string *found)
{
    for (auto val : chunk)
    {
        auto res = md5(val);

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

void SimpleMasterWorker::ExecuteSchema(boost::mpi::communicator &comm)
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
    catch (std::exception e)
    {
        std::cout << e.what() << std::endl;
    }
}

void MasterWorkerDistributedGenerator::Master(communicator &comm)
{
    std::vector<request> requests{};
    std::string found = "";
    std::string target = md5(_target);
    uint64_t actualAddress = 0;
    broadcast(comm, target, 0);
    auto receivedFound = comm.irecv(any_source, FOUND, found);
    requests.push_back(receivedFound);
    for (int i = 1; i < comm.size(); i++)
        requests.push_back(comm.irecv(i, WORK));
    std::cout << "Setup done initiating brute" << std::endl;
    while (found == "")
    {
        auto worker = wait_any(requests.begin(), requests.end());
        if (worker.first.tag() == FOUND)
        {
            std::cout << "Password: " << found << std::endl;
            found = "1";
            break;
        }
        else if (worker.first.tag() == WORK)
        {

            comm.send(worker.first.source(), MESSAGE, std::vector<uint64_t>{actualAddress, (uint64_t)_chunkSize});
            actualAddress += _chunkSize - 1;
        }
        std::cout << "\033[2J\033[1;1H";
        std::cout << "Computed: " << actualAddress << std::endl;
        std::cout << "Work Requests: " << requests.size() << std::endl;
        delreq(requests, worker.second.base());
        requests.push_back(comm.irecv(worker.first.source(), WORK));
    }
    for (int i = 1; i < comm.size(); i++)
        comm.send(i, TERMINATE);
    comm.barrier();
    std::cout << "Found: " << found << std::endl;
}

void MasterWorkerDistributedGenerator::Worker(communicator &comm)
{
    AssignedSequenceGenerator generator{4};
    std::string target = "";
    broadcast(comm, target, 0);
    comm.send(0, WORK);
    bool found = false;
    std::vector<request> reqs{};
    reqs.push_back(comm.irecv(0, TERMINATE));
    while (!found)
    {
        std::vector<uint64_t> addr(2);
        std::string foundStr = "";
        reqs.push_back(comm.irecv(0, MESSAGE, addr));
        auto worker = wait_any(reqs.begin(), reqs.end());
        if (worker.first.tag() == TERMINATE)
        {
            found = true;
            break;
        }
        else if (worker.first.tag() == MESSAGE)
            generator.AssignAddress(addr.at(0));
        auto chunk = generator.generateChunk((int)addr.at(1));
        bool computed = compute(chunk, target, &foundStr);
        if (computed)
        {
            comm.send(0, FOUND, foundStr);
            found = true;
            break;
        }
        delreq(reqs, worker.second.base());
        comm.send(0, WORK);
        chunk.clear();
        addr.clear();
    }
    comm.barrier();
}
