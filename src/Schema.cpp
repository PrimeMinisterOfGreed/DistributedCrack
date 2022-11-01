#include "Schema.hpp"
#include "StringGenerator.hpp"
#include "TimeMachine.hpp"
#include "md5.hpp"
#include <boost/concept_check.hpp>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/nonblocking.hpp>
#include <boost/mpi/request.hpp>
#include <boost/mpi/status.hpp>
#include <boost/serialization/vector.hpp>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

using namespace boost::mpi;

void delreq(std::vector<request> &workRequests, request *worker)
{
    /*for (int i = 0; i < workRequests.size(); i++)
        if (&workRequests.at(i) == worker)
        {
            workRequests.erase(workRequests.begin() + i);
            return;
        }*/

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
        Master(comm);
    else
        Worker(comm);
}

void SimpleMasterWorker::Master(boost::mpi::communicator &comm)
{

    using namespace boost::mpi;
    SequentialGenerator generator{4};
    std::string target = md5(_target);
    std::string found = "";
    std::cout << target << std::endl;
    auto request = comm.irecv(boost::mpi::any_source, FOUND, found);
    std::vector<boost::mpi::request> workRequests{};
    workRequests.push_back(request);
    int computed = 0;
    for (int i = 1; i < comm.size(); i++)
    {
        workRequests.push_back(comm.irecv(i, WORK));
        comm.send(i, MESSAGE, target);
    }
    while (found == "")
    {
        auto worker = wait_any(workRequests.begin(), workRequests.end());
        if (worker.first.tag() == FOUND)
            break;
        auto chunk = generator.generateChunk(_chunkSize);
        comm.send(worker.first.source(), MESSAGE, chunk);
        workRequests.push_back(comm.irecv(worker.first.source(), WORK));
        computed += _chunkSize;
        std::cout << "\033[2J\033[1;1H";
        std::cout << "Computed: " << computed << std::endl;
        std::cout << "Current Combination Length: " << generator.GetCurrentSequenceLength() << std::endl;
        std::cout << "Work Requests: " << workRequests.size() << std::endl;
        std::cout << "First Chunk Word: " << chunk.at(0) << std::endl;
        delreq(workRequests, worker.second.base());
    }
    for (int i = 1; i < comm.size(); i++)
        comm.send(i, TERMINATE);
    comm.barrier();
    std::cout << "Password Found: " << found << std::endl;
}

void SimpleMasterWorker::Worker(boost::mpi::communicator &comm)
{
    using namespace boost::mpi;
    std::vector<std::string> chunk{};
    std::string target = "";
    comm.recv(0, MESSAGE, target);
    std::vector<request> requests{};
    requests.push_back(comm.irecv(0, TERMINATE));
    bool terminate = false;
    std::cout << "Listening for terminate message: " << std::endl;
    comm.send(0, WORK);
    while (!terminate)
    {
        std::string found;
        requests.push_back(comm.irecv(0, MESSAGE, chunk));
        auto res = wait_any(requests.begin(), requests.end());
        if (res.first.tag() == TERMINATE)
            terminate = true;
        else
        {
            if (res.first.tag() == MESSAGE)
            {
                bool computed = compute(chunk, target, &found);
                if (computed)
                {
                    terminate = true;
                    comm.send(0, FOUND, found);
                }
                if (!terminate)
                {
                    comm.send(0, WORK);
                }
            }
        }
        chunk.clear();
        delreq(requests, res.second.base());
    }
    comm.barrier();
}

MasterWorkerDistributedGenerator::MasterWorkerDistributedGenerator(int chunkSize, std::string &target)
    : _chunkSize(chunkSize), _target(target)
{
}

void MasterWorkerDistributedGenerator::ExecuteSchema(boost::mpi::communicator &comm)
{
    if (comm.rank() == 0)
        Master(comm);
    else
        Worker(comm);
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
