#include "RingSchema.hpp"
#include "md5.hpp"
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/nonblocking.hpp>
#include <future>
#include <string>
#include <thread>
#include <vector>

using namespace boost::mpi;

std::future<std::string> ComputeAsync(std::vector<std::string> &chunk, std::string &target, bool *busy)
{
    *busy = true;
    std::promise<std::string> &promise = *new std::promise<std::string>();
    std::thread t{[&]() {
        for (auto value : chunk)
        {
            if (md5(value) == target)
            {
                promise.set_value(value);
            }
        }
        promise.set_value("NULL");
        *busy = false;
    }};
    t.detach();
    return promise.get_future();
}

RingPipeLine::RingPipeLine(int chunkSize, std::string &target) : _chunkSize(chunkSize), _target(target)
{
}

void RingPipeLine::MasterNode(communicator &comm)
{
}

void RingPipeLine::Node(communicator &comm)
{
    std::string target = "";
    std::string result = "";
    broadcast(comm, target, 0);
    int prev = (comm.rank() - 1) % comm.size();
    int next = (comm.rank() + 1) % comm.size();
    bool terminate = false;
    std::vector<request> requests{};
    std::vector<std::string> chunk;
    requests.push_back(comm.irecv(prev, WORK));
    requests.push_back(comm.irecv(prev, MESSAGE, chunk));
    requests.push_back(comm.irecv(prev, TERMINATE, result));
    bool busy = false;
    std::promise<std::string> *actual = nullptr;
    comm.send(next, WORK);
    while (!terminate)
    {
        auto req = wait_any(requests.begin(), requests.end());
        switch (req.first.tag())
        {
        case WORK:
            comm.send(next, WORK);
            break;

        case TERMINATE:
            terminate = true;
            comm.send(next, TERMINATE, result);
            break;

        case MESSAGE:
            if (!busy)
            {
                actual = ComputeAsync(chunk, target, &busy);
            }
            break;
        }
    }
}
