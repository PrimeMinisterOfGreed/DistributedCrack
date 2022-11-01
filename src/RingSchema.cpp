#include "RingSchema.hpp"
#include "StringGenerator.hpp"
#include "md5.hpp"
#include <algorithm>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/nonblocking.hpp>
#include <boost/mpi/request.hpp>
#include <boost/mpi/status.hpp>
#include <boost/serialization/vector.hpp>
#include <functional>
#include <future>
#include <ostream>
#include <string>
#include <thread>
#include <vector>

using namespace boost::mpi;

void ComputeAsync(std::vector<std::string> &chunk, std::string &target, bool *busy,
                  std::function<void(std::string)> callBack)
{
    *busy = true;
    std::thread t{[&]() {
        for (auto value : chunk)
        {
            if (md5(value) == target)
            {
                callBack(value);
            }
            callBack("NULL");
        }
        *busy = false;
    }};
    t.detach();
}

void RingPipeLine::MasterNode(communicator &comm)
{
    using requestVector = std::vector<request>;
    broadcast(comm, _target, 0);
    int next = (comm.rank() + 1 + comm.size()) % comm.size();
    int prev = (comm.rank() - 1 + comm.size()) % comm.size();
    requestVector requests;
    std::string result;
    std::vector<int> address({0, 0});
    int actualAddress = 0;
    requests.push_back(comm.irecv(prev, WORK));
    requests.push_back(comm.irecv(prev, MESSAGE, address));
    requests.push_back(comm.irecv(prev, TERMINATE, result));
    bool terminate = false;
    std::cout << "Setup done[next: " << next << ",prev: " << prev << "]" << std::endl;
    comm.barrier();
    std::cout << "Waiting for message" << std::endl;
    while (!terminate)
    {
        auto req = wait_any(requests.begin(), requests.end());
        requests.erase(requests.begin() + indexOf<request>(requests.begin(), requests.end(),
                                                           [&](request &val) { return &val == req.second.base(); }));
        switch (req.first.tag())
        {
        case WORK:
            actualAddress += _chunkSize;
            comm.send(next, MESSAGE, *new std::vector<int>{actualAddress, _chunkSize});
            requests.push_back(comm.irecv(prev, WORK));
            break;

        case MESSAGE:
            comm.send(next, MESSAGE, address);
            break;

        case TERMINATE:
            terminate = true;
            break;
        }

        std::cout << "\033[2J\033[1;1H";
        std::cout << "Computed: " << actualAddress << std::endl;
        std::cout << "Work Requests: " << requests.size() << std::endl;
    }
    comm.barrier();
    std::cout << "Password is: " << result << std::endl;
}

void RingPipeLine::Node(communicator &comm)
{
    std::string target = "";
    std::string result = "";
    broadcast(comm, target, 0);
    int prev = (comm.rank() - 1 + comm.size()) % comm.size();
    int next = (comm.rank() + 1 + comm.size()) % comm.size();
    bool terminate = false;
    std::vector<request> requests{};
    std::vector<std::string> chunk;
    std::vector<int> address{};
    AssignedSequenceGenerator generator(4);
    requests.push_back(comm.irecv(prev, WORK));
    requests.push_back(comm.irecv(prev, MESSAGE, address));
    requests.push_back(comm.irecv(prev, TERMINATE, result));
    bool busy = false;
    std::promise<std::string> *actual = nullptr;
    comm.barrier();
    comm.send(next, WORK);
    while (!terminate)
    {
        auto req = wait_any(requests.begin(), requests.end());
        requests.erase(requests.begin() + indexOf<request>(requests.begin(), requests.end(),
                                                           [&](request &val) { return &val == req.second.base(); }));
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
                generator.AssignAddress(address.at(0));
                chunk = generator.generateChunk(address.at(1));
                ComputeAsync(chunk, target, &busy, [&](std::string value) {
                    if (value != "NULL")
                    {
                        comm.send(next, TERMINATE, value);
                    }
                    else
                    {
                        comm.send(next, WORK);
                        printf("Send to %d\n", next);
                    }
                });
            }
            else
            {
                comm.send(next, MESSAGE, address);
            }
            break;
        }
    }
    comm.barrier();
}

void RingPipeLine::ExecuteSchema(boost::mpi::communicator &comm)
{
    if (comm.rank() == 0)
        MasterNode(comm);
    else
        Node(comm);
}
