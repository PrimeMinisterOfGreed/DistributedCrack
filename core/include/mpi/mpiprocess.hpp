#pragma once
#include <mpi.h>
#include <vector>
#include "mpi/communicator.hpp"
#include "mpipromise.hpp"



struct MpiProcess{
    private:
    std::vector<std::unique_ptr<MpiPromise>> promises;
    Communicator& _comm;
    public:
    MpiProcess(Communicator& comm);
    void add_future(std::unique_ptr<MpiPromise> promise);
    std::unique_ptr<MpiPromise> wait_any();
    void wait_all();
};
