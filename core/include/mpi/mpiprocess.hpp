#pragma once
#include <mpi.h>
#include <vector>
#include "mpipromise.hpp"



struct MpiProcess{
    private:
    std::vector<MpiPromise> promises;
    public:

    void add_future(MpiPromise promise);
};
