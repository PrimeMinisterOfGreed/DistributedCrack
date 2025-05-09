#pragma once
#include "mpi/communicator.hpp"
#include "mpi/mpiprocess.hpp"
#include <mpi.h>

struct Worker {
    protected:
    Communicator& comm;
    public:
    virtual void process() = 0;
    Worker(Communicator& comm);
};

struct ChunkWorker : public Worker {
    public:
    ChunkWorker(MPI_Comm comm);
    void process() override;
};

struct BruteWorker : public Worker {
    public:
    BruteWorker(Communicator& comm);
    void process() override;
};