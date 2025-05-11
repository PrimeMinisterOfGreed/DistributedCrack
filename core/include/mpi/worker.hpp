#pragma once
#include "mpi/communicator.hpp"
#include "mpi/mpiprocess.hpp"
#include <mpi.h>

struct Worker {
    protected:
    Communicator& comm;
    int balancer_id = 0;
    public:
    virtual void process() = 0;
    Worker(Communicator& comm) : comm(comm) {}
    Worker(Communicator& comm, int balancer_id) : comm(comm), balancer_id(balancer_id) {}
};

struct ChunkWorker : public Worker {
    public:
    ChunkWorker(Communicator& comm): Worker(comm){}
    ChunkWorker(Communicator& comm, int balancer_id): Worker(comm, balancer_id){}
    void process() override;
};

struct BruteWorker : public Worker {
    public:
    BruteWorker(Communicator& comm): Worker(comm){}
    BruteWorker(Communicator& comm, int balancer_id): Worker(comm, balancer_id){}
    void process() override;
};