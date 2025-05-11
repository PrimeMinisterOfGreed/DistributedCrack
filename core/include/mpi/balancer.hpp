#pragma once 
#include "mpi/communicator.hpp"
#include <mpi.h>

struct Balancer{
    protected:
    Communicator& comm;

    public:
    virtual void process() = 0;
    Balancer(Communicator& comm) : comm(comm) {}
};


struct ChunkBalancer : public Balancer {
    
    public:
    ChunkBalancer(Communicator& comm) : Balancer(comm){}
    void process() override;
};

struct BruteBalancer : public Balancer {
    std::vector<std::pair<uint64_t, uint64_t>> ranges{};
    public:
    BruteBalancer(Communicator& comm): Balancer(comm){}
    void process() override;
};
