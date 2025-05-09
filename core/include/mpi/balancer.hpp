#pragma once 
#include <mpi.h>

struct Balancer{
    protected:
    MPI_Comm comm;

    public:
    virtual void process() = 0;
    Balancer(MPI_Comm comm) : comm(comm) {}
};


struct ChunkBalancer : public Balancer {
    
    public:
    ChunkBalancer(MPI_Comm comm);
    void process() override;
};

struct BruteBalancer : public Balancer {
    
    public:
    BruteBalancer(MPI_Comm comm);
    void process() override;
};