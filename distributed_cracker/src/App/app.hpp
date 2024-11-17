#pragma once
#include <vector>
#include <mpi.h>
struct Module{

};


struct MpiApp{
    private:
    std::vector<Module*> _modules{};

    public:
    MpiApp(MPI::Comm& comm);        
    
};