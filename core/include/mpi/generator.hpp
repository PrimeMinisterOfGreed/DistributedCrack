#pragma once
#include "mpi/communicator.hpp"
#include "mpi/mpiprocess.hpp"
#include "utils.hpp"
#include <mpi.h>
#include <optional>

struct Generator{
    protected:
    Communicator& _comm;
    MpiProcess _mpprocess;
    public:
    virtual std::optional<std::string> process() = 0;
    Generator(Communicator& comm): _comm(comm), _mpprocess(MpiProcess(comm)){}
};
struct ChunkedGenerator : public Generator{
    private:
    DictionaryReader reader;
    public:
    ChunkedGenerator(Communicator& comm);
    virtual std::optional<std::string> process() override;
};


struct BruteGenerator : Generator{

};