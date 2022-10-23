#include "Schema.hpp"
#include "StringGenerator.hpp"
#include "md5.hpp"
#include <TimeMachine.hpp>
#include <boost/mpi/communicator.hpp>
#include <iostream>
#include <mpi.h>
int main(int argc, char *argv[])
{

    using namespace boost::mpi;
    MPI_Init(&argc, &argv);
    auto &comm = *new communicator();
    auto time = executeTimeComparison([&]() {
        MasterWorkerDistributedGenerator schema{1000, *new std::string("0000")};
        schema.ExecuteSchema(comm);
    });
    if (comm.rank() == 0)
        std::cout << "Executed in time: " << time.count() << "ms" << std::endl;
    MPI_Finalize();
}