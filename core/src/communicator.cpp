#include "mpi/communicator.hpp"


int Communicator::rank() {
    int rank;
    MPI_Comm_rank(comm, &rank);
    return rank;
}

int Communicator::size() {
    int size;
    MPI_Comm_size(comm, &size);
    return size;
}

MpiContext::MpiContext(int argc, char ** argv)
{
    MPI_Init(&argc, &argv);
}

MpiContext::~MpiContext()
{
    MPI_Finalize();
}

Communicator& MpiContext::world() {
    static Communicator world(MPI_COMM_WORLD);
    return world;
}
