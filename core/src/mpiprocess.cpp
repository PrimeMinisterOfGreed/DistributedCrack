#include <mpi.h>
#include "mpi/mpiprocess.hpp"


MpiProcess::MpiProcess(Communicator& comm)
    : _comm(comm)
{
    
}

void MpiProcess::add_future(std::unique_ptr<MpiPromise> promise) {
    promises.push_back(std::move(promise));
}

std::unique_ptr<MpiPromise> MpiProcess::wait_any() {
    int index;
    MPI_Status status;
    MPI_Request request[promises.size()];
    for (size_t i = 0; i < promises.size(); ++i) {
        request[i] = promises[i]->request();
    }
    MPI_Waitany(promises.size(), request, &index, &status);
    auto promise = std::move(promises[index]);
    promises.erase(promises.begin() + index);
    promise->set_status(status);
    return promise;
}

void MpiProcess::wait_all() {
    MPI_Request request[promises.size()];
    for (size_t i = 0; i < promises.size(); ++i) {
        request[i] = promises[i]->request();
    }
    MPI_Waitall(promises.size(), request, MPI_STATUSES_IGNORE);
}


