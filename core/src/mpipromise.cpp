#include "mpi/mpipromise.hpp"   

int MpiPromise::wait() {
    int flag = 0;
    MPI_Wait(&_request, &_status);
    MPI_Get_count(&_status, MPI_BYTE, &flag);
    return flag;
}

int MpiPromise::test() 
    {
        int flag = 0;
        MPI_Test(&_request, &flag, &_status);
        if (flag) {
            MPI_Get_count(&_status, MPI_BYTE, &flag);
        }
        return flag;
    }


int MpiPromise::cancel() {
    return MPI_Cancel(&_request);
}



MPI_Status& MpiPromise::status(){
    return _status;
}