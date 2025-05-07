#pragma once
#include <cstdint>
#include <mpi.h>
#include <vector>

struct MpiPromise {
    MPI_Status _status;
    MPI_Request _request;
    MPI_Comm comm;
    MpiPromise(MpiPromise& other) = delete;
    virtual ~MpiPromise() = default;
    virtual int wait();

    virtual int test();
    virtual int cancel();
    virtual MPI_Status& status();
    virtual MPI_Request& request();
    virtual void set_status(MPI_Status status);
    MpiPromise(MPI_Comm comm, MPI_Request request, MPI_Status status): comm(comm), _request(request), _status(status) {}
    MpiPromise(MPI_Comm comm) : comm(comm){}
};



/* -------------------------------------------------------------------------- */
/*                              Buffered Promise                              */
/* -------------------------------------------------------------------------- */

template<typename T>
struct BufferedPromise: public MpiPromise{
    private: 
    std::vector<T> buffer;
    public:
    BufferedPromise(MPI_Comm comm, uint32_t count);
    std::vector<T>& get_buffer();
};





template<typename T>
inline BufferedPromise<T>::BufferedPromise(MPI_Comm comm, uint32_t count) : MpiPromise(comm) {
    buffer.resize(count);
}

template<typename T>
inline std::vector<T>& BufferedPromise<T>::get_buffer() { return buffer; }



