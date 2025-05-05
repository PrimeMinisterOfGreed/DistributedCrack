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
    virtual int wait(){
        int flag = 0;
        MPI_Wait(&_request, &_status);
        MPI_Get_count(&_status, MPI_BYTE, &flag);
        return flag;
    }

    virtual int test(){
        int flag = 0;
        MPI_Test(&_request, &flag, &_status);
        if (flag) {
            MPI_Get_count(&_status, MPI_BYTE, &flag);
        }
        return flag;
    }
    virtual int cancel(){
        return MPI_Cancel(&_request);
    }
    virtual MPI_Status& status() {
        return _status;
    }
    virtual MPI_Request& request(){
        return _request;
    }
    virtual void set_status(MPI_Status status) = 0;
    MpiPromise(MPI_Comm comm, MPI_Request request, MPI_Status status): comm(comm), _request(request), _status(status) {}
    MpiPromise(MPI_Comm comm) : comm(comm){}
};


template<typename T>
struct MpiFuture: public MpiPromise{

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
    int cancel() override;

    void set_status(MPI_Status status) override;
    std::vector<T>& get_buffer() { return buffer; }
};





template<typename T>
inline BufferedPromise<T>::BufferedPromise(MPI_Comm comm, uint32_t count) : MpiPromise(comm) {
    buffer.resize(count);
}


