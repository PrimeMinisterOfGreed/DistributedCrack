#pragma once
#include <memory>
#include <mpi.h>
#include <vector>
#include "mpipromise.hpp"

struct Communicator;
struct MpiContext{
    MpiContext(int argc, char ** argv);
    ~MpiContext();
    Communicator& world();    
};

enum MPITags{
    ANY = MPI_ANY_TAG,
    FINISH = 1,
    DATA = 2,
    SIZE =3,
    BRUTE = 4,
    TASK = 5,
    RESULT = 6,
};

struct Communicator{
    private:
    MPI_Comm comm;
    public:
    Communicator(MPI_Comm comm): comm(comm) {}
    MPI_Comm get_comm() const { return comm; }

    template<typename T>
    T recv_object(int source, int tag);
    
    template<typename T>
    void send_object(const T& obj, int dest, int tag);

    template<typename T>
    std::vector<T> recv_vector(int source, int tag);

    template<typename T>
    void send_vector(const std::vector<T>& vec, int dest, int tag);

    template<typename T>
    std::unique_ptr<MpiPromise> irecv(int source, int tag);

    template<typename T>
    std::unique_ptr<MpiPromise> isend(const T& obj, int dest, int tag);

    template<typename T>
    std::unique_ptr<MpiPromise> irecv_vector(int source, int tag, uint32_t count);

    template<typename T>
    std::unique_ptr<MpiPromise> isend_vector(const std::vector<T>& vec, int dest, int tag);
};

/* -------------------------------------------------------------------------- */
/*                                  TEMPLATES                                 */
/* -------------------------------------------------------------------------- */




template<typename T>
inline T Communicator::recv_object(int source, int tag) {
    T obj; 
    MPI_Status status;
    MPI_Recv(&obj, sizeof(T), MPI_BYTE, source, tag, comm, &status);
    return obj;   
}

template<typename T>
inline void Communicator::send_object(const T& obj, int dest, int tag) {
    MPI_Send(&obj, sizeof(T), MPI_BYTE, dest, tag, comm);
}

template<typename T>
inline void Communicator::send_vector(const std::vector<T>& vec, int dest, int tag) {
    int size = vec.size();
    MPI_Send(vec.data(), size * sizeof(T), MPI_BYTE, dest, tag, comm);
}

template<typename T>
inline std::vector<T> Communicator::recv_vector(int source, int tag) {
    MPI_Status status;
    int size;
    MPI_Probe(source, tag, comm, &status);
    MPI_Get_count(&status, MPI_BYTE, &size);
    std::vector<T> vec(size / sizeof(T));
    MPI_Recv(vec.data(), size, MPI_BYTE, source, tag, comm, &status);
    return vec;
}


template<typename T>
inline std::unique_ptr<MpiPromise> Communicator::irecv(int source, int tag) {
    auto promise = BufferedPromise<T>(comm,1);
    promise._request = MPI_Irecv(promise.get_buffer().data(), sizeof(T), MPI_BYTE, source, tag, comm, &promise._request);
    return promise;
}

template<typename T>
inline std::unique_ptr<MpiPromise> Communicator::isend(const T& obj, int dest, int tag) {
    auto promise = BufferedPromise<T>(comm, 1);
    promise.get_buffer().push_back(obj);
    promise._request = MPI_Isend(promise.get_buffer().at(0), sizeof(T), MPI_BYTE, dest, tag, comm, &promise._request);
    promise.count = 1;
    return promise;
}

template<typename T>
inline std::unique_ptr<MpiPromise> Communicator::irecv_vector(int source, int tag, uint32_t count) {
    auto promise = BufferedPromise<T>(comm, count);
    promise._request = MPI_Irecv(promise.get_buffer().data(), count * sizeof(T), MPI_BYTE, source, tag, comm, &promise._request);
    return promise;
}

template<typename T>
inline std::unique_ptr<MpiPromise> Communicator::isend_vector(const std::vector<T>& vec, int dest, int tag) {
    
}


