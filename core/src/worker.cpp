#include "mpi/worker.hpp"
#include "compute.hpp"
#include "mpi/communicator.hpp"
#include "mpi/mpiprocess.hpp"
#include "options.hpp"
#include <cstdint>
#include <mpi.h>
#include <vector>
#include <vector>

Worker::Worker(Communicator& comm)
    : comm(comm)
{
    
}




BruteWorker::BruteWorker(Communicator& comm) 
    : Worker(comm)
{
    
}

void BruteWorker::process() {
    auto process = MpiProcess(comm);
    auto stop = comm.irecv<uint8_t>(MPI_ANY_SOURCE, TERMINATE);
    auto task = comm.irecv_vector<uint64_t>(MPI_ANY_SOURCE, SIZE,2);
    process.add_futures(std::move(stop),std::move(task));
    for(;;){
        auto promise =process.wait_any();
        switch (promise->status().MPI_TAG) {
            case TERMINATE:
                return;
            case SIZE: {
                auto task = static_cast<BufferedPromise<uint64_t>*>(promise.get())->get_buffer();
                auto result = task[0] + task[1];
                auto ctx = ComputeContext::BruteContext{.start = task[0], .end = task[1], .target = ARGS.target_md5};
                auto res = compute({ctx});
                if(res.has_value()){
                    comm.send_vector<uint8_t>({res.value().begin(),res.value().end()}, promise->status().MPI_SOURCE, RESULT);
                }
                break;
            }
            default:
                break;
        
        }
    }

}

