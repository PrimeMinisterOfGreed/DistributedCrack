#include "mpi/generator.hpp"
#include <cstdint>
#include <filesystem>
#include "mpi/communicator.hpp"
#include "options.hpp"
#include <algorithm>
#include <mpi.h>
#include <sys/types.h>
ChunkedGenerator::ChunkedGenerator(Communicator& comm): Generator(comm)  , reader(ARGS.dictionary_file){
}

std::optional<std::string> ChunkedGenerator::process() {
    auto stop = _comm.irecv_vector<uint8_t>(MPI::ANY_SOURCE, RESULT, 42);
    _mpprocess.add_future(std::move(stop));
    auto request = _comm.irecv<uint16_t>(MPI::ANY_SOURCE, MPITags::TASK);    
    _mpprocess.add_future(std::move(request));
    for(;;){
        auto future = _mpprocess.wait_any();
        switch (future->_status.MPI_TAG) {
            case TASK:
            {
                auto req = static_cast<BufferedPromise<uint16_t>*>(future.get());
                auto numtask = req->get_buffer()[0];
                auto source = future->_status.MPI_SOURCE;
                auto res = reader.generate_flatten_chunk(ARGS.chunk_size*numtask);
                if(res.sizes.size() == 0){
                    return std::nullopt;
                }
                _comm.send_vector(res.sizes, source, SIZE);
                _comm.send_vector(res.strings, source, DATA);
                _mpprocess.add_future(_comm.irecv<uint8_t>(MPI_ANY_SOURCE, MPITags::TASK));
            }
            break;

            case RESULT:
            {
                auto res = static_cast<BufferedPromise<uint8_t>*>(future.get());
                auto zidx = std::find(res->get_buffer().begin(), res->get_buffer().end(), 0);
                return std::string(res->get_buffer().begin(), zidx);
            }
        }
    }
}

std::optional<std::string> BruteGenerator::process() {
    auto stop = _comm.irecv_vector<uint8_t>(MPI::ANY_SOURCE, RESULT,42);
    _mpprocess.add_future(std::move(stop));
    auto request = _comm.irecv<uint16_t>(MPI::ANY_SOURCE, TASK);    
    _mpprocess.add_future(std::move(request));
    for(;;){
        auto future = _mpprocess.wait_any();
        switch (future->_status.MPI_TAG) {
            case TASK:
            {
                auto req = static_cast<BufferedPromise<uint16_t>*>(future.get());
                auto numtask = req->get_buffer()[0];
                auto source = future->_status.MPI_SOURCE;
                uint64_t start = current_address;
                uint64_t end = start + ARGS.chunk_size*numtask;
                _comm.send_vector<uint64_t>({start,end}, source, SIZE);
                _mpprocess.add_future(_comm.irecv<uint8_t>(MPI_ANY_SOURCE, MPITags::TASK));
            }
            break;

            case RESULT:
            {
                auto res = static_cast<BufferedPromise<uint8_t>*>(future.get());
                auto zidx = std::find(res->get_buffer().begin(), res->get_buffer().end(), 0);
                return std::string(res->get_buffer().begin(), zidx);
            }
        }
    }
}

BruteGenerator::BruteGenerator(Communicator&comm) : Generator(comm)
{

}



