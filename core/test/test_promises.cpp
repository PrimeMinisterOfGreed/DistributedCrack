#include "mpi/communicator.hpp"
#include "mpi/mpiprocess.hpp"
#include "mpi/mpipromise.hpp"
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <iostream>
#include <memory>
#include <ompi/mpi/cxx/mpicxx.h>

TEST(TestPromise, TestSimplePromise){
    MpiContext ctx{1,NULL};
    auto comm = ctx.world();
    if(comm.rank() == 0){
        auto promise = comm.irecv<uint8_t>(1, 99);
        promise->wait();
        auto obj = reinterpret_cast<BufferedPromise<uint8_t>*>(promise.get())->get_buffer()[0];
        EXPECT_EQ(obj, 10u);
    }
    else{
        comm.send_object<uint8_t>(10u, 0, 99);
    }
}

TEST(TestPromise, TestVectorPromise){
    MpiContext ctx{1,NULL};
    auto comm = ctx.world();
    if(comm.rank() == 0){
        auto promise = comm.irecv_vector<uint8_t>(1, 99, 10);
        promise->wait();
        auto obj = reinterpret_cast<BufferedPromise<uint8_t>*>(promise.get())->get_buffer();
        for(int i=0; i<10; i++){
            EXPECT_EQ(obj[i], i);
        }
    }
    else{
        std::vector<uint8_t> vec(10);
        for(int i=0; i<10; i++){
            vec[i] = i;
        }
        comm.send_vector<uint8_t>(vec, 0, 99);
    }
}

TEST(TestPromise, TestMpiProcessWaitAny){
    MpiContext ctx{1,NULL};
    auto comm = ctx.world();
    if(comm.rank() == 0){
        MpiProcess process{comm};
    process.add_futures(comm.irecv<uint8_t>(1, 99),
        comm.irecv<uint8_t>(1, 98),
        comm.irecv<uint8_t>(1, 97),
        comm.irecv<uint8_t>(1, 96));
        auto promise = process.wait_any();
        auto obj = reinterpret_cast<BufferedPromise<uint8_t>*>(promise.get())->get_buffer()[0];
        EXPECT_EQ(obj, 10u);

        process.add_future(comm.irecv<uint8_t>(1, 99));
        promise = process.wait_any();
        obj = reinterpret_cast<BufferedPromise<uint8_t>*>(promise.get())->get_buffer()[0];
        EXPECT_EQ(obj, 20u);
    }
    else{
        comm.send_object<uint8_t>(10, 0, 99);
        comm.send_object<uint8_t>(20, 0, 99);

    }
}


TEST(TestPromise, TestIRecvWithSendObject){
    MpiContext ctx{1,NULL};
    auto comm = ctx.world();
    if(comm.rank() == 0){
        auto promise = comm.irecv<uint16_t>(1, 99);
        promise->wait();
        auto obj = reinterpret_cast<BufferedPromise<uint16_t>*>(promise.get())->get_buffer()[0];
        EXPECT_EQ(obj, 10u);
    }
    else{
        comm.send_object<uint16_t>(10, 0, 99);
    }
}