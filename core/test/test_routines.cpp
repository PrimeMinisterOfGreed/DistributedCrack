#include "mpi/balancer.hpp"
#include "mpi/communicator.hpp"
#include "mpi/generator.hpp"
#include "mpi/worker.hpp"
#include "options.hpp"
#include <cstdint>
#include <gtest/gtest.h>    
#include <iostream>
#include <string>
#include <thread>


TEST(TestRoutines, TestChunkedGenerator){
    MpiContext ctx{1,NULL};
    ARGS.chunk_size= 10;
    ARGS.dictionary_file= (char*)"./dictionary.txt";
    
    auto comm = ctx.world();
    if(comm.rank() == 0){
        ChunkedGenerator generator{comm};
        auto res = generator.process();
        EXPECT_EQ(res.has_value(), true);
        EXPECT_EQ(res.value(), "hello");
    }
    else{
        comm.send_object<uint16_t>(100, 0, TASK);
        comm.recv_vector<uint8_t>(0, SIZE);
        comm.recv_vector<uint8_t>(0, DATA);
        comm.send_vector<uint8_t>({'h','e','l','l','o'}, 0, RESULT);
    }

}

TEST(TestRoutines, TestBruteGenerator){
    MpiContext ctx{1,NULL};
    ARGS.brute_start= 4;
    ARGS.chunk_size = 100;    
    auto comm = ctx.world();
    if(comm.rank() == 0){
        BruteGenerator generator{comm};
        auto res = generator.process();
        EXPECT_EQ(res.has_value(), true);
        EXPECT_EQ(res.value(), "hello");
    }
    else{
        comm.send_object<uint16_t>(100, 0, TASK);
        auto received = comm.recv_vector<uint64_t>(0, SIZE);
        comm.send_vector<uint8_t>({'h','e','l','l','o'}, 0, RESULT);
        EXPECT_EQ(received[0], 0);
        EXPECT_EQ(received[1], 100*100);
    }
}

TEST(TestRoutines, TestBruteWorker){
    MpiContext ctx{1,NULL};
    ARGS.brute_start= 4;
    ARGS.chunk_size = 100;    
    ARGS.gpu_threads = 100;
    ARGS.target_md5 = (char*)"98abe3a28383501f4bfd2d9077820f11";
    auto comm = ctx.world();
    if(comm.rank() == 0){
        BruteWorker worker{comm};
        worker.process();
    }
    else{
        comm.send_vector<uint64_t>({0,1000}, 0, SIZE);
        auto res = comm.recv_vector<uint8_t>(0, RESULT);
        comm.send_object<uint8_t>(0, 0, TERMINATE);
        EXPECT_EQ(std::string(res.begin(),res.end()), "!!!!");
    }
}

TEST(TestMpi, TestMultipleMessageConcept){
    MpiContext ctx{1,NULL};
    auto comm = ctx.world();
    if(comm.rank() == 0){
        for(int i = 0 ; i < 1000; i++){
            auto promise = comm.irecv<uint8_t>(MPI_ANY_SOURCE, TASK);
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            promise->wait();
        }
    }
    else{
        for(int i = 0 ; i < 1000; i++){
            comm.send_object<uint8_t>(i, 0, TASK);
                }
    }
}

TEST(TestRoutines, TestBruteBalancer){
    MpiContext ctx{1,NULL};
    ARGS.brute_start= 4;
    ARGS.chunk_size = 100;    
    ARGS.gpu_threads = 100;
    ARGS.cluster_degree = 1;
    ARGS.target_md5 = (char*)"98abe3a28383501f4bfd2d9077820f11";
    auto comm = ctx.world();
    if(comm.rank() == 1){
        BruteBalancer balancer{comm};
        balancer.process();
    }
    else{
        comm.send_object<uint8_t>(0, 1, TASK);
        auto response = comm.recv_vector<uint16_t>(1, TASK);
        EXPECT_EQ(response.size(), 1);
        comm.send_vector<uint64_t>({0,ARGS.chunk_size}, 1, SIZE);
        auto received = comm.recv_vector<uint64_t>(1, SIZE);
        EXPECT_EQ(received[0], 0);
        EXPECT_EQ(received[1], ARGS.chunk_size);
        comm.send_object<uint8_t>(0, 1, TERMINATE);
    }
}

#ifdef ENABLE_CUDA

TEST(TestResources, test_scheduling){
    
}



#endif  