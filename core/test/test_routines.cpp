#include "mpi/communicator.hpp"
#include "mpi/generator.hpp"
#include "options.hpp"
#include <cstdint>
#include <gtest/gtest.h>    
#include <iostream>


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