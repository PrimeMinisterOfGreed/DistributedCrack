// main of gtest
#include <gtest/gtest.h>
#include "options.hpp"
#include "compute.hpp"
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}


TEST(TestOptions, TestOptionsLoadFromFile) {
    // get launch.toml from CMAKE_CURRENT_PROJECT_DIR
    const char * filename = "./launch.toml";
    auto opt = options::load_from_file(filename).unwrap();
    EXPECT_STREQ(opt.config_file, filename);
    EXPECT_EQ(opt.use_gpu, true);
    EXPECT_EQ(opt.use_mpi, false);
}

TEST(TestCompute, TestComputeBruteGpu) {
    ComputeContext::BruteContext ctx;
    ARGS.num_threads = 1000;
    ARGS.brute_start = 4;
    ARGS.use_gpu = true;
    ctx.start = 0;
    ctx.end = 1000000;
    ctx.target = "98abe3a28383501f4bfd2d9077820f11";
    auto result = compute({ctx});
    EXPECT_TRUE(result.has_value());
    EXPECT_STREQ(result.value().c_str(), "!!!!");
}

TEST(TestCompute, TestComputeChunkGpu) {
    ComputeContext::ChunkContext ctx;
    ARGS.num_threads = 1000;
    ARGS.brute_start = 4;
    ARGS.use_gpu = true;
    ctx.data = (uint8_t *)malloc(1000000);
    ctx.sizes = (uint8_t *)malloc(1000000);
    ctx.chunk_size = 1000000;
    ctx.target = "98abe3a28383501f4bfd2d9077820f11";
    auto result = compute({ctx});
    EXPECT_TRUE(result.has_value());
    EXPECT_STREQ(result.value().c_str(), "98abe3a28383501f4bfd2d9077820f11");
}