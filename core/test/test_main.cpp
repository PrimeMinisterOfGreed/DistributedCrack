// main of gtest
#include <gtest/gtest.h>
#include "options.hpp"
#include "compute.hpp"
#include "statefile.hpp"
#include "utils.hpp"
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
    EXPECT_EQ(opt.use_mpi, true);
    EXPECT_TRUE(opt.brute_mode());
}

TEST(TestCompute, TestComputeBruteGpu) {
    ComputeContext::BruteContext ctx;
    memset(&ctx, 0, sizeof(ctx));
    ARGS.num_threads = 1000;
    ARGS.brute_start = 4;
    ARGS.use_gpu = true;
    ctx.start = 0u;
    ctx.end = 10000u;
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
    auto res = SequenceGenerator(4).generate_flatten_chunk(10000);
    ctx.data = res.strings.data();
    ctx.sizes = res.sizes.data();
    ctx.chunk_size = 10000;
    ctx.target = "98abe3a28383501f4bfd2d9077820f11";
    auto result = compute({ctx});
    EXPECT_TRUE(result.has_value());
    EXPECT_STREQ(result.value().c_str(), "!!!!");
}

TEST(TestCompute, TestComputeChunkCpu){
    ComputeContext::ChunkContext ctx;
    ARGS.num_threads = 1000;
    ARGS.brute_start = 4;
    ARGS.use_gpu = false;
    auto res = SequenceGenerator(4).generate_flatten_chunk(10000);
    ctx.data = res.strings.data();
    ctx.sizes = res.sizes.data();
    ctx.chunk_size = 10000;
    ctx.target = "98abe3a28383501f4bfd2d9077820f11";
    auto result = compute({ctx});
    EXPECT_TRUE(result.has_value());
    EXPECT_STREQ(result.value().c_str(), "!!!!");
}

TEST(TestCompute, TestComputeBruteCpu) {
    ComputeContext::BruteContext ctx;
    memset(&ctx, 0, sizeof(ctx));
    ARGS.num_threads = 1000;
    ARGS.brute_start = 4;
    ARGS.use_gpu = false;
    ctx.start = 0u;
    ctx.end = 10000u;
    ctx.target = "98abe3a28383501f4bfd2d9077820f11";
    auto result = compute({ctx});
    EXPECT_TRUE(result.has_value());
    EXPECT_STREQ(result.value().c_str(), "!!!!");
}

TEST(TestGenerator, TestSequenceGenerator) {
    SequenceGenerator gen(4);
    std::string seq = gen.current();
    EXPECT_EQ(seq.length(), 4u);
    EXPECT_EQ(seq, "!!!!");
    gen.next_sequence();
    gen.skip_to(10);
    EXPECT_EQ(gen.absolute_index(), 93*3);

    EXPECT_NE(gen.get_buffer(), nullptr);
    auto res = gen.generate_flatten_chunk(5);
    EXPECT_EQ(res.strings.size(), 4 * 5);
    EXPECT_EQ(res.sizes.size(), 5u);
}

TEST(TestGenerator, GenerateHundredElements) {
    SequenceGenerator gen(4);
    auto res = gen.generate_flatten_chunk(100);
    // Ogni elemento ha lunghezza 4, quindi il vettore strings deve avere 400 elementi
    EXPECT_EQ(res.strings.size(), 400u);
    // Il vettore sizes deve avere 100 elementi
    EXPECT_EQ(res.sizes.size(), 100u);
}




TEST(TestGenerator, TestDictionaryReader) {
    DictionaryReader reader("dictionary.txt");
    auto res = reader.generate_flatten_chunk(100);
    EXPECT_EQ(res.sizes.size(), 100u);


}


TEST(TestGenerator, TestHelloPosition){
    SequenceGenerator gen(4);
    for(uint64_t i = 0; ; i++){
        gen.next_sequence();
        if(gen.current() == "hello"){
            fprintf(stderr, "Found hello at %lu\n", i);
            break;
        }
    }
}

TEST(TestSaveFile, TestSerialization){
    StateFile state;
    state.current_address = 0;
    strcpy(state.current_dictionary, "test");
    const char *filename = "test_statefile.dat";
    state.save(filename);
    auto loaded_state = StateFile::load(filename);
    EXPECT_EQ(loaded_state->current_address, state.current_address);
    EXPECT_STREQ(loaded_state->current_dictionary, state.current_dictionary);
}