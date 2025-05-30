// main of gtest
#include "compute.hpp"
#include "md5.h"
#include "options.hpp"
#include "statefile.hpp"
#include "thread.hpp"
#include "timer.hpp"
#include "utils.hpp"
#include <cmath>
#include <cstdio>
#include <gtest/gtest.h>
int main(int argc, char **argv) {
  // remove before this
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

TEST(TestOptions, TestOptionsLoadFromFile) {
  // get launch.toml from CMAKE_CURRENT_PROJECT_DIR
  const char *filename = "./launch.toml";
  auto opt = options::load_from_file(filename).unwrap();
  EXPECT_STREQ(opt.config_file, filename);
  EXPECT_EQ(opt.use_gpu, true);
  EXPECT_EQ(opt.use_mpi, true);
  EXPECT_TRUE(opt.brute_mode());
}

TEST(TestGenerator, TestShift) {
  SequenceGenerator gen(4);
  size_t i = 0;
  while (gen.current() != "!!!!!") {
    gen.next_sequence();
    i++;
  }
  char buffer[512]{};
  sprintf(buffer, "Found %s at %lu\n", gen.current().c_str(), i);
  fprintf(stderr, "%s", buffer);
}

TEST(TestCompute, TestComputeBruteGpu) {
  ComputeContext::BruteContext ctx;
  memset(&ctx, 0, sizeof(ctx));
  ARGS.gpu_threads = 1000;
  ARGS.brute_start = 1;
  ARGS.use_gpu = true;
  ctx.start = pow(94, 4) - 100;
  ctx.end = pow(94, 4) + 800;
  ctx.target = "952bccf9afe8e4c04306f70f7bed6610";
 
  auto result = compute({ctx});
  EXPECT_TRUE(result.has_value());
  EXPECT_STREQ(result.value().c_str(), "!!!!!");
}

TEST(TestCompute, TestComputeChunkGpu) {
  ComputeContext::ChunkContext ctx;
  ARGS.gpu_threads = 1000;
  ARGS.brute_start = 4;
  ARGS.use_gpu = true;
  auto res = SequenceGenerator(5).generate_flatten_chunk(10000);
  ctx.data = res.strings.data();
  ctx.sizes = res.sizes.data();
  ctx.chunk_size = 10000;
  ctx.target = "98abe3a28383501f4bfd2d9077820f11";
  auto result = compute({ctx});
  EXPECT_TRUE(result.has_value());
  EXPECT_STREQ(result.value().c_str(), "!!!!");
}

TEST(TestCompute, TestComputeChunkCpu) {
  ComputeContext::ChunkContext ctx;
  ARGS.gpu_threads = 1000;
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
  ARGS.gpu_threads = 1000;
  ARGS.use_gpu = false;
  ARGS.cpu_threads = 16;
  for (int i = 3; i < 6; i++) {
    ComputeContext::BruteContext ctx;
    ARGS.brute_start = i; 
    memset(&ctx, 0, sizeof(ctx));
    char passwd[24]{};
    char passwd_res[32]{};
    char passwd_digest[16]{};
    ctx.start = pow(94, i)-1000;
    ctx.end = pow(94, i) + 1000;
    memset(passwd, '!', i);
    md5String(reinterpret_cast<char*>(passwd), reinterpret_cast<uint8_t*>(passwd_digest));
    md5HexDigest(reinterpret_cast<uint8_t*>(passwd_digest), passwd_res);
    ctx.target = passwd_res;
    auto result = compute({ctx});
    EXPECT_TRUE(result.has_value()) << "Failed to compute for length " << i+1 << " with target " << passwd_res
    <<"passwd: " << passwd;
    //EXPECT_STREQ(result.value().c_str(), passwd) << " for length " << i;
  }
}

TEST(TestGenerator, TestSequenceGenerator) {
  SequenceGenerator gen(4);
  std::string seq = gen.current();
  EXPECT_EQ(seq.length(), 4u);
  EXPECT_EQ(seq, "!!!!");
  gen.next_sequence();
  gen.skip_to(10);
  EXPECT_EQ(gen.absolute_index(), 93 * 3);

  EXPECT_NE(gen.get_buffer(), nullptr);
  auto res = gen.generate_flatten_chunk(5);
  EXPECT_EQ(res.strings.size(), 4 * 5);
  EXPECT_EQ(res.sizes.size(), 5u);
}

TEST(TestGenerator, GenerateHundredElements) {
  SequenceGenerator gen(4);
  auto res = gen.generate_flatten_chunk(100);
  // Ogni elemento ha lunghezza 4, quindi il vettore strings deve avere 400
  // elementi
  EXPECT_EQ(res.strings.size(), 400u);
  // Il vettore sizes deve avere 100 elementi
  EXPECT_EQ(res.sizes.size(), 100u);
}

TEST(TestGenerator, TestDictionaryReader) {
  DictionaryReader reader("dictionary.txt");
  auto res = reader.generate_flatten_chunk(100);
  EXPECT_EQ(res.sizes.size(), 100u);
}

TEST(TestGenerator, TestHelloPosition) {
  SequenceGenerator gen(4);
  for (uint64_t i = 0;; i++) {
    gen.next_sequence();
    if (gen.current() == "hello") {
      fprintf(stderr, "Found hello at %lu\n", i);
      break;
    }
  }
}



TEST(TestTimerStats, TestStatsToCsv) {
  TimerStats stats("test");
  stats.busy_time = 100;
  stats.observation_time = 200;
  stats.task_completed = 300;
  memcpy(stats.device_name, "test_device", 12);
  memcpy(stats.name, "test", 4);
  std::string csv = stats.to_csv();
  EXPECT_EQ(
      "device_name,context_name,busy_time,observation_time,task_completed\n",
      csv.substr(0, csv.find('\n') + 1));
  EXPECT_EQ("test_device,test,100,200,300\n", csv.substr(csv.find('\n') + 1));
}



TEST(TestThreads, test_thread_instance){
  int res = 0 ;
  Thread thread([](void* a) -> void* {
    fprintf(stderr, "Thread started\n");
    int* b = static_cast<int*>(a);
    *b = 42;
    return a;
  });
  thread.start(&res);
  thread.join();
  EXPECT_EQ(res, 42);
}


TEST(TestThreads, test_parallel_for_int) {
  std::vector<int> data(33, 0);
  parallel_for(4, [&](thread_block blk) {
    int chunk_size = ceil((double)33/ blk.n_threads);
    int index = blk.thread_id * chunk_size ;
    int next = index + chunk_size;
    fprintf(stderr, "Thread %d processing chunk from %d to %d\n", blk.thread_id, index, index + chunk_size);
    for(;index < next && index < 33; index++){
      fprintf(stderr, "Thread %d setting index %d to 1 with chunk size %d \n", blk.thread_id, index,chunk_size);
      data[index] = 1;
    }
  });
  for (const auto& value : data) {
   EXPECT_EQ(value, 1);
  }
}