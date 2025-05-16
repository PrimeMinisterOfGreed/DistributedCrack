//
// Created by drfaust on 10/02/23.
//
#include <gtest/gtest.h>

#include "cuda.h"
#include "cuda_runtime_api.h"
extern "C" {
  #include "md5.h"
  #include "md5_gpu.h"
  #include "string_generator.h"
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  auto res = RUN_ALL_TESTS();
  return res;
}

TEST(TestMd5Gpu, test_bruter){
  const char * target = "98abe3a28383501f4bfd2d9077820f11";
  const char * expected = "!!!!";
  struct Md5BruterResult result = md5_bruter(0, 10000, target, 10000, 4);
  ASSERT_TRUE(result.found);
  ASSERT_STREQ(expected, result.data);
}

TEST(TestMd5Gpu, test_bruter_2){
  const char * target = "952bccf9afe8e4c04306f70f7bed6610";
  const char * expected = "!!!!!";
  struct Md5BruterResult result = md5_bruter(0, 10000, target, 10000, 5);
  ASSERT_TRUE(result.found);
  ASSERT_STREQ(expected, result.data);
}

TEST(TestMd5Gpu, test_transformer){
  const char * target = "98abe3a28383501f4bfd2d9077820f11";
  struct SequenceGeneratorCtx ctx = new_seq_generator(4);
  char buffer[4*1000]; 
  memset(buffer, 0, 4*1000);
  uint8_t sizes[1000];
  memset(sizes, 0, 1000);
  for(int i = 0 ; i < 1000 ; i ++){
    memcpy(buffer+ i*4, ctx.buffer, 4);
    sizes[i] = 4;
    seq_gen_next_sequence(&ctx);
  }
  struct Md5TransformResult result = md5_gpu(buffer,sizes,10,10);
  char extr[33];
  memset(extr, 0, 33);
  memcpy(extr, result.data, 32);
  ASSERT_STREQ(target, extr);
}