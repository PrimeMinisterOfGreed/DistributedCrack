#include "chunk_loader.hpp"
#include "options_bag.hpp"
#include "string_generator.hpp"
#include "md5.hpp"
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <gtest/gtest.h>
#include <string>
#include <fstream>
#include <vector>

TEST(TestGenerator, test_known_sequence) {
  AssignedSequenceGenerator generator{4};
  generator.assign_address(372);
  auto seq = generator.next_sequence();
  assert(seq.at(seq.size() - 1) == (char)minCharInt);
  assert(seq.at(seq.size() - 2) == (char)minCharInt + 4);
}

TEST(TestGenerator, test_sequence_increment) {
  AssignedSequenceGenerator generator{1};
  generator.assign_address(93);
  auto seq = generator.next_sequence();
  assert(seq.at(seq.size() - 1) == (char)minCharInt);
  assert(seq.at(seq.size() - 2) == (char)minCharInt + 1);
}



TEST(testMd5, test_md5) {
  std::string computed = md5("0000");
  printf("computed: %s\n", computed.c_str());
  printf("original %s\n", "4a7d1ed414474e4033ac29ccb8653d9b");
  assert(md5("0000") == "4a7d1ed414474e4033ac29ccb8653d9b");
}


TEST(TestChunkLoader, test_load){
  options.dictionary = "./rose.dict";
  std::ofstream dict{"rose.dict"};
  AssignedSequenceGenerator generator{4};
  auto chunk = generator.generate_chunk(10000);
  for(auto&v: chunk){
    auto write = v + "\n";
    dict.write(write.c_str(), write.size());
  }
  ChunkLoader loader{};
  auto verify = loader.get_chunk(10000);
  for(auto i = 0 ; i < verify.size() ; i++)
    ASSERT_EQ(verify[i], chunk[i]);
}

TEST(TestChunkLoader, test_load_on_buffer){
  options.dictionary = "./rose.dict";
  std::ofstream dict{"rose.dict"};
  AssignedSequenceGenerator generator{4};
  auto chunk = generator.generate_chunk(10000);
  for(auto&v: chunk){
    auto write = v + "\n";
    dict.write(write.c_str(), write.size());
  }
  ChunkLoader loader{};
  char * buffer = new char[10000*12]{};
  auto sizes = loader.get_chunk(10000,buffer);
  size_t disp = 0;
  ASSERT_EQ(sizes.size(), 10000);
  for(auto i = 0 ; i < sizes.size() ; i++){
    char str[12]{};
    memcpy(str, &buffer[disp], sizes[i]);
    ASSERT_EQ(str, chunk[i]);
    disp += sizes[i];
  }
  delete[] buffer;
}


TEST(TestChunkLoader, test_generator){
  options.dictionary = "NONE";
  ASSERT_TRUE(options.dictionary.compare("NONE") == 0);
  AssignedSequenceGenerator generator{1};
  auto chunk = generator.generate_chunk(10000);
  ChunkLoader loader{};
  auto verify = loader.get_chunk(10000);
  ASSERT_EQ(verify.size(), 10000);
  for(auto i = 0 ; i < verify.size() ; i++)
    ASSERT_EQ(verify[i], chunk[i]);
}



TEST(TestChunkLoader, test_generator_on_buffer){
  options.dictionary = "NONE";
  AssignedSequenceGenerator generator{1};
  auto chunk = generator.generate_chunk(10000);
  ChunkLoader loader{};
  char * buffer = new char[12*10000]{};
  auto sizes = loader.get_chunk(10000,buffer);
  ASSERT_EQ(sizes.size(), 10000);
  size_t disp = 0;
  for(auto i = 0 ; i < sizes.size() ; i++){
    char str[12]{};
    memcpy(str, &buffer[disp], sizes[i]);
    ASSERT_EQ(str, chunk[i]);
    disp += sizes[i];
  }
  delete[] buffer;
}
