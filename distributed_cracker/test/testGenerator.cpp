#include "string_generator.hpp"
#include "md5.hpp"
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <gtest/gtest.h>
#include <string>
#include <vector>

TEST(TestGenerator, test_known_sequence) {
  AssignedSequenceGenerator generator{4};
  generator.AssignAddress(372);
  auto seq = generator.nextSequence();
  assert(seq.at(seq.size() - 1) == (char)minCharInt);
  assert(seq.at(seq.size() - 2) == (char)minCharInt + 4);
}

TEST(TestGenerator, test_sequence_increment) {
  AssignedSequenceGenerator generator{1};
  generator.AssignAddress(93);
  auto seq = generator.nextSequence();
  assert(seq.at(seq.size() - 1) == (char)minCharInt);
  assert(seq.at(seq.size() - 2) == (char)minCharInt + 1);
}



TEST(testMd5, test_md5) {
  std::string computed = md5("0000");
  printf("computed: %s\n", computed.c_str());
  printf("original %s\n", "4a7d1ed414474e4033ac29ccb8653d9b");
  assert(md5("0000") == "4a7d1ed414474e4033ac29ccb8653d9b");
}
