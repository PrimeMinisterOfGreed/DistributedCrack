#include <gtest/gtest.h>
#include "md5.hpp"
#include "string_generator.hpp"
int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

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

